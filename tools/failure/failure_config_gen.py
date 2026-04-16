#!/usr/bin/env python
"""Compute CP threshold, update failure_handling.json, export demo video, and compute VLM checkpoint verification.

1. Finds the model folder from record_config.json (pretrained_path).
2. Finds the training dataset folder from the model's train_config.json.
3. Checks if <pretrained_path>/vlm_checkpoints.json exists:
   - If NOT: runs the Gemini VLM pipeline to identify safe checkpoint timestamps per episode,
     then saves verified_cp_ts_by_ep to that file.
   - If YES: loads verified_cp_ts_by_ep from that file (skipping VLM entirely).
4. Computes ACT transformer encoder_out features for all checkpoints.
5. Saves encoder_out_mean, encoder_out_kde, encoder_out_ep_matrix, encoder_out_flat, cp_step to
   <pretrained_path>/checkpoint_features.npz.
   Key convention: {feature_type}_{template_mode} (e.g. encoder_out_mean, encoder_out_kde),
                   {feature_type}_flat for flat/tensor strategies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from offline_utils import load_failure_metrics_jsonl
from vlm_pipeline import (
    VLM_CHECKPOINTS_FILENAME,
    load_vlm_checkpoints,
    run_vlm_pipeline,
    save_vlm_checkpoints,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute CP threshold, create/update failure_handling.json, export demo video, "
            "and compute VLM checkpoint verification npz."
        )
    )
    parser.add_argument(
        "--repo_id", type=str, required=True, help="Eval repo id, e.g. eval/eval_failure_metrics"
    )
    parser.add_argument("--alpha", type=float, default=0.01, help="Miscoverage level alpha. Default 0.01.")
    parser.add_argument("--fps", type=int, default=240, help="Output demo video FPS. Default: 240")
    parser.add_argument("--scale", type=float, default=0.5, help="Output demo video scale. Default: 0.5")
    parser.add_argument("--force", action="store_true", help="Force overwrite demo video.")
    parser.add_argument(
        "--force_vlm",
        action="store_true",
        help="Force re-run VLM pipeline and overwrite vlm_checkpoints.json.",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot").expanduser(),
        help="Lerobot cache root. Default: ~/.cache/huggingface/lerobot",
    )
    parser.add_argument("--trim_episode_frames", type=int, default=30)
    # VLM args
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model name for VLM analysis.",
    )
    parser.add_argument(
        "--vlm_workers", type=int, default=1, help="Number of concurrent VLM workers. Default: 1"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# CP threshold computation
# ---------------------------------------------------------------------------


def compute_cp_threshold(calibration_scores: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(calibration_scores)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(calibration_scores, q_level))
    return cp_threshold, q_level, n


def load_temporal_disagreement(metrics_path: Path, trim: int = 30) -> np.ndarray:
    failure_metrics = load_failure_metrics_jsonl(metrics_path.parent)
    if not failure_metrics:
        raise FileNotFoundError(f"failure_metrics.jsonl not found or empty: {metrics_path}")

    grouped: dict[int, list[dict]] = {}
    for row in failure_metrics.values():
        ep = int(row.get("episode", 0))
        grouped.setdefault(ep, []).append(row)

    scores = []
    for ep in sorted(grouped):
        ep_rows = grouped[ep]
        if trim > 0:
            if len(ep_rows) <= 2 * trim:
                continue
            ep_rows = ep_rows[trim:-trim]
        for row in ep_rows:
            value = row.get("td_raw", row.get("temporal_disagreement"))
            if value is not None and isinstance(value, (int, float)) and np.isfinite(value):
                scores.append(float(value))

    if not scores:
        raise ValueError(f"No valid td_raw values in {metrics_path}")
    return np.array(scores, dtype=np.float64)


def compute_regime_calibration_data(dataset: LeRobotDataset, metrics_path: Path) -> dict:
    """Offline phase: Group actions into regimes and compute baseline mu, sigma for each regime."""
    from tqdm import tqdm

    failure_metrics = load_failure_metrics_jsonl(metrics_path.parent)
    regime_stats: dict[str, list[float]] = {"0": [], "1": []}

    print(f"\n[Calibration] Extracting regime actions for {len(failure_metrics)} steps...")

    # Try to access underlying tabular dataset directly to avoid extremely slow video decoding
    use_hf_fast = hasattr(dataset, "hf_dataset")
    if use_hf_fast:
        print("  -> Using fast tabular dataset access (bypassing video decoding)")

    for row in tqdm(failure_metrics.values(), desc="  Regime Calib", leave=False):
        global_step = row.get("global_step")
        if global_step is None:
            continue
        global_step = int(global_step)

        td_raw = row.get("td_raw", row.get("temporal_disagreement"))
        if td_raw is None:
            continue

        try:
            if use_hf_fast:
                curr_ep = dataset.hf_dataset[global_step]["episode_index"]
                curr_act = np.array(dataset.hf_dataset[global_step]["action"])
                if global_step > 0:
                    prev_ep = dataset.hf_dataset[global_step - 1]["episode_index"]
                    if prev_ep == curr_ep:
                        prev_act = np.array(dataset.hf_dataset[global_step - 1]["action"])
                    else:
                        prev_act = curr_act
                else:
                    prev_act = curr_act
            else:
                curr_item = dataset[global_step]
                curr_ep = curr_item["episode_index"]
                if global_step > 0:
                    prev_item = dataset[global_step - 1]
                    # Fallback to curr_item if crossing an episode boundary
                    if prev_item["episode_index"] != curr_ep:
                        prev_item = curr_item
                else:
                    prev_item = curr_item

                curr_act = curr_item["action"].numpy()
                prev_act = prev_item["action"].numpy()
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError):
            continue

        # Simple regime classifier based on action difference
        act_diff = np.linalg.norm(curr_act - prev_act)
        regime = "1" if act_diff > 0.1 else "0"

        regime_stats[regime].append(float(td_raw))

    calibration_data = {}
    for regime, tides in regime_stats.items():
        if not tides:
            calibration_data[regime] = {"mean": 0.0, "std": 1.0}
        else:
            calibration_data[regime] = {"mean": float(np.mean(tides)), "std": float(np.std(tides)) + 1e-6}
    return calibration_data


def compute_cusum_maxima(
    dataset: LeRobotDataset, metrics_path: Path, calibration_data: dict, trim: int = 30
) -> np.ndarray:
    """Simulate CUSUM tracking and return the maximum C_t for each episode."""
    from tqdm import tqdm

    failure_metrics = load_failure_metrics_jsonl(metrics_path.parent)
    grouped: dict[int, list[dict]] = {}
    for row in failure_metrics.values():
        ep = int(row.get("episode", 0))
        grouped.setdefault(ep, []).append(row)

    use_hf_fast = hasattr(dataset, "hf_dataset")
    maxima = []

    decay_lambda = 0.95
    k = 1.0

    print(f"\n[Calibration] Simulating CUSUM for {len(grouped)} episodes to find maxima...")

    for ep in tqdm(sorted(grouped), desc="  CUSUM Maxima", leave=False):
        ep_rows = grouped[ep]
        if trim > 0:
            if len(ep_rows) <= 2 * trim:
                continue
            ep_rows = ep_rows[trim:-trim]

        c_t = 0.0
        m_i = 0.0

        for row in ep_rows:
            global_step = row.get("global_step")
            if global_step is None:
                continue
            global_step = int(global_step)
            td_raw = float(row.get("td_raw", row.get("temporal_disagreement", 0.0)))

            try:
                if use_hf_fast:
                    curr_ep = dataset.hf_dataset[global_step]["episode_index"]
                    curr_act = np.array(dataset.hf_dataset[global_step]["action"])
                    if global_step > 0:
                        prev_ep = dataset.hf_dataset[global_step - 1]["episode_index"]
                        if prev_ep == curr_ep:
                            prev_act = np.array(dataset.hf_dataset[global_step - 1]["action"])
                        else:
                            prev_act = curr_act
                    else:
                        prev_act = curr_act
                else:
                    curr_item = dataset[global_step]
                    curr_ep = curr_item["episode_index"]
                    if global_step > 0:
                        prev_item = dataset[global_step - 1]
                        if prev_item["episode_index"] != curr_ep:
                            prev_item = curr_item
                    else:
                        prev_item = curr_item
                    curr_act = curr_item["action"].numpy()
                    prev_act = prev_item["action"].numpy()
            except (IndexError, KeyError, TypeError, ValueError, RuntimeError):
                curr_act, prev_act = np.zeros(1), np.zeros(1)

            act_diff = np.linalg.norm(curr_act - prev_act)
            regime = "1" if act_diff > 0.1 else "0"
            calib = calibration_data.get(regime, {"mean": 0.0, "std": 1.0})

            n_tide = max(0.0, (td_raw - calib["mean"]) / calib["std"])
            c_t = max(0.0, decay_lambda * c_t + n_tide - k)
            m_i = max(m_i, c_t)

        maxima.append(m_i)

    if not maxima:
        raise ValueError(f"No valid CUSUM maxima computed in {metrics_path}")
    return np.array(maxima, dtype=np.float64)


# ---------------------------------------------------------------------------
# Encoder-out feature computation
# ---------------------------------------------------------------------------


def compute_checkpoint_features(
    training_dataset: LeRobotDataset,
    ep_df,
    verified_cp_ts_by_ep: dict[int, list[int]],
    pretrained_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute encoder_out features for all VLM-verified checkpoints.

    Hooks ``model.encoder`` output (seq_len, B, d) and mean-pools all
    observation tokens → feature per step.  Works for any policy that
    exposes ``model.encoder`` (ACT, ACT-FM, …).

    ACT prepends a latent token at index 0 (zeros at inference); ACT-FM has
    no latent token.  We detect this via ``config.use_vae`` so no tokens are
    accidentally dropped.

    Each episode contributes n checkpoint timestamps (sorted). Features are
    grouped by checkpoint index and averaged across episodes, yielding a
    (n_slots, d) matrix.

    Returns:
        mean_feat_matrix:   (n_slots, d) mean feature per slot.
        ep_feat_matrix:     (n_ep, n_slots, d) per-episode per-slot features.
        all_feat_vectors:   (N, d) all individual vectors, L2-normalised.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    print(f"\n[EncoderOut] Loading policy from {pretrained_path}...")
    cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    policy = get_policy_class(cfg.type).from_pretrained(str(pretrained_path))
    policy.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(pretrained_path))

    has_latent = bool(getattr(policy.config, "use_vae", False))

    enc_buf: list[torch.Tensor] = []  # (B, d) per forward pass

    def _enc_hook(_module, _input, output):
        # Skip the latent token at index 0 for ACT (use_vae=True); ACT-FM has none.
        obs_tokens = output[1:] if has_latent else output
        enc_buf.append(obs_tokens.mean(dim=0).detach())  # (B, d)

    h = policy.model.encoder.register_forward_hook(_enc_hook)

    episode_to_from_idx = {
        int(ep_row["episode_index"]): int(ep_row["dataset_from_index"]) for _, ep_row in ep_df.iterrows()
    }

    n_slots = max(len(v) for v in verified_cp_ts_by_ep.values())
    feat_by_slot: list[list[torch.Tensor]] = [[] for _ in range(n_slots)]
    feat_by_ep: dict[int, list[torch.Tensor]] = {}

    print(
        f"[EncoderOut] Extracting features for {sum(len(v) for v in verified_cp_ts_by_ep.values())} checkpoints "
        f"({n_slots} slots × {len(verified_cp_ts_by_ep)} episodes)..."
    )

    try:
        with torch.no_grad():
            for ep_id, ts_list in tqdm(sorted(verified_cp_ts_by_ep.items()), desc="Episodes"):
                if not ts_list:
                    continue
                from_idx = episode_to_from_idx.get(ep_id)
                if from_idx is None:
                    tqdm.write(f"  [WARN] ep {ep_id} not found in dataset, skipping.")
                    continue
                ep_feats: list[torch.Tensor] = []
                for slot_idx, ts in enumerate(sorted(ts_list)):
                    global_step = from_idx + ts
                    if global_step >= len(training_dataset):
                        tqdm.write(
                            f"  [WARN] ep {ep_id} ts={ts} → global_step={global_step} out of range, skipping."
                        )
                        continue

                    enc_buf.clear()
                    item = training_dataset[global_step]
                    batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
                    batch_proc = preprocessor(batch)
                    batch_proc = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_proc.items()
                    }
                    policy.predict_action_chunk(batch_proc)

                    if not enc_buf:
                        tqdm.write(f"  [WARN] ep {ep_id} ts={ts}: encoder_out not captured.")
                        continue

                    feat = enc_buf[0].mean(dim=0).cpu()  # (d,) — mean over batch
                    feat_by_slot[slot_idx].append(feat)
                    ep_feats.append(feat)
                if len(ep_feats) == n_slots:
                    feat_by_ep[ep_id] = ep_feats
    finally:
        h.remove()

    if not any(feat_by_slot):
        raise RuntimeError("No encoder_out features extracted. Check verified_cp_ts_by_ep contents.")

    mean_feat_rows = [torch.stack(feats).mean(dim=0) for feats in feat_by_slot if feats]
    encoder_out_mean = torch.stack(mean_feat_rows)  # (n_slots, d)

    encoder_out_ep_matrix = torch.stack(
        [torch.stack(feat_by_ep[ep_id]) for ep_id in sorted(feat_by_ep)]
    )  # (n_ep, n_slots, d)

    # All individual vectors, L2-normalised for fast dot-product similarity at inference
    import torch.nn.functional as F  # noqa: N812

    all_feat_vecs = torch.cat([torch.stack(feats) for feats in feat_by_slot if feats], dim=0)  # (N, d)
    encoder_out_flat = F.normalize(all_feat_vecs, dim=1)  # (N, d), ||v||=1

    print("\n[EncoderOut] Checkpoint feature stats:")
    print(f"  n_slots: {encoder_out_mean.shape[0]}")
    print(f"  feature_dim: {encoder_out_mean.shape[1]}")
    print(f"  encoder_out_ep_matrix shape: {tuple(encoder_out_ep_matrix.shape)}")
    print(f"  encoder_out_flat shape: {tuple(encoder_out_flat.shape)}")

    return (
        encoder_out_mean.numpy().astype(np.float32),
        encoder_out_ep_matrix.numpy().astype(np.float32),
        encoder_out_flat.numpy().astype(np.float32),
    )


def compute_backbone_features(
    training_dataset: LeRobotDataset,
    ep_df,
    verified_cp_ts_by_ep: dict[int, list[int]],
    pretrained_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ResNet backbone features for all VLM-verified checkpoints.

    Runs GAP over each camera's feature map and averages across cameras,
    yielding a (C,) vector per checkpoint step.

    Returns:
        backbone_mean:      (n_slots, C) mean feature per slot.
        backbone_ep_matrix: (n_ep, n_slots, C) per-episode per-slot features.
        backbone_flat:      (N, C) all individual vectors, L2-normalised.
    """
    import torch.nn.functional as F  # noqa: N812

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    print(f"\n[Backbone] Loading policy from {pretrained_path}...")
    cfg = PreTrainedConfig.from_pretrained(str(pretrained_path))
    policy = get_policy_class(cfg.type).from_pretrained(str(pretrained_path))
    policy.eval()
    backbone = policy.model.backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(pretrained_path))
    camera_keys = training_dataset.meta.camera_keys

    episode_to_from_idx = {
        int(ep_row["episode_index"]): int(ep_row["dataset_from_index"]) for _, ep_row in ep_df.iterrows()
    }

    n_slots = max(len(v) for v in verified_cp_ts_by_ep.values())
    feat_by_slot: list[list[torch.Tensor]] = [[] for _ in range(n_slots)]
    feat_by_ep: dict[int, list[torch.Tensor]] = {}

    print(
        f"[Backbone] Extracting features for {sum(len(v) for v in verified_cp_ts_by_ep.values())} checkpoints "
        f"({n_slots} slots × {len(verified_cp_ts_by_ep)} episodes)..."
    )
    with torch.no_grad():
        for ep_id, ts_list in tqdm(sorted(verified_cp_ts_by_ep.items()), desc="Backbone Episodes"):
            if not ts_list:
                continue
            from_idx = episode_to_from_idx.get(ep_id)
            if from_idx is None:
                tqdm.write(f"  [WARN] ep {ep_id} not found in dataset, skipping.")
                continue
            ep_feats: list[torch.Tensor] = []
            for slot_idx, ts in enumerate(sorted(ts_list)):
                global_step = from_idx + ts
                if global_step >= len(training_dataset):
                    tqdm.write(
                        f"  [WARN] ep {ep_id} ts={ts} → global_step={global_step} out of range, skipping."
                    )
                    continue
                item = training_dataset[global_step]
                batch = {k: v.unsqueeze(0) for k, v in item.items() if isinstance(v, torch.Tensor)}
                batch_proc = preprocessor(batch)
                cam_vecs: list[torch.Tensor] = []
                for cam_key in camera_keys:
                    img = batch_proc[cam_key].to(device)
                    if img.ndim == 5:
                        img = img.squeeze(1)
                    feat_map = backbone(img)["feature_map"]
                    gap_vec = F.adaptive_avg_pool2d(feat_map, 1).squeeze(-1).squeeze(-1)
                    cam_vecs.append(gap_vec)
                step_feat = torch.stack(cam_vecs, dim=0).mean(dim=0).squeeze(0).cpu()
                feat_by_slot[slot_idx].append(step_feat)
                ep_feats.append(step_feat)
            if len(ep_feats) == n_slots:
                feat_by_ep[ep_id] = ep_feats

    if not any(feat_by_slot):
        raise RuntimeError("No backbone features extracted. Check verified_cp_ts_by_ep contents.")

    mean_feat_rows = [torch.stack(feats).mean(dim=0) for feats in feat_by_slot if feats]
    backbone_mean = torch.stack(mean_feat_rows)  # (n_slots, C)

    backbone_ep_matrix = torch.stack(
        [torch.stack(feat_by_ep[ep_id]) for ep_id in sorted(feat_by_ep)]
    )  # (n_ep, n_slots, C)

    all_feat_vecs = torch.cat([torch.stack(feats) for feats in feat_by_slot if feats], dim=0)  # (N, C)
    backbone_flat = F.normalize(all_feat_vecs, dim=1)  # (N, C), ||v||=1

    print("\n[Backbone] Checkpoint feature stats:")
    print(f"  n_slots: {backbone_mean.shape[0]}")
    print(f"  feature_dim: {backbone_mean.shape[1]}")
    print(f"  backbone_ep_matrix shape: {tuple(backbone_ep_matrix.shape)}")
    print(f"  backbone_flat shape: {tuple(backbone_flat.shape)}")

    return (
        backbone_mean.numpy().astype(np.float32),
        backbone_ep_matrix.numpy().astype(np.float32),
        backbone_flat.numpy().astype(np.float32),
    )


def compute_kde_features(
    ep_feat_matrix: np.ndarray, use_pca: bool = True, variance_retained: float = 0.95
) -> np.ndarray:
    """For each checkpoint slot, select the episode embedding with the highest KDE density.

    Args:
        ep_feat_matrix: (n_ep, n_slots, d) feature matrix.
        use_pca: Whether to reduce dimensionality with PCA before KDE fitting.
        variance_retained: PCA variance retention ratio (only used when use_pca=True).

    Returns:
        kde_feat_matrix: (n_slots, d) — one representative embedding per slot.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KernelDensity

    n_ep, n_slots, d = ep_feat_matrix.shape
    kde_rows: list[np.ndarray] = []

    for slot_idx in range(n_slots):
        embeddings = ep_feat_matrix[:, slot_idx, :]  # (n_ep, d)

        if use_pca and d > 10:
            pca = PCA(n_components=variance_retained)
            data_to_fit = pca.fit_transform(embeddings)
        else:
            data_to_fit = embeddings

        # Silverman's rule: h = sigma * n^(-1/(d+4))
        n, d = data_to_fit.shape
        sigma = float(np.mean(data_to_fit.std(axis=0)))
        bandwidth = sigma * (n ** (-1.0 / (d + 4)))
        print(f"  [KDE] slot {slot_idx}: auto bandwidth={bandwidth:.6f}, pca_dims={d}, n_ep={n}")

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(data_to_fit)
        log_density = kde.score_samples(data_to_fit)
        best_idx = int(np.argmax(log_density))
        print(
            f"  [KDE] slot {slot_idx}: best episode index={best_idx}, log_density={log_density[best_idx]:.4f}"
        )

        kde_rows.append(embeddings[best_idx])

    return np.stack(kde_rows).astype(np.float32)  # (n_slots, d)


# ---------------------------------------------------------------------------
# Checkpoint image export
# ---------------------------------------------------------------------------


def export_checkpoint_images(
    training_dataset: LeRobotDataset,
    ep_df,
    verified_cp_ts_by_ep: dict[int, list[int]],
    out_dir: Path,
) -> None:
    """Save training-dataset frames for each VLM-verified checkpoint.

    Directory layout::

        out_dir/
          slot_0/   ← all episodes' first checkpoint
            ep0_ts42_observation.images.top.png
            ep1_ts38_observation.images.top.png
          slot_1/
            ...

    Args:
        out_dir: Parent directory (e.g. ``pretrained_path / "checkpoint_images"``).
    """
    camera_keys = training_dataset.meta.camera_keys

    episode_to_from_idx = {
        int(ep_row["episode_index"]): int(ep_row["dataset_from_index"]) for _, ep_row in ep_df.iterrows()
    }

    n_slots = max(len(v) for v in verified_cp_ts_by_ep.values())
    for i in range(n_slots):
        (out_dir / f"slot_{i}").mkdir(parents=True, exist_ok=True)

    print(f"\n[Images] Exporting checkpoint images → {out_dir}")
    for ep_id, ts_list in tqdm(sorted(verified_cp_ts_by_ep.items()), desc="Episodes"):
        if not ts_list:
            continue
        from_idx = episode_to_from_idx.get(ep_id)
        if from_idx is None:
            tqdm.write(f"  [WARN] ep {ep_id} not found in dataset, skipping.")
            continue
        for slot_idx, ts in enumerate(sorted(ts_list)):
            global_step = from_idx + ts
            if global_step >= len(training_dataset):
                tqdm.write(f"  [WARN] ep {ep_id} ts={ts} → global_step={global_step} out of range, skipping.")
                continue
            item = training_dataset[global_step]
            slot_dir = out_dir / f"slot_{slot_idx}"
            cam_imgs: list[np.ndarray] = []
            for cam_key in camera_keys:
                if cam_key not in item:
                    continue
                img_tensor = item[cam_key]
                # Handle optional time dimension: (T, C, H, W) → (C, H, W)
                if img_tensor.ndim == 4:
                    img_tensor = img_tensor[0]
                # Convert to uint8 HWC
                if img_tensor.is_floating_point():
                    img_np = (img_tensor.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
                else:
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                cam_imgs.append(img_np)
            if not cam_imgs:
                continue
            # Resize all to the same height then concatenate horizontally
            target_h = cam_imgs[0].shape[0]
            resized = []
            for img in cam_imgs:
                if img.shape[0] != target_h:
                    w = int(img.shape[1] * target_h / img.shape[0])
                    img = np.array(Image.fromarray(img).resize((w, target_h), Image.BILINEAR))
                resized.append(img)
            stitched = np.concatenate(resized, axis=1)
            fname = slot_dir / f"ep{ep_id}_ts{ts}.png"
            Image.fromarray(stitched).save(fname)

    print(f"[Images] Done. {sum(len(v) for v in verified_cp_ts_by_ep.values())} frames saved.")


# ---------------------------------------------------------------------------
# failure_handling.json helpers
# ---------------------------------------------------------------------------


def _save_failure_handling_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ---------------------------------------------------------------------------
# Training dataset discovery
# ---------------------------------------------------------------------------


def resolve_training_dataset_root(pretrained_path: Path, cache_root: Path) -> tuple[Path, str]:
    train_config_path = pretrained_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config.json not found at {train_config_path}")
    with train_config_path.open() as f:
        train_config = json.load(f)
    training_repo_id: str = train_config["dataset"]["repo_id"]
    root = train_config["dataset"].get("root")
    dataset_root = Path(root).expanduser() if root else cache_root / training_repo_id
    return dataset_root, training_repo_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    repo_dir = args.cache_root / args.repo_id
    metrics_path = repo_dir / "failure_metrics.jsonl"
    record_config_path = repo_dir / "meta" / "record_config.json"

    # ---- Compute Regime Calibration ----
    eval_dataset = LeRobotDataset(args.repo_id, root=repo_dir)
    if eval_dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        eval_dataset.meta.episodes = load_episodes(eval_dataset.root)
    calibration_data = compute_regime_calibration_data(eval_dataset, metrics_path)
    print(f"\n[Calibration] Computed regime calibration data: {calibration_data}")

    # ---- CP threshold (original td_raw) ----
    td_raw_scores = load_temporal_disagreement(metrics_path, trim=args.trim_episode_frames)
    td_threshold, td_q_level, td_n = compute_cp_threshold(td_raw_scores, args.alpha)

    # ---- CP threshold (CUSUM maxima) ----
    calibration_scores = compute_cusum_maxima(
        eval_dataset, metrics_path, calibration_data, trim=args.trim_episode_frames
    )
    cp_threshold, q_level, n = compute_cp_threshold(calibration_scores, args.alpha)

    print(f"\nrepo_id: {args.repo_id}")
    print(f"samples (n_episodes): {n} / (n_steps): {td_n}")
    print(f"alpha: {args.alpha}")
    print(f"trim_episode_frames: {args.trim_episode_frames}")
    print(f"q_level (C_t): {q_level} / (td): {td_q_level}")
    print(f"cusum_threshold: {cp_threshold}")
    print(f"td_threshold: {td_threshold}")

    # ---- Resolve pretrained_path ----
    if not record_config_path.exists():
        print(f"❌ record_config.json not found at {record_config_path}. Cannot proceed.")
        return
    with record_config_path.open() as f:
        record_config = json.load(f)
    policy_cfg = record_config.get("policy", {})
    pretrained_path_str = policy_cfg.get("pretrained_path") or record_config.get("pretrained_path")
    if not pretrained_path_str:
        print("❌ pretrained_path not found in record_config.json.")
        return
    pretrained_path = Path(pretrained_path_str).expanduser()

    # ---- Write failure_handling.json (always overwrite completely) ----
    failure_handling_path = pretrained_path / "failure_handling.json"
    config = {
        "enable_failure_handling": False,
        "enable_logging": True,
        "flush_metrics_every_step": False,
        "detector": {
            "failure_threshold": float(td_threshold),
            "cusum_threshold": float(cp_threshold),
            "td_smoothing_sigma": 4.0,
            "td_window_size": 31,
            "td_rho": 1.0,
            "calibration_data": calibration_data,
        },
        "strategy": {
            "name": "checkpoint_flat",
            "feature_type": "encoder_out",
            "template_mode": "kde",
            "peak_timestep_threshold": 30,
        },
        "perturbation": {
            "enabled": False,
            "fn": "gripper_gaussian_noise",
            "std": 0.02,
        },
        "plugins": [{"name": "attention_entropy", "enabled": True}],
    }
    _save_failure_handling_config(failure_handling_path, config)
    print(
        f"\n[Config] Wrote failure_handling.json (failure_threshold={cp_threshold:.6f}) → {failure_handling_path}"
    )

    # ---- Find training dataset ----
    print(f"\n[Checkpoint] Resolving training dataset from {pretrained_path / 'train_config.json'}...")
    training_dataset_root, training_repo_id = resolve_training_dataset_root(pretrained_path, args.cache_root)
    print(f"[Checkpoint] training_repo_id: {training_repo_id}")
    print(f"[Checkpoint] training_dataset_root: {training_dataset_root}")

    training_dataset = LeRobotDataset(training_repo_id, root=training_dataset_root)
    if training_dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        training_dataset.meta.episodes = load_episodes(training_dataset.root)

    import pandas as pd

    ep_parquet_dir = training_dataset_root / "meta" / "episodes"
    ep_df = pd.concat([pd.read_parquet(p) for p in sorted(ep_parquet_dir.glob("chunk-*"))])
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
    print(f"[Checkpoint] Training dataset: {len(ep_df)} episodes, fps={training_dataset.fps}")

    # ---- VLM checkpoint cache ----
    vlm_cache_path = pretrained_path / VLM_CHECKPOINTS_FILENAME
    verified_cp_ts_by_ep = load_vlm_checkpoints(pretrained_path)

    if verified_cp_ts_by_ep is not None and not args.force_vlm:
        print(f"\n[VLM] Cache found at {vlm_cache_path}, skipping VLM pipeline.")
        print(f"[VLM] Loaded {len(verified_cp_ts_by_ep)} episodes from cache.")
    else:
        if args.force_vlm and verified_cp_ts_by_ep is not None:
            print(f"\n[VLM] --force_vlm set, re-running VLM pipeline (overwriting {vlm_cache_path}).")
        print(f"\n[VLM] No cache at {vlm_cache_path}, running VLM pipeline...")
        verified_cp_ts_by_ep = run_vlm_pipeline(
            training_dataset=training_dataset,
            training_dataset_root=training_dataset_root,
            ep_df=ep_df,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            vlm_workers=args.vlm_workers,
        )
        save_vlm_checkpoints(pretrained_path, verified_cp_ts_by_ep)

    # ---- Filter verified_cp_ts_by_ep to mode checkpoint count ----
    from statistics import mode as stat_mode

    cp_counts = [len(v) for v in verified_cp_ts_by_ep.values() if v]
    modal_count = stat_mode(cp_counts)
    filtered = {ep: ts for ep, ts in verified_cp_ts_by_ep.items() if len(ts) == modal_count}
    bad_eps = {ep: ts for ep, ts in verified_cp_ts_by_ep.items() if len(ts) != modal_count}
    n_dropped = len(bad_eps)
    print(
        f"\n[Filter] Checkpoint count distribution: { {k: cp_counts.count(k) for k in sorted(set(cp_counts))} }"
    )
    print(f"[Filter] Mode = {modal_count}, keeping {len(filtered)} episodes, dropping {n_dropped}.")
    verified_cp_ts_by_ep = filtered

    # ---- Compute encoder_out features ----
    npz_path = pretrained_path / "checkpoint_features.npz"
    encoder_out_mean, encoder_out_ep_matrix, encoder_out_flat = compute_checkpoint_features(
        training_dataset=training_dataset,
        ep_df=ep_df,
        verified_cp_ts_by_ep=verified_cp_ts_by_ep,
        pretrained_path=pretrained_path,
    )

    # ---- Compute KDE features (encoder_out) ----
    print("\n[KDE] Computing KDE-selected checkpoint features...")
    encoder_out_kde = compute_kde_features(encoder_out_ep_matrix)  # (n_slots, d)
    print(f"[KDE] encoder_out_kde shape: {encoder_out_kde.shape}")

    # ---- Compute backbone features ----
    backbone_mean, backbone_ep_matrix, backbone_flat = compute_backbone_features(
        training_dataset=training_dataset,
        ep_df=ep_df,
        verified_cp_ts_by_ep=verified_cp_ts_by_ep,
        pretrained_path=pretrained_path,
    )

    # ---- Compute KDE features (backbone) ----
    print("\n[KDE] Computing KDE-selected backbone checkpoint features...")
    backbone_kde = compute_kde_features(backbone_ep_matrix)  # (n_slots, C)
    print(f"[KDE] backbone_kde shape: {backbone_kde.shape}")

    # ---- Save features to npz ----
    # Key convention: {feature_type}_{template_mode} for list-based strategies,
    #                 {feature_type}_flat for flat/tensor strategies,
    #                 {feature_type}_ep_matrix for raw per-episode data.
    np.savez(
        npz_path,
        encoder_out_mean=encoder_out_mean,  # (n_slots, d)
        encoder_out_kde=encoder_out_kde,  # (n_slots, d), KDE-selected representative
        encoder_out_ep_matrix=encoder_out_ep_matrix,  # (n_ep, n_slots, d), raw per-episode
        encoder_out_flat=encoder_out_flat,  # (N, d), all individual L2-normalised vectors
        backbone_mean=backbone_mean,  # (n_slots, C)
        backbone_kde=backbone_kde,  # (n_slots, C), KDE-selected representative
        backbone_ep_matrix=backbone_ep_matrix,  # (n_ep, n_slots, C), raw per-episode
        backbone_flat=backbone_flat,  # (N, C), all individual L2-normalised vectors
        cp_step=verified_cp_ts_by_ep,
    )
    print(f"\n[Features] Saved checkpoint_features.npz to {npz_path}")

    verify = np.load(npz_path, allow_pickle=True)
    print(f"  Arrays in npz: {verify.files}")
    print(f"  encoder_out_mean shape: {verify['encoder_out_mean'].shape}")
    print(f"  encoder_out_kde shape: {verify['encoder_out_kde'].shape}")
    print(f"  encoder_out_ep_matrix shape: {verify['encoder_out_ep_matrix'].shape}")
    print(f"  encoder_out_flat shape: {verify['encoder_out_flat'].shape}")
    print(f"  backbone_mean shape: {verify['backbone_mean'].shape}")
    print(f"  backbone_kde shape: {verify['backbone_kde'].shape}")
    print(f"  backbone_ep_matrix shape: {verify['backbone_ep_matrix'].shape}")
    print(f"  backbone_flat shape: {verify['backbone_flat'].shape}")

    # ---- Export checkpoint images ----
    img_root = pretrained_path / "checkpoint_images"
    export_checkpoint_images(
        training_dataset=training_dataset,
        ep_df=ep_df,
        verified_cp_ts_by_ep=verified_cp_ts_by_ep,
        out_dir=img_root,
    )
    if bad_eps:
        export_checkpoint_images(
            training_dataset=training_dataset,
            ep_df=ep_df,
            verified_cp_ts_by_ep=bad_eps,
            out_dir=img_root / "bad",
        )


if __name__ == "__main__":
    main()
