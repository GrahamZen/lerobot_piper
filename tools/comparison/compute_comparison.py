"""Offline comparison score computation.

Loads a LeRobot dataset and an ACT policy, runs simulated inference
step-by-step, and saves each detector's per-step anomaly score to a
JSONL file under ``<output_dir>/<method>_scores.jsonl``.

The model path is resolved automatically from the dataset's
``meta/record_config.json`` if not explicitly provided (same logic as
``tools/failure/visualize_dataset_ckpt_clue.py``).

The execution horizon (how many steps between re-predictions) is also
auto-detected from the record config:
- If ``temporal_ensemble_coeff`` is set  → k = 1 (re-predict every step)
- Otherwise                              → k = n_action_steps

Workflow
--------
1. Run compute (scores + calibration in one call)::

    python tools/comparison/compute_comparison.py \\
        --repo_id eval/eval_pick_and_place_act \\
        --calibration_repo_id eval/eval_pick_and_place_calibrate

3. Visualise::

    python tools/comparison/visualize_comparison.py \\
        --repo_id eval/eval_pick_and_place_act
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "failure"))

from offline_utils import _extract_pretrained_path, get_episode_bounds  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tools.comparison.methods.base import BaseDetector  # noqa: E402

# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------


def _make_loader(
    dataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Build a sequential DataLoader over a contiguous slice of *dataset*.

    ``batch_size`` controls how many frames are prefetched at once by the
    background workers.  The policy is still called once per frame (see
    callers) to avoid subtle differences with batched transformer inference.
    """
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def _read_record_config(dataset_root: Path) -> dict:
    cfg_path = dataset_root / "meta" / "record_config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def resolve_model_path(dataset_root: Path, override: str | None) -> Path:
    """Return the pretrained model path, preferring *override* if given."""
    if override:
        return Path(override).expanduser()
    record_config = _read_record_config(dataset_root)
    path = _extract_pretrained_path(record_config)
    if path is None:
        raise ValueError(
            f"Could not resolve model path from {dataset_root}/meta/record_config.json. "
            "Pass --model_path explicitly."
        )
    return path


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def load_policy(model_path: Path, device: torch.device):
    """Load policy from *model_path* in eval mode on *device*.

    Policy type is resolved from the checkpoint's config so that both
    ACTPolicy and ACTFMPolicy (and any future policy type) are supported.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(str(model_path))
    policy = get_policy_class(cfg.type).from_pretrained(str(model_path))
    policy.eval().to(device)
    print(f"[INFO] Loaded {cfg.type} policy from {model_path}")
    return policy


# ---------------------------------------------------------------------------
# Chunk sampling
# ---------------------------------------------------------------------------


def sample_action_chunks(policy, batch: dict[str, Tensor]) -> Tensor:
    """Call ACT's predict_action_chunk and return ``(1, chunk_size, action_dim)``."""
    with torch.no_grad():
        return policy.predict_action_chunk(batch)


def _policy_uses_vae(policy) -> bool:
    """Return True if policy is ACT with use_vae=True (needs sample_latent for stochastic inference)."""
    cfg = getattr(policy, "config", None)
    return cfg is not None and bool(getattr(cfg, "use_vae", False))


def _collect_multi_chunks(
    policy,
    obs: dict[str, Tensor],
    n_samples: int,
    sample_latent: bool,
) -> Tensor:
    """Run the policy once on an expanded batch and return per-frame samples.

    Expands the observation from ``(B, ...)`` to ``(B*n_samples, ...)`` via
    ``repeat_interleave``, then runs a single forward pass.  This is
    ~n_samples× faster than calling the policy in a loop because the GPU
    processes the full batch in one kernel launch (image backbone + transformer).

    Args:
        policy: The loaded policy.
        obs: Observation dict with batch dimension B (from DataLoader).
        n_samples: Number of independent samples to draw per frame.
        sample_latent: Pass True for ACT (CVAE) so each element in the
            expanded batch draws a different z ~ N(0, I).

    Returns:
        ``(B, n_samples, H, action_dim)`` tensor.
    """
    batch_size = next(iter(obs.values())).shape[0]
    # Expand: frame 0 repeats n_samples times, then frame 1, etc.
    # (B, ...) → (B*n_samples, ...)
    expanded = {k: v.repeat_interleave(n_samples, dim=0) for k, v in obs.items()}
    with torch.no_grad():
        if sample_latent:
            chunks_flat = policy.predict_action_chunk(expanded, sample_latent=True)
        else:
            chunks_flat = policy.predict_action_chunk(expanded)
    # (B*n_samples, H, D) → (B, n_samples, H, D)
    horizon, action_dim = chunks_flat.shape[1], chunks_flat.shape[2]
    return chunks_flat.view(batch_size, n_samples, horizon, action_dim)


# ---------------------------------------------------------------------------
# RND training
# ---------------------------------------------------------------------------


def _train_logpzo_checkpoint(
    calibration_repo_id: str,
    calibration_root: str | None,
    policy,
    extractor,
    device: torch.device,
    output_path: Path,
    in_dim: int = 14,
    n_epochs: int = 200,
    batch_size_train: int = 128,
    lr: float = 1e-4,
    cal_fraction: float = 0.3,
) -> None:
    """Collect encoder embeddings from the calibration dataset and train a logpZO flow model.

    Follows FAIL-Detect's training procedure exactly:
      - Input: encoder embeddings reshaped via adjust_xshape(emb, in_dim)
      - Flow matching loss: MSE between predicted and true velocity (x1 - x0)
      - Time-scale: 100 (discrete UNet timesteps)

    Only the first ``(1 - cal_fraction)`` of calibration episodes are used for training.
    The remaining episodes are left untouched so that ``run_calibration`` can compute
    honest (non-overfitted) calibration scores on held-out data, which is essential for
    setting a meaningful threshold.  With a dataset of 10 episodes and cal_fraction=0.3,
    3 episodes are held out — the threshold will be based on those 3 honest scores.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    from tools.comparison.methods.logpzo import _adjust_xshape, build_flow_model

    cal_dataset = LeRobotDataset(calibration_repo_id, root=calibration_root)
    if cal_dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        cal_dataset.meta.episodes = load_episodes(cal_dataset.root)

    n_eps = len(cal_dataset.meta.episodes)
    # Hold out the last cal_fraction of episodes for threshold calibration so that
    # run_calibration() can compute honest (non-overfitted) scores on unseen data.
    n_train_eps = max(1, int(n_eps * (1.0 - cal_fraction)))
    print(
        f"[logpZO train] Collecting embeddings from {calibration_repo_id} "
        f"({n_eps} episodes, using first {n_train_eps} for training, "
        f"last {n_eps - n_train_eps} held out for calibration) ..."
    )

    embeddings: list[np.ndarray] = []
    for ep_idx in tqdm(range(n_train_eps), desc="[logpZO train] episodes"):
        ep_from, ep_to = get_episode_bounds(cal_dataset, ep_idx)
        loader = _make_loader(cal_dataset, list(range(ep_from, ep_to)), batch_size=16, num_workers=4)
        for batch in loader:
            obs = {
                k: (v if isinstance(v, Tensor) else torch.as_tensor(v)).to(device, non_blocking=True)
                for k, v in batch.items()
                if k.startswith("observation.") or k == "observation_state"
            }
            with torch.no_grad():
                policy.predict_action_chunk(obs)
            batch_embs = extractor.last_batch
            if batch_embs is not None:
                embeddings.extend(batch_embs.copy())

    if not embeddings:
        print("[logpZO train] No embeddings collected — skipping training.")
        return

    emb_arr = np.array(embeddings, dtype=np.float32)
    print(f"[logpZO train] {len(emb_arr)} embeddings collected (emb_dim={emb_arr.shape[1]})")

    # z-score normalise: critical so that the flow model can learn a useful
    # mapping toward N(0,I).  ACT encoder features are not naturally ~N(0,I);
    # without this the score collapses to a constant for all episodes.
    emb_mean = emb_arr.mean(axis=0, keepdims=True)  # (1, D)
    emb_std = emb_arr.std(axis=0, keepdims=True).clip(min=1e-6)  # (1, D)
    emb_arr_norm = (emb_arr - emb_mean) / emb_std
    print(
        f"[logpZO train] Embedding mean norm: {np.linalg.norm(emb_mean):.3f},  "
        f"std mean: {emb_std.mean():.3f}  (after normalisation: std≈1)"
    )

    # Reshape embeddings for UNet: (N, D) → (N, seq_len, in_dim)
    emb_t = torch.tensor(emb_arr_norm, dtype=torch.float32)
    emb_seq = _adjust_xshape(emb_t, in_dim)  # (N, seq_len, in_dim)
    print(f"[logpZO train] Reshaped to {emb_seq.shape}  (in_dim={in_dim})")

    net = build_flow_model(in_dim).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loader_train = DataLoader(TensorDataset(emb_seq), batch_size=batch_size_train, shuffle=True)
    time_scale = 100

    for epoch in tqdm(range(n_epochs), desc="[logpZO train] epochs"):
        net.train()
        total = 0.0
        for (x0,) in loader_train:
            x0 = x0.to(device)
            x1 = torch.randn_like(x0)
            vtrue = x1 - x0
            cont_t = torch.rand(len(x0), device=device).view(-1, 1, 1)
            xnow = x0 + cont_t * vtrue
            t_discrete = (cont_t.view(-1) * time_scale).long()
            vhat = net(xnow, t_discrete)
            loss = (vhat - vtrue).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 20 == 0 or epoch == 0:
            tqdm.write(f"  epoch {epoch + 1:3d}/{n_epochs}  loss={total / len(loader_train):.6f}")

    net.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": net.state_dict(),
            "in_dim": in_dim,
            "emb_mean": emb_mean[0].tolist(),  # (D,)
            "emb_std": emb_std[0].tolist(),  # (D,)
        },
        output_path,
    )
    print(f"[logpZO train] Saved checkpoint → {output_path}")


def _train_rnd_checkpoint(
    calibration_repo_id: str,
    calibration_root: str | None,
    policy,
    extractor,
    device: torch.device,
    output_path: Path,
    hidden_dim: int = 1024,
    out_dim: int = 512,
    n_epochs: int = 50,
    batch_size_train: int = 256,
    lr: float = 1e-4,
) -> None:
    """Collect encoder embeddings from the calibration dataset and train a RND checkpoint.

    Reuses the already-loaded *policy* and *extractor* so no second model load
    is needed.  The trained checkpoint is saved to *output_path*.
    """
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm import tqdm

    from tools.comparison.methods.rnd import _MLP

    cal_dataset = LeRobotDataset(calibration_repo_id, root=calibration_root)
    if cal_dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        cal_dataset.meta.episodes = load_episodes(cal_dataset.root)

    n_eps = len(cal_dataset.meta.episodes)
    print(f"[RND train] Collecting embeddings from {calibration_repo_id} ({n_eps} episodes) ...")

    embeddings: list[np.ndarray] = []
    for ep_idx in tqdm(range(n_eps), desc="[RND train] episodes"):
        ep_from, ep_to = get_episode_bounds(cal_dataset, ep_idx)
        loader = _make_loader(cal_dataset, list(range(ep_from, ep_to)), batch_size=16, num_workers=4)
        for batch in loader:
            obs = {
                k: (v if isinstance(v, Tensor) else torch.as_tensor(v)).to(device, non_blocking=True)
                for k, v in batch.items()
                if k.startswith("observation.") or k == "observation_state"
            }
            with torch.no_grad():
                policy.predict_action_chunk(obs)
            batch_embs = extractor.last_batch
            if batch_embs is not None:
                embeddings.extend(batch_embs.copy())

    if not embeddings:
        print("[RND train] No embeddings collected — skipping training.")
        return

    emb_arr = np.array(embeddings, dtype=np.float32)
    in_dim = emb_arr.shape[1]
    print(f"[RND train] {len(emb_arr)} embeddings collected (dim={in_dim})")

    target = _MLP(in_dim, hidden_dim, out_dim).to(device)
    predictor = _MLP(in_dim, hidden_dim, out_dim).to(device)
    for p in target.parameters():
        p.requires_grad_(False)
    target.eval()

    optimizer = optim.Adam(predictor.parameters(), lr=lr)
    data = torch.tensor(emb_arr, dtype=torch.float32)
    loader_train = DataLoader(TensorDataset(data), batch_size=batch_size_train, shuffle=True)

    for epoch in tqdm(range(n_epochs), desc="[RND train] epochs"):
        predictor.train()
        total = 0.0
        for (x,) in loader_train:
            x = x.to(device)
            with torch.no_grad():
                t = target(x)
            loss = nn.functional.mse_loss(predictor(x), t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"  epoch {epoch + 1:3d}/{n_epochs}  loss={total / len(loader_train):.6f}")

    predictor.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "target": target.state_dict(),
            "predictor": predictor.state_dict(),
        },
        output_path,
    )
    print(f"[RND train] Saved checkpoint → {output_path}")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def run_calibration(
    calibration_repo_id: str,
    calibration_root: str | None,
    policy,
    detectors: list[BaseDetector],
    extractor,  # EmbeddingExtractor | None
    device: torch.device,
    delta: float,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 8,
    n_samples: int = 1,
) -> tuple[dict[str, list[list[float]]], Path]:
    """Calibrate all detectors on successful episodes and save thresholds.

    For embedding-based detectors (``pca_kmeans``, ``similarity``), embeddings are
    collected in a single policy pass.  ``PCAKMeansDetector`` and
    ``SimilarityDetector`` are then fitted from all collected embeddings before
    calibration scores are computed by replaying the stored data — no second
    policy run is needed.

    For each detector, the episode-level ``calibration_score()`` is collected
    and the ``(1-δ)`` quantile is saved as the failure threshold.

    Returns:
        ``episode_step_scores``: per-detector per-episode per-step scores from
        the calibration dataset.  Used by inline evaluation to form the
        train / calibration split for FunctionalPredictor.
    """
    from tools.comparison.methods.pca_kmeans import PCAKMeansDetector

    cal_dataset = LeRobotDataset(calibration_repo_id)
    if cal_dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        cal_dataset.meta.episodes = load_episodes(cal_dataset.root)

    n_eps = len(cal_dataset.meta.episodes)
    print(f"[Calibration] {n_eps} episode(s) from {calibration_repo_id}")

    needs_emb = extractor is not None
    use_multi = n_samples > 1
    sample_latent = use_multi and _policy_uses_vae(policy)
    # With multi-sampling the DataLoader batch is expanded n_samples× inside
    # _collect_multi_chunks, so reduce the loader batch_size proportionally to
    # keep GPU memory constant.
    effective_batch = max(1, batch_size // n_samples) if use_multi else batch_size

    # ------------------------------------------------------------------
    # Single policy pass on expanded batch per episode per step.
    # _collect_multi_chunks expands (effective_batch, ...) → (effective_batch*n_samples, ...)
    # and runs ONE forward pass, giving (effective_batch, n_samples, H, D).
    # ------------------------------------------------------------------
    # cal_data[ep_idx] = list of (chunks_per_frame, emb | None, state | None)
    cal_data: list[list[tuple[Tensor, np.ndarray | None, np.ndarray | None]]] = []

    from tqdm import tqdm

    for ep_idx in tqdm(range(n_eps), desc="[Calibration] episodes"):
        ep_from, ep_to = get_episode_bounds(cal_dataset, ep_idx)
        ep_data: list[tuple[Tensor, np.ndarray | None, np.ndarray | None]] = []

        loader = _make_loader(cal_dataset, list(range(ep_from, ep_to)), effective_batch, num_workers)
        for batch in tqdm(loader, desc=f"  ep {ep_idx + 1}", leave=False):
            obs = {
                k: (v if isinstance(v, Tensor) else torch.as_tensor(v)).to(device, non_blocking=True)
                for k, v in batch.items()
                if k.startswith("observation.") or k == "observation_state"
            }
            if use_multi:
                # Single forward pass on expanded batch → (B, n_samples, H, D)
                multi = _collect_multi_chunks(policy, obs, n_samples, sample_latent)
                raw_embs = extractor.last_batch if needs_emb else None  # (B*n_samples, D)
                embs_batch = raw_embs[::n_samples] if raw_embs is not None else None  # (B, D)
            else:
                with torch.no_grad():
                    multi = policy.predict_action_chunk(obs).unsqueeze(1)  # (B, 1, H, D)
                embs_batch = extractor.last_batch if needs_emb else None  # (B, D)
            states_val = (
                batch.get("observation.state") if isinstance(batch.get("observation.state"), Tensor) else None
            )

            actual_b = multi.shape[0]
            for i in range(actual_b):
                emb = embs_batch[i].copy() if embs_batch is not None else None
                state: np.ndarray | None = None
                if states_val is not None:
                    state = states_val[i].cpu().float().numpy()
                ep_data.append((multi[i], emb, state))  # (n_samples, H, D)
        cal_data.append(ep_data)

    # ------------------------------------------------------------------
    # Fit embedding-based detectors (PCAKMeans, SimilarityDetector) from calibration
    # ------------------------------------------------------------------
    from tools.comparison.methods.similarity import SimilarityDetector

    emb_fit_detectors = [d for d in detectors if isinstance(d, (PCAKMeansDetector, SimilarityDetector))]
    if emb_fit_detectors:
        all_embs = [emb for ep in cal_data for _, emb, _ in ep if emb is not None]
        if all_embs:
            emb_arr = np.array(all_embs, dtype=np.float32)
            for d in emb_fit_detectors:
                d.fit(emb_arr)
            # Persist embeddings so cached calibration can re-fit detectors
            emb_cache = output_dir / "calibration_embeddings.npy"
            np.save(emb_cache, emb_arr)
            print(f"[Calibration] Saved {len(emb_arr)} embeddings → {emb_cache}")
        else:
            print("[WARN] No embeddings collected — embedding-based detectors not fitted.")

    # ------------------------------------------------------------------
    # Replay stored data to compute calibration scores (no policy re-run)
    # ------------------------------------------------------------------
    episode_max_scores: dict[str, list[float]] = {d.name: [] for d in detectors}
    episode_step_scores: dict[str, list[list[float]]] = {d.name: [] for d in detectors}

    for _ep_idx, ep_data in enumerate(cal_data):
        for d in detectors:
            d.reset()
        ep_raw: dict[str, list[float]] = {d.name: [] for d in detectors}
        for ep_step, (chunks, emb, state) in enumerate(ep_data):
            if needs_emb and emb is not None:
                extractor.set_embedding(emb)
            if extractor is not None and state is not None:
                extractor.set_state(state)
            for d in detectors:
                score = d.update(ep_step, chunks)
                ep_raw[d.name].append(score)
        for d in detectors:
            episode_max_scores[d.name].append(d.calibration_score())
            episode_step_scores[d.name].append(ep_raw[d.name])

    # ------------------------------------------------------------------
    # Save thresholds — episode-max quantile
    # ------------------------------------------------------------------
    quantile = 1.0 - delta
    for d in detectors:
        scores = episode_max_scores[d.name]
        threshold_val = float(np.quantile(scores, quantile))
        out = {
            "method": d.name,
            "threshold": threshold_val,  # float
            "type": "quantile",
            "delta": delta,
            "n_episodes": len(scores),
            "episode_scores": scores,
        }
        print(f"[Calibration] {d.name}: threshold={threshold_val:.6f}  (δ={delta}, N={len(scores)})")
        path = output_dir / f"{d.name}_threshold.json"
        with path.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"  → {path}")

    # Save per-step scores for the calibration dataset as JSONL so that
    # evaluate_comparison.py can use them via --cal_scores_dir.
    cal_output_dir = Path(cal_dataset.root) / "comparison"
    cal_output_dir.mkdir(parents=True, exist_ok=True)
    for d in detectors:
        rows: list[dict[str, Any]] = []
        gs = 0
        for ep_idx, step_scores in enumerate(episode_step_scores[d.name]):
            for step_in_ep, score in enumerate(step_scores):
                gs += 1
                if score <= -1e8:  # warmup sentinel — skip
                    continue
                rows.append(
                    {
                        "global_step": gs,
                        "episode": ep_idx,
                        "step_in_episode": step_in_ep,
                        "score": score,
                    }
                )
        cal_path = cal_output_dir / f"{d.name}_scores.jsonl"
        BaseDetector.save_scores(rows, cal_path)
        print(f"[Calibration] Cal scores → {cal_path}")

    return episode_step_scores, cal_output_dir


# ---------------------------------------------------------------------------
# Per-episode computation
# ---------------------------------------------------------------------------


def compute_episode(
    episode_idx: int,
    episode_from: int,
    episode_to: int,
    dataset: LeRobotDataset,
    policy,
    detectors: list[BaseDetector],
    extractor,  # EmbeddingExtractor | None
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 8,
    n_samples: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    """Run all detectors on one episode; return ``{name: [score_row, ...]}``.

    Each row: ``{global_step, episode, step_in_episode, score}``.

    Frames are prefetched by *num_workers* DataLoader workers so IO runs
    in the background while the GPU runs the current batch.

    When ``n_samples > 1`` the policy is called *n_samples* times per
    DataLoader batch and detectors receive ``(n_samples, H, D)`` chunks
    instead of ``(1, H, D)``.  Required for ``ActionEntropyDetector``.
    ACT (CVAE) policies automatically use ``sample_latent=True`` so that
    each call draws a different latent z from N(0, I).
    """
    results: dict[str, list[dict]] = {d.name: [] for d in detectors}
    for d in detectors:
        d.reset()

    from tqdm import tqdm

    use_multi = n_samples > 1
    sample_latent = use_multi and _policy_uses_vae(policy)
    effective_batch = max(1, batch_size // n_samples) if use_multi else batch_size

    indices = list(range(episode_from, episode_to))
    loader = _make_loader(dataset, indices, effective_batch, num_workers)
    ep_step = 0

    for batch in tqdm(loader, desc=f"  ep {episode_idx + 1}", leave=False):
        obs = {
            k: (v if isinstance(v, Tensor) else torch.as_tensor(v)).to(device, non_blocking=True)
            for k, v in batch.items()
            if k.startswith("observation.") or k == "observation_state"
        }

        if use_multi:
            # Single forward pass on expanded batch (B*n_samples, ...) → (B, n_samples, H, D)
            multi = _collect_multi_chunks(policy, obs, n_samples, sample_latent)
            # extractor.last_batch is (B*n_samples, D); take the first sample per frame
            raw_embs = extractor.last_batch if extractor is not None else None
            embs_batch = raw_embs[::n_samples] if raw_embs is not None else None  # (B, D)
        else:
            with torch.no_grad():
                multi = policy.predict_action_chunk(obs).unsqueeze(1)  # (B, 1, H, D)
            embs_batch = extractor.last_batch if extractor is not None else None  # (B, D)

        states_val = (
            batch.get("observation.state") if isinstance(batch.get("observation.state"), Tensor) else None
        )

        actual_b = multi.shape[0]
        for i in range(actual_b):
            frame_idx = episode_from + ep_step
            if extractor is not None and embs_batch is not None:
                extractor.set_embedding(embs_batch[i])
            if extractor is not None and states_val is not None:
                extractor.set_state(states_val[i].cpu().float().numpy())
            chunks_i = multi[i]  # (n_samples, H, D)
            for d in detectors:
                score = d.update(ep_step, chunks_i)
                if score <= -1e8:  # warmup sentinel — skip writing to JSONL
                    continue
                results[d.name].append(
                    {
                        "global_step": frame_idx,
                        "episode": episode_idx,
                        "step_in_episode": ep_step,
                        "score": score,
                    }
                )
            ep_step += 1

    return results


# ---------------------------------------------------------------------------
# Inline evaluation helper
# ---------------------------------------------------------------------------


def _generate_barplot(
    df: pd.DataFrame,
    output_dir: Path,
    title: str = "Failure Detection Comparison",
    fontsize: int = 14,
) -> None:
    """Generate a 3-panel bar chart (Accuracy | Weighted Accuracy | Detection Time).

    Adapted from FAIL-Detect (RSS 2025).  Top-3 methods are highlighted:
    1st = red, 2nd = skyblue, 3rd = green; rest = grey.
    For Detection Time, bottom-3 (lower is better) are highlighted instead.
    """
    import matplotlib.pyplot as plt

    rank_colors = ["red", "skyblue", "green"]

    def _top3_colors(values: pd.Series, higher_is_better: bool) -> list[str]:
        colors = ["grey"] * len(values)
        unique_sorted = np.sort(values.unique())
        if higher_is_better:
            unique_sorted = unique_sorted[::-1]
        for i, val in enumerate(values):
            rank = np.where(unique_sorted == val)[0]
            if len(rank) > 0 and rank[0] < 3:
                colors[i] = rank_colors[rank[0]]
        return colors

    def _bottom3_colors(values: pd.Series) -> list[str]:
        colors = ["grey"] * len(values)
        non_zero = values[values > 0]
        if non_zero.empty:
            return colors
        sorted_asc = np.sort(non_zero.unique())
        for i, val in enumerate(values):
            if val == 0:
                continue
            rank = np.where(sorted_asc == val)[0]
            if len(rank) > 0 and rank[0] < 3:
                colors[i] = rank_colors[rank[0]]
        return colors

    to_plot = ["Accuracy", "Accuracy_weighted", "Detection_time"]
    titles = ["Accuracy", "Weighted Accuracy", "Detection Time"]
    methods = list(df.index)
    n_methods = len(methods)
    x = np.arange(n_methods)

    fig, axes = plt.subplots(1, 3, figsize=(7 * 3, 5))
    fig.suptitle(title, fontsize=fontsize + 4, y=1.01)

    for ax, metric, panel_title in zip(axes, to_plot, titles, strict=False):
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        vals = df[metric].fillna(0.0)
        is_time = metric == "Detection_time"
        bar_colors = _bottom3_colors(vals) if is_time else _top3_colors(vals, higher_is_better=True)
        ax.bar(x, vals, color=bar_colors)

        if is_time and "Detection_time_SE" in df.columns:
            se = df["Detection_time_SE"].fillna(0.0)
            ax.errorbar(x, vals, yerr=se, fmt="none", ecolor="black", capsize=3)

        if is_time:
            max_val = vals.max()
            ax.set_ylim(0, max_val * 1.35 if max_val > 0 else 1.0)
        else:
            ax.set_ylim(0, 1.15)

        se_vals = df.get("Detection_time_SE", pd.Series([0.0] * len(df))).fillna(0.0)
        for i, (v, se) in enumerate(zip(vals, se_vals, strict=False)):
            if is_time:
                label = "NaN" if v == 0 else str(int(round(v)))
                offset = se * 1.01 if se > 0 else v * 0.02
            else:
                label = f"{v:.3f}"
                offset = v * 0.02
            ax.text(i, v + offset, label, ha="center", va="bottom", fontsize=fontsize - 2, color="black")

        ax.set_title(panel_title, fontsize=fontsize + 4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right", fontsize=fontsize)
        ax.tick_params(axis="y", labelsize=fontsize)

    fig.tight_layout()
    out_png = output_dir / "results_barplot.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[Barplot] Saved → {out_png}")


def _run_inline_evaluation(
    detectors: list[BaseDetector],
    cal_step_scores: dict[str, list[list[float]]],
    all_rows: dict[str, list[dict]],
    labels_path: Path,
    num_train: int,
    num_cal: int,
    alpha: float,
    output_dir: Path,
    debug: bool = False,
    dataset_root: Path | None = None,
    step_quantile_methods: set[str] | None = None,
    step_quantile_alpha: float = 0.01,
    tail_steps: int = 0,
) -> None:
    """Run FunctionalPredictor CP-band evaluation inline after scoring.

    Uses ``cal_step_scores`` (from the calibration dataset) as the train/cal
    split for FunctionalPredictor, and the main-dataset scores together with
    ``episode_labels.json`` as the test split.

    Saves ``results.pkl`` and ``results.csv`` to *output_dir* and prints a
    metrics table.

    Split logic
    -----------
    * The first ``num_train`` calibration episodes form the band-centre fit.
    * The next ``num_cal`` calibration episodes provide the conformal quantile.
    * If the calibration dataset has fewer episodes than ``num_train + num_cal``,
      the available episodes are split 50/50.
    * All main-dataset episodes (both success and failure) form the test set.
    """
    import pickle  # nosec B403

    import pandas as pd
    from sklearn.metrics import confusion_matrix  # noqa: F401 — used inside evaluate_comparison

    from tools.comparison.evaluate_comparison import (
        _INVERTED_METHODS,
        _detect_max_threshold,
        _detect_min_threshold,  # noqa: F401 — kept for future use
        _print_episode_debug,
        compute_metrics,
        load_episode_labels,
    )

    print(f"\n[Evaluation] Loading labels from {labels_path}")
    all_labels = load_episode_labels(labels_path)
    n_success = sum(v == 1 for v in all_labels.values())
    n_fail = sum(v == 0 for v in all_labels.values())
    print(f"[Evaluation] {len(all_labels)} episodes (success={n_success}, failure={n_fail})")

    # Build per-episode step-score dict from all_rows
    main_scores: dict[str, dict[int, list[float]]] = {}
    for d in detectors:
        per_ep: dict[int, list[tuple[int, float]]] = {}
        for row in all_rows[d.name]:
            ep = int(row["episode"])
            step = int(row["step_in_episode"])
            score = float(row["score"])
            per_ep.setdefault(ep, []).append((step, score))
        main_scores[d.name] = {ep: [s for _, s in sorted(steps)] for ep, steps in per_ep.items()}

    metric_names = ["TPR", "TNR", "Accuracy", "Accuracy_weighted", "Detection_time", "Detection_time_SE"]
    records: dict[str, dict[str, float]] = {}

    for d in detectors:
        method = d.name
        cal_seqs = cal_step_scores.get(method, [])
        n_cal_eps = len(cal_seqs)

        if n_cal_eps == 0:
            print(f"[{method}] No calibration step scores — skipping evaluation.")
            continue

        # Split cal into train / cal
        if n_cal_eps < num_train + num_cal:
            _nt = n_cal_eps // 2
            _nc = n_cal_eps - _nt
            print(
                f"[{method}] Only {n_cal_eps} cal episodes; "
                f"using {_nt} train + {_nc} cal (requested {num_train}+{num_cal})."
            )
        else:
            _nt, _nc = num_train, num_cal

        train_seqs = cal_seqs[:_nt]
        cal_seqs_split = cal_seqs[_nt : _nt + _nc]

        # Test set: all main-dataset episodes with labels
        ep_scores = main_scores[method]
        common_eps = sorted(set(ep_scores) & set(all_labels))
        if not common_eps:
            print(f"[{method}] No episodes with both scores and labels — skipping.")
            continue

        test_scores = [ep_scores[ep] for ep in common_eps]
        test_labels = [all_labels[ep] for ep in common_eps]
        n_test_fail = sum(1 for lbl in test_labels if lbl == 0)

        if n_test_fail == 0:
            print(f"[{method}] No failure episodes in test set — skipping.")
            continue

        # ---- Threshold selection ----
        # Default: CP formula over per-episode max calibration scores.
        # For small calibration sets (n < 1/alpha - 1) the CP formula gives q_level=1.0
        # (threshold = max of cal scores), which is too tight — even test-success episodes
        # exceed it (TNR=0).
        #
        # For methods listed in step_quantile_methods (default: logpzo) we use the same
        # approach as the TD detector: flat-pool ALL calibration step-level scores and
        # apply the CP formula with step_quantile_alpha (default 0.01).  With ~9000+
        # calibration frames this gives q_level ≈ 0.99 — a high but meaningful percentile.
        # tail_steps>0 uses tail-mean episode scoring — skip step_quantile (step-level threshold
        # doesn't pair with episode-level tail-mean scoring).
        _use_step_quantile = (
            tail_steps == 0 and step_quantile_methods is not None and method in step_quantile_methods
        )
        _saved_threshold: float | None = None

        if _use_step_quantile:
            all_step_scores = [s for seq in (train_seqs + cal_seqs_split) for s in seq]
            if all_step_scores:
                n_s = len(all_step_scores)
                q_s = min(float(np.ceil((n_s + 1) * (1.0 - step_quantile_alpha)) / n_s), 1.0)
                _saved_threshold = float(np.quantile(all_step_scores, q_s))
                print(
                    f"[{method}] Per-step CP threshold: n={n_s}, alpha={step_quantile_alpha}, "
                    f"q={q_s:.4f} → threshold={_saved_threshold:.4f}"
                )
        elif tail_steps == 0:
            # Fallback: load saved threshold from run_calibration if CP would give q_level=1.0
            _thresh_file = output_dir / f"{method}_threshold.json"
            if _thresh_file.exists():
                try:
                    with _thresh_file.open() as _f:
                        _td = json.load(_f)
                    _saved_threshold = float(_td["threshold"])
                    print(f"[{method}] Using saved threshold={_saved_threshold:.6f} from {_thresh_file.name}")
                except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as exc:
                    print(f"[{method}] Failed to load saved threshold from {_thresh_file.name}: {exc}")
        # When tail_steps>0: threshold=None → computed from tail-mean of cal sequences inside _detect_max_threshold

        print(
            f"[{method}] train={_nt}  cal={_nc}  "
            f"test={len(common_eps)} (success={len(common_eps) - n_test_fail}, failure={n_test_fail})"
        )

        _detect_fn = _detect_min_threshold if method in _INVERTED_METHODS else _detect_max_threshold
        _tail = 0 if method in _INVERTED_METHODS else tail_steps
        # When tail_steps>0, keep the saved max-based threshold — it is always ≥ any tail-mean
        # of a success episode, so it covers the tail-mean success distribution without being
        # inflated by transient mid-trajectory anomaly peaks.
        y_true, y_pred, first_steps = _detect_fn(
            train_seqs + cal_seqs_split,
            test_scores,
            test_labels,
            alpha=alpha,
            threshold=_saved_threshold,
            tail_steps=_tail,
        )

        metrics = compute_metrics(y_true, y_pred)
        metrics["Detection_time"] = float(np.mean(first_steps)) if first_steps else float("nan")
        metrics["Detection_time_SE"] = (
            float(np.std(first_steps) / np.sqrt(len(first_steps))) if len(first_steps) > 1 else 0.0
        )
        if debug:
            _print_episode_debug(method, common_eps, y_true, y_pred, test_scores)
        records[method] = metrics

    # --- Ours (TD) ---
    if dataset_root is not None:
        from offline_utils import load_failure_metrics_jsonl

        td_raw = load_failure_metrics_jsonl(dataset_root)
        if td_raw:
            # Build global_step → (episode, step_in_episode) from any detector's rows
            gs_to_ep: dict[int, tuple[int, int]] = {}
            first_rows = next(iter(all_rows.values()), [])
            for row in first_rows:
                gs_to_ep[int(row["global_step"])] = (int(row["episode"]), int(row["step_in_episode"]))

            # Group td_smoothed scores by episode
            td_by_ep: dict[int, list[tuple[int, float]]] = {}
            for gs, row_data in td_raw.items():
                score = row_data.get("td_smoothed")
                if score is None:
                    continue
                ep_info = gs_to_ep.get(gs)
                if ep_info is None:
                    continue
                ep_idx, step = ep_info
                td_by_ep.setdefault(ep_idx, []).append((step, float(score)))

            td_scores: dict[int, list[float]] = {
                ep: [s for _, s in sorted(steps)] for ep, steps in td_by_ep.items()
            }

            # Write td_scores.jsonl so evaluate_comparison.py can find it
            if td_scores:
                td_jsonl_rows: list[dict[str, Any]] = []
                gs_counter = 0
                for ep_idx in sorted(td_scores):
                    for step_in_ep, score in enumerate(td_scores[ep_idx]):
                        td_jsonl_rows.append(
                            {
                                "global_step": gs_counter,
                                "episode": ep_idx,
                                "step_in_episode": step_in_ep,
                                "score": score,
                            }
                        )
                        gs_counter += 1
                td_jsonl_path = output_dir / "td_scores.jsonl"
                BaseDetector.save_scores(td_jsonl_rows, td_jsonl_path)
                print(f"[Ours (TD)] Saved {len(td_jsonl_rows)} rows → {td_jsonl_path}")

            # Load TD threshold from failure_handling.json
            td_threshold: float | None = None
            pretrained_path = _extract_pretrained_path(_read_record_config(dataset_root))
            if pretrained_path is not None:
                try:
                    fh_json = pretrained_path / "failure_handling.json"
                    if fh_json.exists():
                        with fh_json.open() as _f:
                            fh = json.load(_f)
                        det = fh.get("detector") or fh
                        v = float(det.get("failure_threshold", det.get("threshold", 0.0)))
                        td_threshold = v or None
                except Exception:  # nosec B110
                    pass

            if td_scores and td_threshold is not None:
                common_eps_td = sorted(set(td_scores) & set(all_labels))
                test_scores_td = [td_scores[ep] for ep in common_eps_td]
                test_labels_td = [all_labels[ep] for ep in common_eps_td]
                n_test_fail_td = sum(1 for lbl in test_labels_td if lbl == 0)
                if n_test_fail_td > 0:
                    print(
                        f"[Ours (TD)] test={len(common_eps_td)} "
                        f"(success={len(common_eps_td) - n_test_fail_td}, failure={n_test_fail_td})  "
                        f"threshold={td_threshold:.6f}"
                    )
                    y_true_td, y_pred_td, first_steps_td = _detect_max_threshold(
                        [],  # no calibration sequences — threshold already saved
                        test_scores_td,
                        test_labels_td,
                        alpha=alpha,
                        threshold=td_threshold,
                    )
                    metrics_td = compute_metrics(y_true_td, y_pred_td)
                    metrics_td["Detection_time"] = (
                        float(np.mean(first_steps_td)) if first_steps_td else float("nan")
                    )
                    metrics_td["Detection_time_SE"] = (
                        float(np.std(first_steps_td) / np.sqrt(len(first_steps_td)))
                        if len(first_steps_td) > 1
                        else 0.0
                    )
                    if debug:
                        _print_episode_debug("Ours (TD)", common_eps_td, y_true_td, y_pred_td, test_scores_td)
                    records["Ours (TD)"] = metrics_td
            elif td_scores and td_threshold is None:
                print("[Ours (TD)] No threshold found in failure_handling.json — skipping evaluation.")
            elif not td_raw:
                print("[Ours (TD)] failure_metrics.jsonl not found or empty — skipping.")

    if not records:
        print("[Evaluation] No methods produced valid results.")
        return

    df = pd.DataFrame(records, index=metric_names).T
    df.index.name = "Method"

    print("\n" + "=" * 60)
    print(df.round(4).to_string())
    print("=" * 60)

    pkl_path = output_dir / "results.pkl"
    csv_path = output_dir / "results.csv"
    with pkl_path.open("wb") as f:
        pickle.dump(df, f)
    df.to_csv(csv_path)
    print(f"\n[Evaluation] Saved → {pkl_path}")
    print(f"[Evaluation] Saved → {csv_path}")

    # --- Auto-generate barplot ---
    _generate_barplot(df, output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute offline comparison scores.")
    parser.add_argument("--repo_id", default=None, help="e.g. eval/eval_pick_and_place_act")
    parser.add_argument("--root", default=None, help="Dataset root, e.g. ~/.cache/huggingface/lerobot")
    parser.add_argument(
        "--model_path",
        default=None,
        help="Path to pretrained ACT checkpoint.  Auto-resolved from record_config.json if omitted.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        choices=["stac", "rnd", "pca_kmeans", "similarity", "logpzo", "action_entropy", "all"],
        help="Methods to run (default: all available).  Pass 'all' explicitly or omit to run every method.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Independent policy samples per step.  Required for action_entropy (recommended: 16).  "
        "ACT (CVAE) automatically uses sample_latent=True; ACT-FM / Diffusion are already stochastic.",
    )
    # RND (FIPER)
    parser.add_argument(
        "--rnd_checkpoint",
        default=None,
        help="Path to RND .pt checkpoint.  Defaults to outputs/rnd/<calibration_repo_id_name>.pt",
    )
    parser.add_argument("--output_dir", default=None, help="Defaults to <dataset_root>/comparison/")
    parser.add_argument("--num_episode", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # STAC
    parser.add_argument(
        "--stac_execution_horizon",
        type=int,
        default=10,
        help="Steps between re-predictions for STAC (default: 10).",
    )
    parser.add_argument(
        "--stac_gamma",
        default="median",
        help="RBF gamma for STAC K(x,y)=exp(-γ||x-y||²): 'median' (default) or a float.",
    )
    # logpZO (FAIL-Detect)
    parser.add_argument(
        "--logpzo_checkpoint",
        default=None,
        help="Path to logpZO .pt checkpoint.  Defaults to outputs/logpzo/<calibration_repo_id_name>.pt",
    )
    parser.add_argument(
        "--logpzo_in_dim",
        type=int,
        default=14,
        help="Token width when reshaping the embedding for the flow model (default: 14 = action_dim).",
    )
    parser.add_argument(
        "--logpzo_epochs",
        type=int,
        default=200,
        help="Training epochs for the logpZO flow model (default: 200).",
    )
    parser.add_argument(
        "--logpzo_cal_fraction",
        type=float,
        default=0.3,
        help="Fraction of calibration episodes held out for threshold calibration (not used for training). "
        "Prevents threshold collapse from train/calibration overlap (default: 0.3).",
    )
    # PCA+KMeans (FAIL-Detect)
    parser.add_argument(
        "--pca_kmeans_emb_dim",
        type=int,
        default=32,
        help="PCA target dimensionality for pca_kmeans (default: 32).",
    )
    parser.add_argument(
        "--pca_kmeans_clusters", type=int, default=64, help="K-means centroids for pca_kmeans (default: 64)."
    )
    # DataLoader / throughput
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Frames per policy forward pass (default: 32).  "
        "Larger values improve GPU utilisation; reduce if OOM.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader background workers for data prefetching (default: 8).  "
        "Set to 0 to disable multiprocessing.",
    )
    # Calibration
    parser.add_argument(
        "--calibration_repo_id",
        default=None,
        help="repo_id of successful episodes for threshold calibration.  "
        "Required for 'pca_kmeans' and 'similarity' methods.",
    )
    parser.add_argument(
        "--calibration_root", default=None, help="Root for the calibration dataset (defaults to --root)."
    )
    parser.add_argument(
        "--calibration_delta",
        type=float,
        default=0.05,
        help="False-positive tolerance δ; threshold = (1-δ) quantile (default: 0.05).",
    )
    # Inline evaluation (FunctionalPredictor CP band + metrics)
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to episode_labels.json for the eval dataset.  "
        "Auto-detected at <output_dir>/episode_labels.json if omitted.  "
        "When found, evaluation metrics are computed inline.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=20,
        help="Cal episodes used to fit the FunctionalPredictor band centre (default: 20).",
    )
    parser.add_argument(
        "--num_cal", type=int, default=40, help="Cal episodes used for the CP quantile (default: 40)."
    )
    parser.add_argument("--alpha", type=float, default=0.025, help="CP significance level (default: 0.025).")
    parser.add_argument(
        "--step_quantile_methods",
        nargs="*",
        default=["logpzo"],
        help="Methods that use per-step CP quantile (like TD) instead of per-episode max. "
        "Needed for methods with small calibration datasets. Default: logpzo.",
    )
    parser.add_argument(
        "--step_quantile_alpha",
        type=float,
        default=0.01,
        help="Alpha for per-step CP quantile threshold (default: 0.01, matching TD detector).",
    )
    parser.add_argument(
        "--tail_steps",
        type=int,
        default=20,
        help="Use mean of last N steps as episode score instead of max.  "
        "Failures end in persistent anomalous states; successes return to normal. "
        "0 = use max (original behaviour).  Default: 20.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force re-running calibration even if cached results exist.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print per-episode prediction table during inline evaluation."
    )

    args = parser.parse_args()

    batch_file = Path(__file__).parent / "batch_runs.json"
    if not batch_file.exists() and args.repo_id is None:
        parser.error("--repo_id is required when batch_runs.json is not present.")
    if batch_file.exists():
        with batch_file.open() as f:
            entries = json.load(f)
        print(f"[INFO] Batch mode: {len(entries)} run(s) from {batch_file}")
        for i, entry in enumerate(entries):
            print(f"\n{'=' * 60}")
            print(f"[Batch {i + 1}/{len(entries)}] repo_id={entry['repo_id']}")
            print("=" * 60)
            args.repo_id = entry["repo_id"]
            args.calibration_repo_id = entry.get("calibration_repo_id", args.calibration_repo_id)
            _run_single(args)
    else:
        _run_single(args)


def _run_single(args) -> None:
    """Run the full compute+calibrate+evaluate pipeline for one (repo_id, calibration_repo_id) pair."""
    device = torch.device(args.device)

    _all_methods = ["action_entropy"]
    raw = args.methods
    if raw is None or raw == [] or raw == ["all"]:
        methods = list(_all_methods)
    else:
        methods = [m for m in raw if m != "all"]

    n_samples: int = getattr(args, "n_samples", 1)
    if "action_entropy" in methods and n_samples < 2:
        print("[WARN] action_entropy requires n_samples >= 2.  Setting n_samples=16.")
        n_samples = 16
    print(f"[INFO] Requested methods: {methods}  n_samples={n_samples}")

    # Validate — remove methods whose dependencies are missing
    # RND checkpoint path is resolved now; actual training (if needed) happens after
    # the policy is loaded so we can reuse it for embedding collection.
    rnd_ckpt: Path | None = None
    if "rnd" in methods:
        if not args.calibration_repo_id:
            print("[WARN] 'rnd' requires --calibration_repo_id for training.  Removing it.")
            methods.remove("rnd")
        else:
            rnd_ckpt = Path(
                args.rnd_checkpoint
                if args.rnd_checkpoint
                else f"outputs/rnd/{Path(args.calibration_repo_id).name}.pt"
            )

    logpzo_ckpt: Path | None = None
    if "logpzo" in methods:
        if not args.calibration_repo_id:
            print("[WARN] 'logpzo' requires --calibration_repo_id for training.  Removing it.")
            methods.remove("logpzo")
        else:
            logpzo_ckpt = Path(
                args.logpzo_checkpoint
                if args.logpzo_checkpoint
                else f"outputs/logpzo/{Path(args.calibration_repo_id).name}.pt"
            )

    if "pca_kmeans" in methods and not args.calibration_repo_id:
        print("[WARN] 'pca_kmeans' requires --calibration_repo_id for fitting.  Removing it.")
        methods.remove("pca_kmeans")
    if "similarity" in methods and not args.calibration_repo_id:
        print("[WARN] 'similarity' requires --calibration_repo_id for fitting.  Removing it.")
        methods.remove("similarity")

    # --- Dataset ---
    dataset = LeRobotDataset(args.repo_id, root=args.root)
    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    dataset_root = Path(dataset.root)
    total_eps = len(dataset.meta.episodes)
    n_eps = total_eps if args.num_episode is None else min(args.num_episode, total_eps)
    print(f"[INFO] Dataset: {dataset_root}")
    print(f"[INFO] Processing {n_eps}/{total_eps} episode(s).")

    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Model path ---
    model_path = resolve_model_path(dataset_root, args.model_path)
    print(f"[INFO] Model path: {model_path}")

    # --- Policy ---
    policy = load_policy(model_path, device)

    # --- Embedding extractor (shared by rnd, pca_kmeans, similarity, logpzo) ---
    extractor = None
    needs_emb = any(m in methods for m in ("rnd", "pca_kmeans", "similarity", "logpzo"))
    if needs_emb:
        from tools.comparison.methods.embedding_extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        extractor.attach(policy)
        print("[INFO] EmbeddingExtractor attached to policy.model.encoder")

    # --- Train logpZO if checkpoint missing (or --recalibrate) ---
    if "logpzo" in methods and logpzo_ckpt is not None:
        if logpzo_ckpt.exists() and not args.recalibrate:
            print(f"[logpZO] Using cached checkpoint: {logpzo_ckpt}")
        else:
            print(
                f"[logpZO] {'Forced retraining' if args.recalibrate else 'No checkpoint found — training'}: {logpzo_ckpt}"
            )
            _train_logpzo_checkpoint(
                calibration_repo_id=args.calibration_repo_id,
                calibration_root=args.calibration_root or args.root,
                policy=policy,
                extractor=extractor,
                device=device,
                output_path=logpzo_ckpt,
                in_dim=args.logpzo_in_dim,
                n_epochs=args.logpzo_epochs,
                cal_fraction=args.logpzo_cal_fraction,
            )
    # --- Train RND if checkpoint missing (or --recalibrate) ---
    if "rnd" in methods and rnd_ckpt is not None:
        if rnd_ckpt.exists() and not args.recalibrate:
            print(f"[RND] Using cached checkpoint: {rnd_ckpt}")
        else:
            print(
                f"[RND] {'Forced retraining' if args.recalibrate else 'No checkpoint found — training'}: {rnd_ckpt}"
            )
            _train_rnd_checkpoint(
                calibration_repo_id=args.calibration_repo_id,
                calibration_root=args.calibration_root or args.root,
                policy=policy,
                extractor=extractor,
                device=device,
                output_path=rnd_ckpt,
            )
    # --- Build detectors (calibration runs first to fit PCAKMeans / thresholds) ---
    detectors: list[BaseDetector] = []

    if "stac" in methods:
        from tools.comparison.methods.stac import STACDetector

        stac_gamma = args.stac_gamma if args.stac_gamma == "median" else float(args.stac_gamma)
        detectors.append(
            STACDetector(
                execution_horizon=args.stac_execution_horizon,
                gamma=stac_gamma,
            )
        )
        print(f"[INFO] STAC: k={args.stac_execution_horizon}, gamma={args.stac_gamma}")

    if "rnd" in methods and rnd_ckpt is not None:
        from tools.comparison.methods.rnd import RNDDetector

        detectors.append(RNDDetector(checkpoint_path=rnd_ckpt, extractor=extractor, device=device))

    if "pca_kmeans" in methods:
        from tools.comparison.methods.pca_kmeans import PCAKMeansDetector

        detectors.append(
            PCAKMeansDetector(
                extractor=extractor,
                emb_dim=args.pca_kmeans_emb_dim,
                n_clusters=args.pca_kmeans_clusters,
                device=device,
            )
        )
        print(f"[INFO] PCAKMeans: emb_dim={args.pca_kmeans_emb_dim}, n_clusters={args.pca_kmeans_clusters}")

    if "similarity" in methods:
        from tools.comparison.methods.similarity import SimilarityDetector

        detectors.append(SimilarityDetector(extractor=extractor))
        print("[INFO] SimilarityDetector (Mahalanobis) added.")

    if "logpzo" in methods and logpzo_ckpt is not None:
        from tools.comparison.methods.logpzo import LogpZODetector

        detectors.append(
            LogpZODetector(
                checkpoint_path=logpzo_ckpt,
                extractor=extractor,
                device=device,
                in_dim=args.logpzo_in_dim,
            )
        )
        print(f"[INFO] LogpZODetector loaded (in_dim={args.logpzo_in_dim}).")

    if "action_entropy" in methods:
        from tools.comparison.methods.action_entropy import ActionEntropyDetector

        detectors.append(ActionEntropyDetector())
        use_vae = _policy_uses_vae(policy)
        print(
            f"[INFO] ActionEntropyDetector added  "
            f"(n_samples={n_samples}, sample_latent={use_vae and n_samples > 1})."
        )

    cal_step_scores: dict[str, list[list[float]]] = {}
    cal_scores_dir: Path | None = None
    if args.calibration_repo_id:
        # Try to load cached calibration results unless --recalibrate is set
        _cached = False
        if not args.recalibrate:
            _cal_ds_tmp = LeRobotDataset(args.calibration_repo_id, root=args.calibration_root or args.root)
            _candidate_dir = Path(_cal_ds_tmp.root) / "comparison"
            _needed = [_candidate_dir / f"{d.name}_scores.jsonl" for d in detectors]
            threshold_needed = [output_dir / f"{d.name}_threshold.json" for d in detectors]
            # Invalidate cache if any checkpoint is newer than the cached scores.
            # This prevents stale-cache bugs where the checkpoint was retrained after
            # calibration scores were computed, causing threshold/test-score mismatch.
            _ckpt_files: list[Path] = []
            if logpzo_ckpt is not None and logpzo_ckpt.exists():
                _ckpt_files.append(logpzo_ckpt)
            if rnd_ckpt is not None and rnd_ckpt.exists():
                _ckpt_files.append(rnd_ckpt)
            _cache_mtime = min((p.stat().st_mtime for p in _needed if p.exists()), default=0.0)
            _ckpt_newer = any(c.stat().st_mtime > _cache_mtime for c in _ckpt_files)
            if _ckpt_newer:
                print("[INFO] Checkpoint(s) newer than cached calibration scores — forcing recalibration.")
            if (
                all(p.exists() for p in _needed)
                and all(p.exists() for p in threshold_needed)
                and not _ckpt_newer
            ):
                from tools.comparison.evaluate_comparison import load_scores_by_episode

                cal_scores_dir = _candidate_dir
                for d in detectors:
                    by_ep = load_scores_by_episode(_candidate_dir / f"{d.name}_scores.jsonl")
                    cal_step_scores[d.name] = [by_ep[ep] for ep in sorted(by_ep)]
                # Re-fit embedding-based detectors from cached embeddings
                emb_cache = output_dir / "calibration_embeddings.npy"
                if emb_cache.exists():
                    emb_arr = np.load(emb_cache)
                    from tools.comparison.methods.pca_kmeans import PCAKMeansDetector
                    from tools.comparison.methods.similarity import SimilarityDetector

                    for d in detectors:
                        if isinstance(d, (PCAKMeansDetector, SimilarityDetector)):
                            d.fit(emb_arr)
                    # RNDDetector is pre-trained; no fitting needed from emb_cache
                    print(f"[INFO] Re-fitted detectors from {emb_cache} ({len(emb_arr)} embeddings)")
                else:
                    print(
                        "[WARN] calibration_embeddings.npy not found — "
                        "embedding detectors not fitted.  Re-run with --recalibrate."
                    )
                print(f"[INFO] Loaded cached calibration from {_candidate_dir}")
                _cached = True
            del _cal_ds_tmp

        if not _cached:
            cal_step_scores, cal_scores_dir = run_calibration(
                calibration_repo_id=args.calibration_repo_id,
                calibration_root=args.calibration_root or args.root,
                policy=policy,
                detectors=detectors,
                extractor=extractor,
                device=device,
                delta=args.calibration_delta,
                output_dir=output_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                n_samples=n_samples,
            )

    # --- Score main dataset ---
    from tqdm import tqdm

    all_rows: dict[str, list[dict]] = {d.name: [] for d in detectors}

    for ep_idx in tqdm(range(n_eps), desc="Scoring episodes"):
        ep_from, ep_to = get_episode_bounds(dataset, ep_idx)
        ep_results = compute_episode(
            episode_idx=ep_idx,
            episode_from=ep_from,
            episode_to=ep_to,
            dataset=dataset,
            policy=policy,
            detectors=detectors,
            extractor=extractor,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_samples=n_samples,
        )
        for name, rows in ep_results.items():
            all_rows[name].extend(rows)

    for d in detectors:
        out_path = output_dir / f"{d.name}_scores.jsonl"
        BaseDetector.save_scores(all_rows[d.name], out_path)
        print(f"[INFO] Saved {len(all_rows[d.name])} rows → {out_path}")

    if extractor is not None:
        extractor.detach()

    if cal_scores_dir is not None:
        print(
            f"\n[INFO] To run barplot evaluation separately:\n"
            f"  uv run python tools/comparison/evaluate_comparison.py \\\n"
            f"    --scores_dir {output_dir} \\\n"
            f"    --cal_scores_dir {cal_scores_dir}\n"
        )

    # --- Inline evaluation ---
    labels_path = Path(args.labels) if args.labels else output_dir / "episode_labels.json"
    if labels_path.exists() and cal_step_scores:
        _run_inline_evaluation(
            detectors=detectors,
            cal_step_scores=cal_step_scores,
            all_rows=all_rows,
            labels_path=labels_path,
            num_train=args.num_train,
            num_cal=args.num_cal,
            alpha=args.alpha,
            output_dir=output_dir,
            debug=args.debug,
            dataset_root=dataset_root,
            step_quantile_methods=set(args.step_quantile_methods or ["logpzo"]),
            step_quantile_alpha=args.step_quantile_alpha,
            tail_steps=args.tail_steps,
        )
    elif labels_path.exists() and not cal_step_scores:
        print(
            "[WARN] episode_labels.json found but no calibration data available "
            "(--calibration_repo_id not set).  Skipping inline evaluation.\n"
            "       Run evaluate_comparison.py separately if calibration was done earlier."
        )


if __name__ == "__main__":
    main()
