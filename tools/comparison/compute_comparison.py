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

    # ------------------------------------------------------------------
    # Single policy pass: collect (chunks, embedding) per episode per step.
    # DataLoader workers prefetch frames in the background; policy runs
    # one frame at a time to stay identical to the original code path.
    # ------------------------------------------------------------------
    # cal_data[ep_idx] = list of (chunks_1frame, emb | None, state | None)
    cal_data: list[list[tuple[Tensor, np.ndarray | None, np.ndarray | None]]] = []

    from tqdm import tqdm

    for ep_idx in tqdm(range(n_eps), desc="[Calibration] episodes"):
        ep_from, ep_to = get_episode_bounds(cal_dataset, ep_idx)
        ep_data: list[tuple[Tensor, np.ndarray | None, np.ndarray | None]] = []

        loader = _make_loader(cal_dataset, list(range(ep_from, ep_to)), batch_size, num_workers)
        for batch in tqdm(loader, desc=f"  ep {ep_idx + 1}", leave=False):
            obs = {
                k: (v if isinstance(v, Tensor) else torch.as_tensor(v)).to(device, non_blocking=True)
                for k, v in batch.items()
                if k.startswith("observation.") or k == "observation_state"
            }
            with torch.no_grad():
                chunks_batch = policy.predict_action_chunk(obs)  # (B, horizon, action_dim)
            embs_batch = extractor.last_batch if needs_emb else None  # (B, D) or None
            states_val = (
                batch.get("observation.state") if isinstance(batch.get("observation.state"), Tensor) else None
            )

            actual_b = chunks_batch.shape[0]
            for i in range(actual_b):
                emb = embs_batch[i].copy() if embs_batch is not None else None
                state: np.ndarray | None = None
                if states_val is not None:
                    state = states_val[i].cpu().float().numpy()
                ep_data.append((chunks_batch[i : i + 1], emb, state))
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
            if state is not None:
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
                rows.append(
                    {
                        "global_step": gs,
                        "episode": ep_idx,
                        "step_in_episode": step_in_ep,
                        "score": score,
                    }
                )
                gs += 1
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
) -> dict[str, list[dict[str, Any]]]:
    """Run all detectors on one episode; return ``{name: [score_row, ...]}``.

    Each row: ``{global_step, episode, step_in_episode, score}``.

    Frames are prefetched by *num_workers* DataLoader workers so IO runs
    in the background while the GPU runs the current batch.
    """
    results: dict[str, list[dict]] = {d.name: [] for d in detectors}
    for d in detectors:
        d.reset()

    from tqdm import tqdm

    indices = list(range(episode_from, episode_to))
    loader = _make_loader(dataset, indices, batch_size, num_workers)
    ep_step = 0

    for batch in tqdm(loader, desc=f"  ep {episode_idx + 1}", leave=False):
        obs = {
            k: (v if isinstance(v, Tensor) else torch.as_tensor(v)).to(device, non_blocking=True)
            for k, v in batch.items()
            if k.startswith("observation.") or k == "observation_state"
        }
        with torch.no_grad():
            chunks_batch = policy.predict_action_chunk(obs)  # (B, horizon, action_dim)
        embs_batch = extractor.last_batch if extractor is not None else None  # (B, D) or None
        states_val = (
            batch.get("observation.state") if isinstance(batch.get("observation.state"), Tensor) else None
        )

        actual_b = chunks_batch.shape[0]
        for i in range(actual_b):
            frame_idx = episode_from + ep_step
            if extractor is not None and embs_batch is not None:
                extractor.set_embedding(embs_batch[i])
            if extractor is not None and states_val is not None:
                extractor.set_state(states_val[i].cpu().float().numpy())
            for d in detectors:
                score = d.update(ep_step, chunks_batch[i : i + 1])
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
        _detect_max_threshold,
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

        print(
            f"[{method}] train={_nt}  cal={_nc}  "
            f"test={len(common_eps)} (success={len(common_eps) - n_test_fail}, failure={n_test_fail})"
        )

        y_true, y_pred, first_steps = _detect_max_threshold(
            train_seqs + cal_seqs_split,
            test_scores,
            test_labels,
            alpha=alpha,
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
        choices=["pca_kmeans", "similarity", "all"],
        help="Methods to run (default: all available).  Pass 'all' explicitly or omit to run every method.",
    )
    parser.add_argument("--output_dir", default=None, help="Defaults to <dataset_root>/comparison/")
    parser.add_argument("--num_episode", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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

    _all_methods = ["pca_kmeans", "similarity"]
    raw = args.methods
    if raw is None or raw == [] or raw == ["all"]:
        methods = list(_all_methods)
    else:
        methods = [m for m in raw if m != "all"]
    print(f"[INFO] Requested methods: {methods}")

    # Validate — remove methods whose dependencies are missing
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

    # --- Embedding extractor (shared by pca_kmeans, similarity) ---
    extractor = None
    needs_emb = any(m in methods for m in ("pca_kmeans", "similarity"))
    if needs_emb:
        from tools.comparison.methods.embedding_extractor import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        extractor.attach(policy)
        print("[INFO] EmbeddingExtractor attached to policy.model.encoder")

    # --- Build detectors (calibration runs first to fit PCAKMeans / thresholds) ---
    detectors: list[BaseDetector] = []

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
            if all(p.exists() for p in _needed) and all(p.exists() for p in threshold_needed):
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
        )
    elif labels_path.exists() and not cal_step_scores:
        print(
            "[WARN] episode_labels.json found but no calibration data available "
            "(--calibration_repo_id not set).  Skipping inline evaluation.\n"
            "       Run evaluate_comparison.py separately if calibration was done earlier."
        )


if __name__ == "__main__":
    main()
