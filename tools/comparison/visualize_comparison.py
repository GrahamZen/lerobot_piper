"""Visualise multiple failure-detection methods side-by-side in Rerun.

Each method's anomaly scores are pre-computed (by ``compute_comparison.py``
or by the existing failure-handling pipeline) and stored as JSONL files.
This script reads those files and streams them into Rerun, drawing one
TimeSeriesView per method stacked vertically, with camera frames below.

Supported score sources
-----------------------
- **Our method (TD)**: reads ``failure_metrics.jsonl`` from the dataset root
  (field ``td_smoothed``).  Always included when the file is present.
- **Any method computed by** ``compute_comparison.py``: reads
  ``<scores_dir>/<method>_scores.jsonl`` (field ``score``).

Usage example::

    python tools/comparison/visualize_comparison.py \\
        --repo_id my_org/my_dataset \\
        --root /data/datasets \\
        --scores_dir /data/datasets/my_dataset/comparison \\
        --methods similarity \\
        --stride 1 \\
        --num_episode 5
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
import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "failure"))

from offline_utils import get_episode_bounds, load_failure_metrics_jsonl  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from tools.comparison.methods.base import BaseDetector  # noqa: E402

# ---------------------------------------------------------------------------
# Score loading
# ---------------------------------------------------------------------------

# Display name, colour (R,G,B) and optional threshold to draw
_METHOD_STYLE: dict[str, dict] = {
    "td": {"label": "Ours (TD smoothed)", "color": (52, 152, 219)},  # blue
    "similarity": {"label": "Similarity (Mahalanobis)", "color": (230, 126, 34)},  # orange
    "pca_kmeans": {"label": "PCA+KMeans", "color": (155, 89, 182)},  # purple
}

_DEFAULT_COLOR = (150, 150, 150)


def _load_td_scores(dataset_root: Path) -> dict[int, float]:
    """Load our TD-smoothed scores from ``failure_metrics.jsonl``."""
    rows = load_failure_metrics_jsonl(dataset_root)
    return {k: float(v.get("td_smoothed", 0.0)) for k, v in rows.items() if v.get("td_smoothed") is not None}


def _load_calibrated_threshold(scores_dir: Path, method: str) -> float | None:
    """Load calibrated constant threshold from ``<scores_dir>/<method>_threshold.json``.

    Always returns a scalar ``float`` (or ``None``).  Legacy list-valued
    (time-varying) thresholds are collapsed to their maximum.
    """
    path = scores_dir / f"{method}_threshold.json"
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        raw = data["threshold"]
        if isinstance(raw, list):
            v = float(max(raw))
            print(f"[INFO] {method} threshold={v:.6f} (collapsed from time-varying list, len={len(raw)})")
            return v
        v = float(raw)
        print(f"[INFO] {method} threshold={v:.6f} (δ={data.get('delta')}, N={data.get('n_episodes')})")
        return v
    except Exception:
        return None


def _load_td_threshold(dataset_root: Path) -> float | None:
    """Try to read failure_threshold from failure_handling.json in the model dir."""
    record_cfg = dataset_root / "meta" / "record_config.json"
    if not record_cfg.exists():
        return None
    try:
        with record_cfg.open() as f:
            rc = json.load(f)
        pretrained = rc.get("pretrained_path") or (rc.get("policy") or {}).get("pretrained_path")
        if not pretrained:
            return None
        fh_json = Path(pretrained).expanduser() / "failure_handling.json"
        if not fh_json.exists():
            return None
        with fh_json.open() as f:
            fh = json.load(f)
        det = fh.get("detector") or fh
        return float(det.get("failure_threshold", det.get("threshold", 0.0))) or None
    except Exception:
        return None


def load_all_scores(
    dataset_root: Path,
    scores_dir: Path,
    methods: list[str],
) -> dict[str, dict[int, float]]:
    """Return ``{method_name: {global_step: score}}`` for all requested methods.

    Always attempts to load ``td`` from ``failure_metrics.jsonl``.
    Other methods are loaded from ``<scores_dir>/<method>_scores.jsonl``.
    """
    all_scores: dict[str, dict[int, float]] = {}

    # TD (our method) — always load if present
    td = _load_td_scores(dataset_root)
    if td:
        all_scores["td"] = td
        print(f"[INFO] TD scores: {len(td)} steps")

    for method in methods:
        if method == "td":
            continue  # already handled
        path = scores_dir / f"{method}_scores.jsonl"
        scores = BaseDetector.load_scores(path)
        if scores:
            all_scores[method] = scores
            print(f"[INFO] {method} scores: {len(scores)} steps")
        else:
            print(f"[WARN] No scores found for method '{method}' at {path}")

    return all_scores


# ---------------------------------------------------------------------------
# Rerun blueprint builder
# ---------------------------------------------------------------------------


def build_blueprint(
    methods: list[str],
    has_cameras: bool,
) -> rrb.Blueprint:
    """Build a stacked vertical layout: one plot per method + cameras row."""

    def _ts_view(method: str) -> rrb.TimeSeriesView:
        style = _METHOD_STYLE.get(method, {})
        label = style.get("label", method)
        contents = [f"comparison/{method}/**"]
        return rrb.TimeSeriesView(name=label, contents=contents)

    views: list[Any] = [_ts_view(m) for m in methods]

    if has_cameras:
        cam_row = rrb.Horizontal(
            rrb.Vertical(
                rrb.TextDocumentView(name="Episode", origin="overlay/episode_id"),
                rrb.Spatial2DView(name="Camera", origin="cameras/stitched"),
                row_shares=[1, 8],
            ),
        )
        views.append(cam_row)
        row_shares = [1.0] * len(methods) + [1.5]
    else:
        row_shares = [1.0] * len(methods)

    return rrb.Blueprint(
        rrb.Vertical(*views, row_shares=row_shares),
        collapse_panels=True,
    )


# ---------------------------------------------------------------------------
# Camera helper (re-used from visualize_dataset_ckpt_clue.py)
# ---------------------------------------------------------------------------


def _log_stitched_cameras(item: dict, camera_keys: list[str]) -> None:
    cam_imgs: list[np.ndarray] = []
    for key in camera_keys:
        img = item.get(key)
        if img is None:
            continue
        if isinstance(img, dict) and "bytes" in img:
            continue
        arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        cam_imgs.append(arr)

    if not cam_imgs:
        return

    target_h = cam_imgs[0].shape[0]
    rows: list[np.ndarray] = []
    for img in cam_imgs:
        if img.shape[0] != target_h:
            from PIL import Image as PILImage

            w = int(img.shape[1] * target_h / img.shape[0])
            img = np.array(PILImage.fromarray(img).resize((w, target_h), PILImage.BILINEAR))
        rows.append(img)
    rr.log("cameras/stitched", rr.Image(np.concatenate(rows, axis=1)))


# ---------------------------------------------------------------------------
# Static series styling
# ---------------------------------------------------------------------------


def _init_series_styles(methods: list[str], thresholds: dict[str, float]) -> None:
    """Register static colour/marker styles for each method's Rerun paths."""
    for method in methods:
        style = _METHOD_STYLE.get(method, {})
        color = style.get("color", _DEFAULT_COLOR)
        rr.log(
            f"comparison/{method}/score",
            rr.SeriesLines(colors=list(color), names=style.get("label", method)),
            static=True,
        )
        rr.log(
            f"comparison/{method}/failure_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
            static=True,
        )
        if method in thresholds:
            rr.log(
                f"comparison/{method}/threshold",
                rr.SeriesLines(colors=[255, 165, 0], names=f"{style.get('label', method)} threshold"),
                static=True,
            )


# ---------------------------------------------------------------------------
# Main visualisation loop
# ---------------------------------------------------------------------------


def visualize(
    repo_id: str,
    root: str | None,
    scores_dir: str,
    methods: list[str],
    stride: int,
    num_episode: int | None,
) -> None:
    dataset = LeRobotDataset(repo_id, root=root)
    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    dataset_root = Path(dataset.root)
    scores_path = Path(scores_dir) if scores_dir else dataset_root / "comparison"

    total_eps = len(dataset.meta.episodes)
    n_eps = total_eps if num_episode is None else min(num_episode, total_eps)
    print(f"[INFO] Visualising {n_eps}/{total_eps} episode(s).")

    # Load all scores
    all_scores = load_all_scores(dataset_root, scores_path, methods)
    if not all_scores:
        print("[WARN] No scores loaded — only camera stream will be shown.")

    active_methods = list(all_scores.keys())
    if not active_methods:
        active_methods = methods  # still build blueprint slots even if empty

    # Load thresholds: calibrated files first, then fall back to TD config file
    thresholds: dict[str, float] = {}
    for method in active_methods:
        t = _load_calibrated_threshold(scores_path, method)
        if t is not None:
            thresholds[method] = t
    # TD fallback: read from failure_handling.json
    if "td" not in thresholds:
        t = _load_td_threshold(dataset_root)
        if t is not None:
            thresholds["td"] = t

    # --- Rerun setup ---
    camera_keys = dataset.meta.camera_keys
    blueprint = build_blueprint(active_methods, has_cameras=bool(camera_keys))
    rr.init("Failure Detection Comparison", spawn=True)
    rr.send_blueprint(blueprint)
    _init_series_styles(active_methods, thresholds)

    # --- Build frame list ---
    all_frames: list[tuple[int, int]] = []
    episode_from_idx: dict[int, int] = {}
    for ep_idx in range(n_eps):
        from_idx, to_idx = get_episode_bounds(dataset, ep_idx)
        episode_from_idx[ep_idx] = from_idx
        for frame_idx in range(from_idx, to_idx, stride):
            all_frames.append((ep_idx, frame_idx))

    all_frame_indices = [f for _, f in all_frames]
    loader = DataLoader(
        Subset(dataset, all_frame_indices),
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=lambda b: b[0],
        shuffle=False,
    )

    from tqdm import tqdm

    current_ep = -1
    for global_step, ((ep_idx, frame_idx), item) in enumerate(
        tqdm(zip(all_frames, loader, strict=True), total=len(all_frames), desc="Streaming")
    ):
        if ep_idx != current_ep:
            current_ep = ep_idx

        rr.set_time("step", sequence=frame_idx)
        rr.set_time("global_step", sequence=global_step)
        rr.log("overlay/episode_id", rr.TextDocument(f"Episode {ep_idx}"), static=False)

        # Camera
        _log_stitched_cameras(item, camera_keys)

        # Scores — one path per method
        for method, scores_map in all_scores.items():
            score = scores_map.get(frame_idx)
            if score is None:
                continue
            rr.log(f"comparison/{method}/score", rr.Scalars(score))

            threshold_val = thresholds.get(method)
            if threshold_val is None:
                continue
            rr.log(f"comparison/{method}/threshold", rr.Scalars(threshold_val))
            if score > threshold_val:
                rr.log(f"comparison/{method}/failure_markers", rr.Scalars(score))

    print("[INFO] Done streaming to Rerun.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise failure-detection comparison in Rerun.")
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument(
        "--scores_dir",
        default=None,
        help="Directory containing <method>_scores.jsonl files.  Defaults to <dataset_root>/comparison/.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Methods to load (e.g. similarity pca_kmeans).  "
        "Omit to auto-detect all *_scores.jsonl files in --scores_dir.  "
        "'td' is always included when failure_metrics.jsonl is present.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride (default: 1 = every frame)")
    parser.add_argument("--num_episode", type=int, default=None)
    args = parser.parse_args()

    # Auto-detect methods from existing *_scores.jsonl files when not specified
    methods = args.methods
    if methods is None:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        _ds = LeRobotDataset(args.repo_id, root=args.root)
        _sdir = Path(args.scores_dir) if args.scores_dir else Path(_ds.root) / "comparison"
        methods = sorted(p.stem.replace("_scores", "") for p in _sdir.glob("*_scores.jsonl"))
        if methods:
            print(f"[INFO] Auto-detected methods: {methods}")
        else:
            methods = []

    visualize(
        repo_id=args.repo_id,
        root=args.root,
        scores_dir=args.scores_dir,
        methods=methods,
        stride=args.stride,
        num_episode=args.num_episode,
    )


if __name__ == "__main__":
    main()
