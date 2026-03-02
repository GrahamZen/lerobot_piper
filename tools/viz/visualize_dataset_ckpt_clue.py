"""
This script streams an entire LeRobot dataset to Rerun for continuous visualization.
It renders camera streams and the robot's 3D posture across all episodes in a single timeline.

Key Features:
- Continuous playback: Automatically scrolls through all episodes in the dataset.
- 3D Visualization: Uses forward kinematics to render the robot's arm and gripper state.
- Global Timeline: Displays a "global_step" timeline for the entire dataset plus per-episode steps.
- Episode Overlays: Shows the current episode index as a large 3D text overlay.
- Stride control: Adjust the playback speed using the --stride argument.

Usage:
    python tools/viz/visualize_dataset.py --repo_id local/lerobot_pick_and_place --stride 7
"""

import argparse
import json
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# =========================
# Checkpoint Detection Config
# =========================
CHECKPOINT_SIGNAL_CONFIG = {
    "window_size": 31,
    "eval_delay": 15,
    "valley_lookback": 8,
    "valley_lookahead": 8,
    "smoothing_sigma": 2.0,
    "valley_prominence": None,
    "safety_margin": 40,
}

ATTENTION_SLOPE_HIGHLIGHT_THRESHOLD = 0.05


def load_checkpoint_signal_config_from_dataset(dataset_root):
    print(f"[INFO] Checking dataset root for record_config.json: {dataset_root}")
    record_config_path = Path(dataset_root) / "meta" / "record_config.json"
    if not record_config_path.exists():
        print(f"[WARN] record_config.json not found at {record_config_path}")
        return {}

    print(f"[INFO] Found record_config.json at {record_config_path}")
    try:
        with open(record_config_path) as f:
            record_config = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse record_config.json: {e}")
        return {}

    pretrained_path = record_config.get("pretrained_path")
    if not pretrained_path and isinstance(record_config.get("policy"), dict):
        pretrained_path = record_config["policy"].get("pretrained_path")

    if not pretrained_path:
        for key in ("model", "train", "training"):
            section = record_config.get(key)
            if isinstance(section, dict) and section.get("pretrained_path"):
                pretrained_path = section["pretrained_path"]
                break

    if not pretrained_path:
        print("[WARN] Could not find 'pretrained_path' in record_config.json")
        return {}

    pretrained_path = Path(pretrained_path).expanduser()
    print(f"[INFO] Resolved pretrained_path to: {pretrained_path}")

    failure_handling_json_path = pretrained_path / "failure_handling.json"
    if not failure_handling_json_path.exists():
        print(f"[WARN] failure_handling.json not found at primary path {failure_handling_json_path}")
        nested_candidate = pretrained_path / "pretrained_model" / "failure_handling.json"
        if nested_candidate.exists():
            print(f"[INFO] Found failure_handling.json at fallback path {nested_candidate}")
            failure_handling_json_path = nested_candidate
        else:
            print(f"[WARN] failure_handling.json not found at fallback path {nested_candidate} either.")
            return {}
    else:
        print(f"[INFO] Found failure_handling.json at {failure_handling_json_path}")

    try:
        with open(failure_handling_json_path) as f:
            failure_handling = json.load(f)
        print(f"[INFO] Reading failure handling config from: {failure_handling_json_path}")
        print("\n[RAW JSON CONFIG IN USE]")
        print(json.dumps(failure_handling, ensure_ascii=False, indent=2))
        print("-" * 40)
    except Exception as e:
        print(f"[ERROR] Failed to read failure handling config at {failure_handling_json_path}: {e}")
        traceback.print_exc()
        return {}
    return failure_handling


def build_failed_points_table_markdown(
    failed_steps, failure_metrics, previous_checkpoint_by_step, cp_threshold
):
    title = "# Failed Points Summary"
    threshold_line = (
        f"\n\n- `cp_threshold`: `{cp_threshold:.6f}`"
        if cp_threshold is not None
        else "\n\n- `cp_threshold`: `N/A`"
    )
    table_header = "\n\n| # | failed_step (x) | temporal_disagreement (y_td) | previous_checkpoint_step (y_cp) |\n| :--- | :--- | :--- | :--- |"
    if not failed_steps:
        return f"{title}{threshold_line}{table_header}\n| - | - | - | - |"
    rows = []
    for idx, step in enumerate(failed_steps, start=1):
        td = float(failure_metrics.get(step, {}).get("temporal_disagreement", 0.0))
        prev_cp = previous_checkpoint_by_step.get(step, np.nan)
        prev_cp_text = "-" if (isinstance(prev_cp, float) and np.isnan(prev_cp)) else f"{float(prev_cp):.0f}"
        rows.append(f"| {idx} | {int(step)} | {td:.6f} | {prev_cp_text} |")
    return f"{title}{threshold_line}{table_header}\n" + "\n".join(rows)


def build_checkpoint_series(
    failure_metrics,
    window_size=31,
    eval_delay=15,
    valley_lookback=8,
    valley_lookahead=8,
    smoothing_sigma=2.0,
    valley_prominence=None,
    safety_margin=40,
):
    if not failure_metrics:
        return {}, {}, {}
    steps = np.array(sorted(failure_metrics.keys()), dtype=np.int64)
    smoothed_by_step = {}
    checkpoint_flag_by_step = {}
    previous_checkpoint_by_step = {}

    eval_delay = max(eval_delay, valley_lookahead)
    min_required = eval_delay + valley_lookback + 1
    window_size = max(window_size, min_required)

    recent_raw = deque(maxlen=window_size)
    recent_steps = deque(maxlen=window_size)
    checkpoint_set = set()
    checkpoint_history = []

    for step in steps:
        step_int = int(step)
        disagreement = float(failure_metrics[step_int].get("temporal_disagreement", 0.0))
        recent_raw.append(disagreement)
        recent_steps.append(step_int)

        if len(recent_raw) >= min_required:
            smoothed_window = gaussian_filter1d(np.array(recent_raw), sigma=smoothing_sigma)
            eval_idx = len(recent_raw) - 1 - eval_delay
            eval_val = smoothed_window[eval_idx]
            eval_step = recent_steps[eval_idx]
            smoothed_by_step[eval_step] = eval_val

            past_vals = smoothed_window[eval_idx - valley_lookback : eval_idx]
            future_vals = smoothed_window[eval_idx + 1 : eval_idx + 1 + valley_lookahead]

            current_prominence = (
                valley_prominence
                if valley_prominence is not None
                else max(1e-6, 0.35 * float(np.std(smoothed_window)))
            )
            is_valley = True
            if eval_val > min(past_vals) or eval_val >= min(future_vals):
                is_valley = False
            else:
                if (
                    max(past_vals) - eval_val < current_prominence
                    or max(future_vals) - eval_val < current_prominence
                ):
                    is_valley = False
            if is_valley:
                checkpoint_set.add(eval_step)
                checkpoint_history.append(float(eval_step))

        safe_checkpoint = np.nan
        for cp in reversed(checkpoint_history):
            if step_int - cp >= safety_margin:
                safe_checkpoint = cp
                break

        checkpoint_flag_by_step[step_int] = 1.0 if step_int in checkpoint_set else 0.0
        previous_checkpoint_by_step[step_int] = safe_checkpoint

    for step in steps:
        if step not in smoothed_by_step:
            smoothed_by_step[step] = float(failure_metrics[step].get("temporal_disagreement", 0.0))
    return smoothed_by_step, previous_checkpoint_by_step, checkpoint_flag_by_step


# =====================================================================
# Updated module: includes Mahalanobis distance computation
# =====================================================================
def build_advanced_fusion_metrics(
    failure_metrics, smoothed_td_by_step, window_size=20, dataset_episodes=None, mahal_cfg=None
):
    if not failure_metrics:
        return {}

    steps = sorted(failure_metrics.keys())
    entropies = np.array([float(failure_metrics[step].get("attention_entropy", 0.0)) for step in steps])

    step_to_episode = {}
    if dataset_episodes is not None:
        for ep_meta in dataset_episodes:
            from_idx = int(
                ep_meta["dataset_from_index"]
                if not isinstance(ep_meta["dataset_from_index"], list)
                else ep_meta["dataset_from_index"][0]
            )
            to_idx = int(
                ep_meta["dataset_to_index"]
                if not isinstance(ep_meta["dataset_to_index"], list)
                else ep_meta["dataset_to_index"][0]
            )
            for step in range(from_idx, to_idx):
                step_to_episode[step] = (from_idx, to_idx)

    # 1. Compute local variance
    variances = np.zeros_like(entropies)
    for i, step in enumerate(steps):
        ep_bounds = step_to_episode.get(step)
        from_idx = ep_bounds[0] if ep_bounds else 0

        start_idx = i
        while start_idx > 0 and start_idx > i - window_size and steps[start_idx - 1] >= from_idx:
            start_idx -= 1

        full_window = entropies[start_idx : i + 1]
        if len(full_window) > 3:
            variances[i] = np.var(full_window)

    # Extract Mahalanobis distance configuration
    mu = None
    inv_cov = None
    if mahal_cfg and "mu" in mahal_cfg and "inv_cov" in mahal_cfg:
        mu = np.array(mahal_cfg["mu"])
        inv_cov = np.array(mahal_cfg["inv_cov"])

    advanced_metrics = {}
    for i, step in enumerate(steps):
        var_val = float(variances[i])
        td_val = smoothed_td_by_step.get(step, 0.0)

        # Use raw TD for Mahalanobis distance (aligned with the threshold calculation script)
        raw_td = float(failure_metrics[step].get("temporal_disagreement", 0.0))

        # Compute Mahalanobis distance for the fused score
        mahal_dist = 0.0
        if mu is not None and inv_cov is not None:
            # Build the 2D feature vector X for the current step
            x_t = np.array([raw_td, var_val])
            diff = x_t - mu
            # Formula: sqrt((X - mu)^T * Sigma^{-1} * (X - mu))
            left = np.dot(diff, inv_cov)
            mahal_dist = np.sqrt(np.abs(np.dot(left, diff)))

        # Keep the original weighted version for comparison
        alpha_scale = 20.0
        fusion_score = td_val + (alpha_scale * var_val)

        advanced_metrics[step] = {
            "local_variance": var_val,
            "fusion_sum": fusion_score,
            "mahalanobis_dist": float(mahal_dist),
        }
    return advanced_metrics


# =====================================================================


def visualize_dataset(repo_id, root=None, stride=7, checkpoint_signal_config=None):
    expanded_root = Path(root).expanduser() if root else None
    dataset_path = Path(repo_id).expanduser()
    if not dataset_path.is_dir() and expanded_root:
        dataset_path = expanded_root / repo_id

    if dataset_path.is_dir():
        print(f"Found local dataset at: {dataset_path}")
        repo_id = str(dataset_path)
        root = None
    else:
        root = expanded_root

    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    total_episodes = len(dataset.meta.episodes)

    failure_metrics = {}
    metrics_path = Path(dataset.root) / "failure_metrics.jsonl"
    if metrics_path.exists():
        with open(metrics_path) as f:
            for line in f:
                try:
                    m = json.loads(line)
                    if "step" in m:
                        failure_metrics[int(m["step"])] = m
                except (KeyError, ValueError, TypeError):
                    continue

    smoothed_td_by_step = {}
    previous_checkpoint_by_step = {}
    checkpoint_flag_by_step = {}

    td_failed_steps = []
    td_failed_step_set = set()
    mahal_failed_step_set = set()

    td_cp_threshold = None
    mahal_cp_threshold = None
    mahal_cfg = {}
    advanced_metrics = {}

    failure_handling_cfg = load_checkpoint_signal_config_from_dataset(dataset.root)

    if "metrics" in failure_handling_cfg:
        metrics_cfg = failure_handling_cfg["metrics"]
        if "temporal_disagreement" in metrics_cfg:
            td_cp_threshold = metrics_cfg["temporal_disagreement"].get("cp_threshold")
        if "fusion_mahalanobis" in metrics_cfg:
            mahal_cfg = metrics_cfg["fusion_mahalanobis"]
            mahal_cp_threshold = mahal_cfg.get("cp_threshold")

    # Backward compatibility (if still stored at top level)
    if td_cp_threshold is None and "cp_threshold" in failure_handling_cfg:
        td_cp_threshold = failure_handling_cfg.get("cp_threshold")

    signal_cfg = dict(CHECKPOINT_SIGNAL_CONFIG)
    if checkpoint_signal_config:
        signal_cfg.update(checkpoint_signal_config)

    if failure_metrics:
        # Step 1: compute smoothed TD
        smoothed_td_by_step, previous_checkpoint_by_step, checkpoint_flag_by_step = build_checkpoint_series(
            failure_metrics,
            window_size=signal_cfg["window_size"],
            eval_delay=signal_cfg["eval_delay"],
            valley_lookback=signal_cfg["valley_lookback"],
            valley_lookahead=signal_cfg["valley_lookahead"],
            smoothing_sigma=signal_cfg["smoothing_sigma"],
            valley_prominence=signal_cfg["valley_prominence"],
            safety_margin=signal_cfg["safety_margin"],
        )

        # Step 2: pass TD in and compute Mahalanobis-based fused features
        advanced_metrics = build_advanced_fusion_metrics(
            failure_metrics,
            smoothed_td_by_step,
            window_size=20,
            dataset_episodes=dataset.meta.episodes,
            mahal_cfg=mahal_cfg,
        )

        # Step 3: identify red markers based on conformal prediction thresholds
        if td_cp_threshold is None:
            td_cp_threshold = float("inf")
        if mahal_cp_threshold is None:
            mahal_cp_threshold = float("inf")

        for step in sorted(failure_metrics.keys()):
            # TD decision
            if float(failure_metrics[step].get("temporal_disagreement", 0.0)) > td_cp_threshold:
                td_failed_steps.append(step)
                td_failed_step_set.add(step)
            # Mahalanobis-distance decision
            if advanced_metrics[step].get("mahalanobis_dist", 0.0) > mahal_cp_threshold:
                mahal_failed_step_set.add(step)

    camera_names = [key.replace("observation.images.", "") for key in dataset.meta.camera_keys]

    if failure_metrics:
        camera_views = [rrb.Spatial2DView(origin=f"cameras/{cam}") for cam in camera_names]

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            name="Temporal Disagreement", origin="metrics/temporal_disagreement"
                        ),
                        rrb.TimeSeriesView(
                            name="[ADV] Attention Local Variance", origin="metrics/attention_local_variance"
                        ),
                        # --- Core new panel: Mahalanobis distance and its failure markers ---
                        rrb.TimeSeriesView(
                            name="[ULTIMATE] Mahalanobis Fusion Dist",
                            origin="metrics/mahalanobis_fusion_dist",
                        ),
                        rrb.TimeSeriesView(
                            name="Temporal Disagreement Smoothed",
                            origin="metrics/temporal_disagreement_smoothed",
                        ),
                    ),
                    rrb.Vertical(
                        rrb.TextDocumentView(
                            origin="summary/failed_points_table", name="Failed Points Table"
                        ),
                    ),
                ),
                rrb.Vertical(
                    rrb.Vertical(*camera_views) if camera_views else rrb.Spatial3DView(origin="simulation"),
                    row_shares=[5, 1],
                ),
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        )
    else:
        blueprint = None

    rr.init("LeRobot Dataset Visualizer", spawn=True)
    if blueprint:
        rr.send_blueprint(blueprint)

    # Register static marker styles in Rerun
    if failure_metrics:
        rr.log(
            "metrics/temporal_disagreement/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )
        # Register red markers for Mahalanobis distance
        rr.log(
            "metrics/mahalanobis_fusion_dist/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )

        failed_table_md = build_failed_points_table_markdown(
            td_failed_steps, failure_metrics, previous_checkpoint_by_step, td_cp_threshold
        )
        rr.log(
            "summary/failed_points_table",
            rr.TextDocument(failed_table_md, media_type=rr.MediaType.MARKDOWN),
            static=True,
        )

    global_step = 0

    for episode_idx in range(total_episodes):
        print(f"Streaming Episode {episode_idx}/{total_episodes}...", end="\r")
        ep_meta = dataset.meta.episodes[episode_idx]
        from_idx = int(
            ep_meta["dataset_from_index"]
            if not isinstance(ep_meta["dataset_from_index"], list)
            else ep_meta["dataset_from_index"][0]
        )
        to_idx = int(
            ep_meta["dataset_to_index"]
            if not isinstance(ep_meta["dataset_to_index"], list)
            else ep_meta["dataset_to_index"][0]
        )

        for i in range(from_idx, to_idx, stride):
            rr.set_time_sequence("global_step", global_step)
            global_step += 1
            rr.log("overlay/episode_id", rr.TextDocument(f"# Episode {episode_idx}"), static=False)

            item = None
            try:
                item = dataset[i]
            except (IndexError, KeyError, RuntimeError, OSError) as exc:
                print(f"Skipping step {i}: failed to load item ({exc})")

            if item is None:
                continue

            for img_key in [k for k in item if "image" in k]:
                img_data = item[img_key]
                clean_key = img_key.replace("observation.images.", "")
                if isinstance(img_data, dict) and "bytes" in img_data:
                    import io

                    rr.log(f"cameras/{clean_key}", rr.Image(Image.open(io.BytesIO(img_data["bytes"]))))
                else:
                    arr = img_data.numpy() if hasattr(img_data, "numpy") else img_data
                    if arr.ndim == 3 and arr.shape[0] <= 4:
                        arr = np.transpose(arr, (1, 2, 0))
                    rr.log(f"cameras/{clean_key}", rr.Image(arr))

            if failure_metrics:
                m = failure_metrics.get(i, {})
                raw_td = float(m.get("temporal_disagreement", 0.0))
                smooth_td = smoothed_td_by_step.get(i, raw_td)

                rr.log("metrics/temporal_disagreement", rr.Scalars(raw_td))
                rr.log("metrics/temporal_disagreement_smoothed", rr.Scalars(smooth_td))

                adv = advanced_metrics.get(
                    i, {"local_variance": 0.0, "fusion_sum": 0.0, "mahalanobis_dist": 0.0}
                )
                rr.log("metrics/attention_local_variance", rr.Scalars(adv["local_variance"]))

                # --- Log Mahalanobis distance and its detected red markers ---
                rr.log("metrics/mahalanobis_fusion_dist", rr.Scalars(adv["mahalanobis_dist"]))

                if i in td_failed_step_set:
                    rr.log("metrics/temporal_disagreement/failed_markers", rr.Scalars(raw_td))
                if i in mahal_failed_step_set:
                    rr.log(
                        "metrics/mahalanobis_fusion_dist/failed_markers", rr.Scalars(adv["mahalanobis_dist"])
                    )

    print("\nDone streaming to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LeRobot dataset directly to Rerun.")
    parser.add_argument("--repo_id", type=str, help="Dataset repository ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")
    parser.add_argument("--stride", type=int, default=7, help="Visualization stride (speed)")

    args = parser.parse_args()
    visualize_dataset(args.repo_id, args.root, args.stride)
