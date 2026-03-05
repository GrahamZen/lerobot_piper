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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from tools.failure.offline_utils import (
        load_failure_config,
        load_failure_handling_json,
        load_failure_metrics_jsonl,
        replay_checkpoint_series,
    )
except ModuleNotFoundError:
    from offline_utils import (
        load_failure_config,
        load_failure_handling_json,
        load_failure_metrics_jsonl,
        replay_checkpoint_series,
    )

DEFAULT_SAFETY_MARGIN = 40


def _load_timing_series_from_npz(npz_path: Path):
    if not npz_path.exists():
        return None, None

    with np.load(npz_path, allow_pickle=False) as data:
        series = {}
        for key in data.files:
            arr = np.asarray(data[key], dtype=np.float64)
            if arr.ndim != 1 or arr.size == 0:
                continue
            series[key] = arr * 1e3

    if not series:
        return None, None

    min_len = min(len(v) for v in series.values())
    series = {k: v[:min_len] for k, v in series.items()}
    steps = np.arange(min_len, dtype=np.int64)
    return series, steps


def save_timing_curves_plot(
    dataset_root: Path,
    timing_plot_path: str | None = None,
    timing_npz_path: str | None = None,
):
    npz_path = (
        Path(timing_npz_path).expanduser()
        if timing_npz_path
        else dataset_root / "meta" / "policy_timing_steps.npz"
    )

    series, x_vals = _load_timing_series_from_npz(npz_path)

    if series is None:
        print(f"[WARN] No timing npz found at: {npz_path}")
        return None

    ordered_keys = [
        "obs_get_s",
        "obs_process_s",
        "build_observation_frame_s",
        "prepare_observation_s",
        "preprocess_s",
        "policy_select_action_s",
        "postprocess_s",
        "policy_to_robot_action_s",
        "send_action_s",
        "dataset_write_s",
        "ui_queue_s",
        "loop_total_s",
        "total_s",
    ]
    plot_keys = [k for k in ordered_keys if k in series] + [k for k in series if k not in ordered_keys]

    if timing_plot_path is None:
        timing_plot_file = dataset_root / "meta" / "policy_timing_curves.png"
    else:
        timing_plot_file = Path(timing_plot_path).expanduser()
    timing_plot_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 8))
    for key in plot_keys:
        plt.plot(x_vals, series[key], linewidth=1.2, label=key)

    plt.title("Policy Stage Timing Curves")
    plt.xlabel("Profiled Step")
    plt.ylabel("Latency (ms)")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(timing_plot_file, dpi=180)
    plt.close()

    print(f"[INFO] Saved timing curves plot to: {timing_plot_file}")
    return timing_plot_file


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


def visualize_dataset(
    repo_id,
    root=None,
    stride=7,
    checkpoint_signal_config=None,
    save_timing_plot=True,
    timing_plot_path=None,
    timing_npz_path=None,
):
    try:
        dataset = LeRobotDataset(repo_id, root=None)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    if save_timing_plot:
        save_timing_curves_plot(
            Path(dataset.root),
            timing_plot_path=timing_plot_path,
            timing_npz_path=timing_npz_path,
        )

    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    total_episodes = len(dataset.meta.episodes)

    failure_metrics = load_failure_metrics_jsonl(dataset.root)

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

    failure_handling_cfg = load_failure_handling_json(dataset.root)
    failure_cfg = load_failure_config(dataset.root)

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

    if td_cp_threshold is None:
        td_cp_threshold = float(failure_cfg.metrics.temporal_disagreement.cp_threshold)

    safety_margin = DEFAULT_SAFETY_MARGIN
    if checkpoint_signal_config:
        td_cfg = failure_cfg.metrics.temporal_disagreement
        td_cfg.window_size = int(checkpoint_signal_config.get("window_size", td_cfg.window_size))
        td_cfg.eval_delay = int(checkpoint_signal_config.get("eval_delay", td_cfg.eval_delay))
        td_cfg.valley_lookback = int(checkpoint_signal_config.get("valley_lookback", td_cfg.valley_lookback))
        td_cfg.valley_lookahead = int(
            checkpoint_signal_config.get("valley_lookahead", td_cfg.valley_lookahead)
        )
        td_cfg.smoothing_sigma = float(
            checkpoint_signal_config.get("smoothing_sigma", td_cfg.smoothing_sigma)
        )
        valley_prominence = checkpoint_signal_config.get("valley_prominence", td_cfg.valley_prominence)
        td_cfg.valley_prominence = (
            td_cfg.valley_prominence if valley_prominence is None else float(valley_prominence)
        )
        td_cfg.__post_init__()
        safety_margin = int(checkpoint_signal_config.get("safety_margin", DEFAULT_SAFETY_MARGIN))

    if failure_metrics:
        (
            smoothed_td_by_step,
            previous_checkpoint_by_step,
            checkpoint_flag_by_step,
            recent_checkpoints_by_step,
        ) = replay_checkpoint_series(
            failure_metrics,
            failure_cfg,
            safety_margin=safety_margin,
            dataset_episodes=dataset.meta.episodes,
        )

        # Step 2: pass TD in and compute Mahalanobis-based fused features
        advanced_metrics = build_advanced_fusion_metrics(
            failure_metrics,
            smoothed_td_by_step,
            window_size=20,
            dataset_episodes=dataset.meta.episodes,
            mahal_cfg=mahal_cfg,
        )

        # Step 3: identify red markers based on thresholds
        if mahal_cp_threshold is None:
            mahal_cp_threshold = float("inf")

        for step in sorted(failure_metrics.keys()):
            raw_td = float(failure_metrics[step].get("temporal_disagreement", 0.0))
            smooth_td = float(smoothed_td_by_step.get(step, raw_td))

            # TD decision (align with online detect_failure using smoothed signal)
            if smooth_td > td_cp_threshold:
                td_failed_steps.append(step)
                td_failed_step_set.add(step)
            # Mahalanobis-distance decision
            if advanced_metrics[step].get("mahalanobis_dist", 0.0) > mahal_cp_threshold:
                mahal_failed_step_set.add(step)

    camera_names = [key.replace("observation.images.", "") for key in dataset.meta.camera_keys]

    if failure_metrics:
        camera_views = [rrb.Spatial2DView(origin=f"cameras/{cam}") for cam in camera_names]
        id_overlay_view = rrb.TextDocumentView(
            name="Episode ID",
            origin="overlay/episode_id",
        )

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
                        rrb.TimeSeriesView(name="Following Error", origin="metrics/following_error"),
                        rrb.TimeSeriesView(
                            name="Previous Checkpoint Step", origin="metrics/previous_checkpoint_step"
                        ),
                        rrb.TimeSeriesView(name="Attention Entropy", origin="metrics/attention_entropy"),
                        rrb.TimeSeriesView(
                            name="Mahalanobis Distance", origin="metrics/mahalanobis_distance"
                        ),
                        rrb.TimeSeriesView(name="Endpoint Shift", origin="metrics/endpoint_shift"),
                        rrb.TimeSeriesView(name="Action Jerk", origin="metrics/action_jerk"),
                        rrb.TimeSeriesView(
                            name="Attention Entropy Downward Slope",
                            origin="metrics/attention_entropy_downward_slope",
                        ),
                        rrb.TimeSeriesView(name="Checkpoint Flag", origin="metrics/checkpoint_flag"),
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="Checkpoint -5", origin="checkpoints/cp_0"),
                        rrb.Spatial2DView(name="Checkpoint -4", origin="checkpoints/cp_1"),
                        rrb.Spatial2DView(name="Checkpoint -3", origin="checkpoints/cp_2"),
                        rrb.Spatial2DView(name="Checkpoint -2", origin="checkpoints/cp_3"),
                        rrb.Spatial2DView(name="Checkpoint -1", origin="checkpoints/cp_4"),
                    ),
                ),
                rrb.Vertical(*camera_views, id_overlay_view, row_shares=[6] * len(camera_views) + [1]),
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
        rr.log(
            "metrics/temporal_disagreement_smoothed/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )
        # Register red markers for Mahalanobis distance
        rr.log(
            "metrics/mahalanobis_fusion_dist/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )
        rr.log(
            "metrics/previous_checkpoint_step/failed_markers",
            rr.SeriesPoints(colors=[255, 0, 0], markers="circle", marker_sizes=6.0),
            static=True,
        )

    global_step = 0
    prev_attention_entropy = None
    prev_attention_step = None
    warned_insufficient_checkpoint_steps: set[int] = set()

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
            rr.log("overlay/episode_id", rr.TextDocument(f"{episode_idx}"), static=False)

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

                rr.log("metrics/following_error", rr.Scalars(m.get("following_error", 0.0)))
                attention_entropy = float(m.get("attention_entropy", 0.0))
                rr.log("metrics/attention_entropy", rr.Scalars(attention_entropy))

                if prev_attention_entropy is None or prev_attention_step is None or i == prev_attention_step:
                    attention_entropy_downward_slope = 0.0
                else:
                    attention_entropy_downward_slope = (prev_attention_entropy - attention_entropy) / float(
                        i - prev_attention_step
                    )
                rr.log(
                    "metrics/attention_entropy_downward_slope",
                    rr.Scalars(attention_entropy_downward_slope),
                )

                prev_attention_entropy = attention_entropy
                prev_attention_step = i
                rr.log("metrics/mahalanobis_distance", rr.Scalars(m.get("mahalanobis_distance", 0.0)))
                rr.log("metrics/endpoint_shift", rr.Scalars(m.get("endpoint_shift", 0.0)))
                rr.log("metrics/action_jerk", rr.Scalars(m.get("action_jerk", 0.0)))
                rr.log(
                    "metrics/previous_checkpoint_step/value",
                    rr.Scalars(previous_checkpoint_by_step.get(i, np.nan)),
                )
                rr.log("metrics/checkpoint_flag", rr.Scalars(checkpoint_flag_by_step.get(i, 0.0)))

                if i in td_failed_step_set:
                    rr.log("metrics/temporal_disagreement/failed_markers", rr.Scalars(raw_td))
                    rr.log("metrics/temporal_disagreement_smoothed/failed_markers", rr.Scalars(smooth_td))
                    rr.log(
                        "metrics/previous_checkpoint_step/failed_markers",
                        rr.Scalars(previous_checkpoint_by_step.get(i, np.nan)),
                    )
                if i in mahal_failed_step_set:
                    rr.log(
                        "metrics/mahalanobis_fusion_dist/failed_markers", rr.Scalars(adv["mahalanobis_dist"])
                    )

                is_failed = i in td_failed_step_set or i in mahal_failed_step_set
                if is_failed:
                    cps = recent_checkpoints_by_step.get(i, [])
                    cps = cps[-5:]
                    if len(cps) < 5 and i not in warned_insufficient_checkpoint_steps:
                        print(
                            f"[WARN] Step {i}: only {len(cps)} checkpoint(s) available (<5). "
                            "Missing slots are rendered as black placeholders."
                        )
                        warned_insufficient_checkpoint_steps.add(i)
                    padded_cps = [None] * (5 - len(cps)) + cps
                    for cp_idx, cp_step in enumerate(padded_cps):
                        if cp_step is not None:
                            try:
                                cp_item = dataset[cp_step]
                                top_key = next(
                                    (k for k in cp_item if "image" in k and "middle" in k.lower()), None
                                )
                                if not top_key:
                                    top_key = next((k for k in cp_item if "image" in k), None)

                                if top_key:
                                    img_data = cp_item[top_key]
                                    if isinstance(img_data, dict) and "bytes" in img_data:
                                        import io

                                        cp_img = Image.open(io.BytesIO(img_data["bytes"]))
                                    else:
                                        cp_img = img_data.numpy() if hasattr(img_data, "numpy") else img_data
                                        if cp_img.ndim == 3 and cp_img.shape[0] <= 4:
                                            cp_img = np.transpose(cp_img, (1, 2, 0))
                                    rr.log(f"checkpoints/cp_{cp_idx}", rr.Image(cp_img))
                                else:
                                    rr.log(
                                        f"checkpoints/cp_{cp_idx}",
                                        rr.Image(np.zeros((10, 10, 3), dtype=np.uint8)),
                                    )
                            except Exception:
                                rr.log(
                                    f"checkpoints/cp_{cp_idx}",
                                    rr.Image(np.zeros((10, 10, 3), dtype=np.uint8)),
                                )
                        else:
                            rr.log(
                                f"checkpoints/cp_{cp_idx}", rr.Image(np.zeros((10, 10, 3), dtype=np.uint8))
                            )
                else:
                    for cp_idx in range(5):
                        rr.log(f"checkpoints/cp_{cp_idx}", rr.Image(np.zeros((10, 10, 3), dtype=np.uint8)))

    print("\nDone streaming to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LeRobot dataset directly to Rerun.")
    parser.add_argument("--repo_id", type=str, help="Dataset repository ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")
    parser.add_argument("--stride", type=int, default=7, help="Visualization stride (speed)")
    parser.add_argument(
        "--save_timing_plot",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Generate and save a standalone multi-stage timing curves plot.",
    )
    parser.add_argument(
        "--timing_plot_path",
        type=str,
        default=None,
        help="Output png path for timing curves. Default: <dataset_root>/meta/policy_timing_curves.png",
    )
    parser.add_argument(
        "--timing_npz_path",
        type=str,
        default=None,
        help="Optional timing npz path. Default: <dataset_root>/meta/policy_timing_steps.npz",
    )

    args = parser.parse_args()
    visualize_dataset(
        args.repo_id,
        args.root,
        args.stride,
        save_timing_plot=args.save_timing_plot,
        timing_plot_path=args.timing_plot_path,
        timing_npz_path=args.timing_npz_path,
    )
