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
MAX_VLM_PAIR_SLOTS = 6
_VLM_EMPTY_IMAGE = np.zeros((16, 16, 3), dtype=np.uint8)


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_vlm_record_view(rec_dir: Path, rec_index: int, meta: dict) -> dict:
    parts = sorted(meta.get("request_parts", []), key=lambda x: int(x.get("index", 0)))

    sequence_lines = [
        "Request parts raw sequence (original order):",
    ]
    for p in parts:
        sequence_lines.append(f"- idx={p.get('index')} | type={p.get('type')} | file={p.get('file')}")
    if not parts:
        sequence_lines.append("- <empty>")

    response_path = rec_dir / "response.txt"
    response_text = (
        response_path.read_text(encoding="utf-8") if response_path.exists() else "<No response.txt>"
    )

    def _read_part_text(part: dict) -> str:
        p_file = part.get("file")
        p_path = rec_dir / p_file if p_file else rec_dir / ""
        p_text = p_path.read_text(encoding="utf-8") if p_file and p_path.exists() else f"[Missing] {p_file}"
        return f"[part {part.get('index')}] {p_file}\n{p_text}"

    def _read_part_image(part: dict) -> np.ndarray:
        p_file = part.get("file")
        p_path = rec_dir / p_file if p_file else rec_dir / ""
        if p_file and p_path.exists():
            img = Image.open(p_path).convert("RGB")
            return np.asarray(img)
        return _VLM_EMPTY_IMAGE.copy()

    pair_rows: list[dict] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        part_type = part.get("type")

        if part_type == "text":
            row_text = _read_part_text(part)
            row_image = _VLM_EMPTY_IMAGE.copy()
            if i + 1 < len(parts) and parts[i + 1].get("type") == "image":
                row_image = _read_part_image(parts[i + 1])
                i += 2
            else:
                i += 1
            pair_rows.append({"text": row_text, "image": row_image})
            continue

        if part_type == "image":
            row_text = f"[no paired text]\n[part {part.get('index')}] {part.get('file')}"
            row_image = _read_part_image(part)
            pair_rows.append({"text": row_text, "image": row_image})
            i += 1
            continue

        i += 1

    if not pair_rows:
        pair_rows.append({"text": "<No request parts>", "image": _VLM_EMPTY_IMAGE.copy()})

    header = (
        f"record #{rec_index} | episode={meta.get('episode')} | step={meta.get('step')} "
        f"| selected_index={meta.get('selected_index')} | selected_step={meta.get('selected_step')}"
    )

    return {
        "episode": _safe_int(meta.get("episode")),
        "step": _safe_int(meta.get("step")),
        "record_index": rec_index,
        "header": header,
        "sequence_text": "\n".join(sequence_lines),
        "pair_rows": pair_rows,
        "response_text": response_text,
    }


def _load_vlm_message_records(dataset_root: Path) -> list[dict]:
    debug_root = dataset_root / "vlm" / "debug_records"
    if not debug_root.exists():
        return []

    session_dirs = sorted([p for p in debug_root.iterdir() if p.is_dir()])
    records: list[dict] = []
    for session_dir in session_dirs:
        manifest_path = session_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        for rec in sorted(manifest.get("records", []), key=lambda x: int(x.get("index", 0))):
            rec_index = int(rec.get("index", 0))
            rec_dir = session_dir / rec.get("dir", "")
            meta_path = rec_dir / "meta.json"
            if not meta_path.exists():
                continue

            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            record_view = _build_vlm_record_view(rec_dir, rec_index, meta)
            record_view["session_dir"] = str(session_dir)
            records.append(record_view)

    return records


def _select_vlm_record_for_step(vlm_records: list[dict], episode_idx: int, step: int, max_step_gap: int = 3):
    if not vlm_records:
        return None

    episode_candidates = {episode_idx, episode_idx + 1}

    def _in_episode_candidates(record: dict) -> bool:
        ep = record.get("episode")
        return ep in episode_candidates if isinstance(ep, int) else False

    records_in_episode = [r for r in vlm_records if _in_episode_candidates(r)]

    exact = [r for r in records_in_episode if r.get("step") == step]
    if exact:
        return exact[-1]

    # In practice VLM record `step` can lag/lead failure trigger by 1-2 steps.
    # Use nearest record within a small tolerance to keep panels aligned with failure events.
    nearby: list[dict] = []
    for r in records_in_episode:
        step_value = r.get("step")
        if not isinstance(step_value, int):
            continue
        if abs(step_value - step) <= max_step_gap:
            nearby.append(r)
    if nearby:
        selected = sorted(
            nearby,
            key=lambda x: (abs(x.get("step", 10**9) - step), -int(x.get("record_index", -1))),
        )[0]
        selected_step = selected.get("step")
        if selected_step != step:
            print(
                f"[WARN] VLM record fuzzy match: failure at step {step}, but matched VLM record at step {selected_step}"
            )
        return selected

    candidates = []
    for r in records_in_episode:
        step_value = r.get("step")
        if not isinstance(step_value, int):
            continue
        if step_value <= step and (step - step_value) <= max_step_gap:
            candidates.append(r)
    if candidates:
        selected = sorted(candidates, key=lambda x: (x.get("step"), x.get("record_index", -1)))[-1]
        selected_step = selected.get("step")
        if selected_step != step:
            print(
                f"[WARN] VLM record fuzzy match: failure at step {step}, but matched VLM record at step {selected_step}"
            )
        return selected

    return None


def _log_vlm_record_windows(record: dict | None):
    if record is None:
        rr.log("vlm_message/metadata", rr.TextDocument("<No VLM record matched current step>"))
        for idx in range(MAX_VLM_PAIR_SLOTS):
            rr.log(f"vlm_message/pairs/{idx}/text", rr.TextDocument("<No paired text>"))
            rr.log(f"vlm_message/pairs/{idx}/image", rr.Image(_VLM_EMPTY_IMAGE))
        rr.log("vlm_message/response", rr.TextDocument("<No response>"))
        return

    rr.log(
        "vlm_message/metadata",
        rr.TextDocument(f"{record['header']}\n\n{record['sequence_text']}"),
    )
    for idx in range(MAX_VLM_PAIR_SLOTS):
        if idx < len(record["pair_rows"]):
            row = record["pair_rows"][idx]
            rr.log(f"vlm_message/pairs/{idx}/text", rr.TextDocument(row["text"]))
            rr.log(f"vlm_message/pairs/{idx}/image", rr.Image(row["image"]))
        else:
            rr.log(f"vlm_message/pairs/{idx}/text", rr.TextDocument("<No paired text>"))
            rr.log(f"vlm_message/pairs/{idx}/image", rr.Image(_VLM_EMPTY_IMAGE))
    rr.log("vlm_message/response", rr.TextDocument(record["response_text"]))


def _log_checkpoint_windows(dataset, recent_checkpoints_by_step: dict, step: int):
    cps = recent_checkpoints_by_step.get(step, [])
    cps = cps[-5:]
    padded_cps = [None] * (5 - len(cps)) + cps
    for cp_idx, cp_step in enumerate(padded_cps):
        if cp_step is not None:
            try:
                cp_item = dataset[cp_step]
                top_key = next((k for k in cp_item if "image" in k and "middle" in k.lower()), None)
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
                    rr.log(f"checkpoints/cp_{cp_idx}", rr.Image(np.zeros((10, 10, 3), dtype=np.uint8)))
            except Exception:
                rr.log(f"checkpoints/cp_{cp_idx}", rr.Image(np.zeros((10, 10, 3), dtype=np.uint8)))
        else:
            rr.log(f"checkpoints/cp_{cp_idx}", rr.Image(np.zeros((10, 10, 3), dtype=np.uint8)))


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
    dataset_root = Path(dataset.root)
    use_vlm_panels = (dataset_root / "vlm").exists()
    if use_vlm_panels:
        vlm_records = _load_vlm_message_records(dataset_root)
        if vlm_records:
            print(f"[INFO] Loaded {len(vlm_records)} VLM debug record(s) from dataset directory.")
        else:
            print("[WARN] No VLM debug records found under <dataset_root>/vlm/debug_records.")
    else:
        vlm_records = []
        print("[INFO] No <dataset_root>/vlm folder found. Falling back to checkpoint image windows.")

    failure_metrics = load_failure_metrics_jsonl(dataset.root)

    smoothed_td_by_step = {}
    previous_checkpoint_by_step = {}
    checkpoint_flag_by_step = {}
    detect_failure_by_step = {}

    td_failed_steps = []
    td_failed_step_set = set()

    # Load required config from model's failure_handling.json (raises error if not found)
    failure_handling_cfg = load_failure_handling_json(dataset.root, required=True)
    failure_cfg = load_failure_config(dataset.root, required=True)

    if (
        "metrics" not in failure_handling_cfg
        or "temporal_disagreement" not in failure_handling_cfg["metrics"]
    ):
        raise ValueError(
            "ERROR: failure_handling.json must contain 'metrics' -> 'temporal_disagreement' section"
        )

    metrics_cfg = failure_handling_cfg["metrics"]
    td_config = metrics_cfg["temporal_disagreement"]

    # Extract required cp_threshold
    td_cp_threshold = td_config.get("cp_threshold")
    if td_cp_threshold is None:
        raise ValueError(
            "ERROR: 'cp_threshold' not found in failure_handling.json temporal_disagreement config"
        )

    td_cp_threshold = float(td_cp_threshold)
    print(f"Loaded cp_threshold from failure_handling.json: {td_cp_threshold:.6f}")

    # safety_margin is optional (visualization-only parameter)
    safety_margin = td_config.get("safety_margin", DEFAULT_SAFETY_MARGIN)

    # Ensure failure_cfg uses the correct values from JSON (not defaults)
    # The FailureConfig object may use defaults due to schema validation issues
    td_cfg = failure_cfg.metrics.temporal_disagreement
    td_cfg.window_size = int(td_config.get("window_size", td_cfg.window_size))
    td_cfg.eval_delay = int(td_config.get("eval_delay", td_cfg.eval_delay))
    td_cfg.valley_lookback = int(td_config.get("valley_lookback", td_cfg.valley_lookback))
    td_cfg.valley_lookahead = int(td_config.get("valley_lookahead", td_cfg.valley_lookahead))
    td_cfg.smoothing_sigma = float(td_config.get("smoothing_sigma", td_cfg.smoothing_sigma))
    valley_prominence_json = td_config.get("valley_prominence", td_cfg.valley_prominence)
    td_cfg.valley_prominence = (
        td_cfg.valley_prominence if valley_prominence_json is None else float(valley_prominence_json)
    )
    td_cfg.cp_threshold = td_cp_threshold
    td_cfg.__post_init__()

    print("Using checkpoint detection parameters:")
    print(f"  window_size: {td_cfg.window_size}")
    print(f"  eval_delay: {td_cfg.eval_delay}")
    print(f"  valley_lookback: {td_cfg.valley_lookback}")
    print(f"  valley_lookahead: {td_cfg.valley_lookahead}")
    print(f"  smoothing_sigma: {td_cfg.smoothing_sigma}")
    print(f"  valley_prominence: {td_cfg.valley_prominence}")
    print(f"  safety_margin: {safety_margin}")

    # Allow command-line override of parameters
    if checkpoint_signal_config:
        print("Applying command-line overrides:")
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
        safety_margin = int(checkpoint_signal_config.get("safety_margin", safety_margin))
        print(f"  smoothing_sigma: {td_cfg.smoothing_sigma}")
        print(f"  valley_prominence: {td_cfg.valley_prominence}")

    if failure_metrics:
        (
            smoothed_td_by_step,
            previous_checkpoint_by_step,
            checkpoint_flag_by_step,
            recent_checkpoints_by_step,
            detect_failure_by_step,
        ) = replay_checkpoint_series(
            failure_metrics,
            failure_cfg,
            safety_margin=safety_margin,
            dataset_episodes=dataset.meta.episodes,
        )

        # Identify red markers based on FailureMetrics.detect_failure()
        for step in sorted(failure_metrics.keys()):
            # TD decision must exactly follow FailureMetrics.detect_failure()
            if detect_failure_by_step.get(step, False):
                td_failed_steps.append(step)
                td_failed_step_set.add(step)

    camera_names = [key.replace("observation.images.", "") for key in dataset.meta.camera_keys]

    if failure_metrics:
        camera_views = [rrb.Spatial2DView(origin=f"cameras/{cam}") for cam in camera_names]
        id_overlay_view = rrb.TextDocumentView(
            name="Episode ID",
            origin="overlay/episode_id",
        )

        if use_vlm_panels:
            # Create text views (left column)
            text_views = [
                rrb.TextDocumentView(
                    name=f"Window 3.{idx} - Request Pair Text",
                    origin=f"vlm_message/pairs/{idx}/text",
                )
                for idx in range(MAX_VLM_PAIR_SLOTS)
            ]
            # Create image views (right column)
            image_views = [
                rrb.Spatial2DView(
                    name=f"Window 3.{idx} - Request Pair Image",
                    origin=f"vlm_message/pairs/{idx}/image",
                )
                for idx in range(MAX_VLM_PAIR_SLOTS)
            ]

            middle_right_panel = rrb.Vertical(
                rrb.TextDocumentView(name="Window 1 - Message Metadata", origin="vlm_message/metadata"),
                rrb.Horizontal(
                    rrb.Vertical(*text_views),
                    rrb.Vertical(*image_views),
                    column_shares=[1, 1],
                ),
                rrb.TextDocumentView(name="Window 4 - VLM Response", origin="vlm_message/response"),
                row_shares=[0.5, 6, 1],
            )
        else:
            middle_right_panel = rrb.Vertical(
                rrb.Spatial2DView(name="Checkpoint -5", origin="checkpoints/cp_0"),
                rrb.Spatial2DView(name="Checkpoint -4", origin="checkpoints/cp_1"),
                rrb.Spatial2DView(name="Checkpoint -3", origin="checkpoints/cp_2"),
                rrb.Spatial2DView(name="Checkpoint -2", origin="checkpoints/cp_3"),
                rrb.Spatial2DView(name="Checkpoint -1", origin="checkpoints/cp_4"),
            )

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            name="Temporal Disagreement", origin="metrics/temporal_disagreement"
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
                        # rrb.TimeSeriesView(
                        #     name="Mahalanobis Distance", origin="metrics/mahalanobis_distance"
                        # ),
                        rrb.TimeSeriesView(name="Endpoint Shift", origin="metrics/endpoint_shift"),
                        rrb.TimeSeriesView(name="Action Jerk", origin="metrics/action_jerk"),
                        # rrb.TimeSeriesView(
                        #     name="Attention Entropy Downward Slope",
                        #     origin="metrics/attention_entropy_downward_slope",
                        # ),
                        rrb.TimeSeriesView(name="Checkpoint Flag", origin="metrics/checkpoint_flag"),
                    ),
                    middle_right_panel,
                    column_shares=[1, 1],
                ),
                rrb.Vertical(*camera_views, id_overlay_view, row_shares=[6] * len(camera_views) + [1]),
                column_shares=[4, 1],
            ),
            collapse_panels=True,
        )
    else:
        blueprint = None

    rr.init("LeRobot Dataset Visualizer", spawn=True)
    if blueprint:
        rr.send_blueprint(blueprint)

    # Register static marker styles in Rerun for all metrics
    if failure_metrics:
        metric_names = [
            "temporal_disagreement_smoothed",
            "previous_checkpoint_step",
        ]
        for metric_name in metric_names:
            rr.log(
                f"metrics/{metric_name}/failed_markers",
                rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
                static=True,
            )

    global_step = 0
    prev_attention_entropy = None
    prev_attention_step = None

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

                # Log red failure markers only when TD failure is detected (smoothed_td > cp_threshold)
                is_td_failed = i in td_failed_step_set
                if is_td_failed:
                    rr.log("metrics/temporal_disagreement_smoothed/failed_markers", rr.Scalars(smooth_td))
                    rr.log(
                        "metrics/previous_checkpoint_step/failed_markers",
                        rr.Scalars(previous_checkpoint_by_step.get(i, np.nan)),
                    )

                if use_vlm_panels:
                    matched_record = _select_vlm_record_for_step(vlm_records, episode_idx, i)
                    if is_td_failed:
                        _log_vlm_record_windows(matched_record)
                    else:
                        _log_vlm_record_windows(None)
                else:
                    if is_td_failed:
                        _log_checkpoint_windows(dataset, recent_checkpoints_by_step, i)
                    else:
                        for cp_idx in range(5):
                            rr.log(
                                f"checkpoints/cp_{cp_idx}", rr.Image(np.zeros((10, 10, 3), dtype=np.uint8))
                            )

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
