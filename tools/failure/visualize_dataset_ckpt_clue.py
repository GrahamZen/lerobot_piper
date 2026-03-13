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
import io
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

try:
    from tools.failure.action_entropy_memory import precompute_action_entropy_in_memory
except ModuleNotFoundError:
    from action_entropy_memory import precompute_action_entropy_in_memory

try:
    from tools.failure.offline_utils import (
        _extract_pretrained_path,
        load_failure_config,
        load_failure_handling_json,
        load_failure_metrics_jsonl,
        replay_checkpoint_series,
    )
except ModuleNotFoundError:
    from offline_utils import (
        _extract_pretrained_path,
        load_failure_config,
        load_failure_handling_json,
        load_failure_metrics_jsonl,
        replay_checkpoint_series,
    )

DEFAULT_SAFETY_MARGIN = 40
MAX_VLM_PAIR_SLOTS = 6
_VLM_EMPTY_IMAGE = np.zeros((16, 16, 3), dtype=np.uint8)
ACTION_ENTROPY_NPZ_FILENAME = "action_entropy.npz"


def _checkpoint_marker_origin_for_source(source: str) -> str:
    source = str(source).strip().lower()
    origin_map = {
        "temporal_disagreement": "metrics/temporal_disagreement_smoothed",
        "following_error": "metrics/following_error",
        "attention_entropy": "metrics/attention_entropy",
        "mahalanobis_distance": "metrics/mahalanobis_distance",
        "endpoint_shift": "metrics/endpoint_shift",
        "action_jerk": "metrics/action_jerk",
        "action_entropy": "metrics/action_entropy_compare/jsonl",
        "action_entropy_max_diff": "metrics/action_entropy_max_diff_compare/jsonl",
    }
    return origin_map.get(source, "metrics/temporal_disagreement_smoothed")


def _checkpoint_curve_label_by_origin(checkpoint_origin: str) -> dict[str, str]:
    labels = {
        "metrics/temporal_disagreement_smoothed": "Temporal Disagreement Smoothed",
        "metrics/following_error": "Following Error",
        "metrics/attention_entropy": "Attention Entropy",
        "metrics/mahalanobis_distance": "Mahalanobis Distance",
        "metrics/endpoint_shift": "Endpoint Shift",
        "metrics/action_jerk": "Action Jerk",
        "metrics/action_entropy_compare": "Action Entropy (JSONL vs Offline)",
        "metrics/action_entropy_max_diff_compare": "Action Sample Diversity (JSONL vs Offline)",
    }

    decorated = {}
    for origin, label in labels.items():
        if checkpoint_origin == f"{origin}/jsonl" or checkpoint_origin == origin:
            decorated[origin] = f"{label} (Checkpoint Source)"
        else:
            decorated[origin] = label
    return decorated


def _checkpoint_marker_value_for_step(
    *,
    source: str,
    metric_row: dict,
    smoothed_td: float,
    step: int,
    entropy_jsonl_by_step: dict[int, float],
    entropy_jsonl_max_diff_by_step: dict[int, float],
) -> float | None:
    source = str(source).strip().lower()

    if source == "temporal_disagreement":
        return float(smoothed_td)

    if source == "action_entropy":
        value = entropy_jsonl_by_step.get(step)
        return float(value) if value is not None else None

    if source == "action_entropy_max_diff":
        value = entropy_jsonl_max_diff_by_step.get(step)
        if value is None:
            for key in (
                "action_entropy_max_diff",
                "action_entropy_sample_max_diff",
                "action_sample_max_diff",
            ):
                raw = metric_row.get(key)
                if isinstance(raw, (int, float)):
                    value = float(raw)
                    break
        return float(value) if value is not None else None

    key_map = {
        "following_error": "following_error",
        "attention_entropy": "attention_entropy",
        "mahalanobis_distance": "mahalanobis_distance",
        "endpoint_shift": "endpoint_shift",
        "action_jerk": "action_jerk",
    }
    metric_key = key_map.get(source)
    if metric_key is None:
        return float(smoothed_td)

    value = metric_row.get(metric_key)
    return float(value) if isinstance(value, (int, float)) else None


def _resolve_pretrained_path_from_dataset_root(dataset_root: Path) -> Path | None:
    record_config_path = dataset_root / "meta" / "record_config.json"
    if not record_config_path.exists():
        return None

    try:
        with record_config_path.open("r", encoding="utf-8") as file:
            record_config = json.load(file)
    except Exception:
        return None

    return _extract_pretrained_path(record_config)


def _load_action_entropy_npz(npz_path: Path) -> tuple[dict[int, float], dict[int, float]] | None:
    if not npz_path.exists():
        return None

    with np.load(npz_path, allow_pickle=False) as data:
        if not {"step", "entropy", "max_diff"}.issubset(set(data.files)):
            return None
        steps = np.asarray(data["step"], dtype=np.int64)
        entropies = np.asarray(data["entropy"], dtype=np.float64)
        max_diffs = np.asarray(data["max_diff"], dtype=np.float64)

    if not (len(steps) == len(entropies) == len(max_diffs)):
        return None

    entropy_by_step = {int(step): float(entropies[idx]) for idx, step in enumerate(steps)}
    max_diff_by_step = {int(step): float(max_diffs[idx]) for idx, step in enumerate(steps)}
    return entropy_by_step, max_diff_by_step


def _load_action_entropy_from_failure_metrics(
    failure_metrics: dict[int, dict],
) -> tuple[dict[int, float], dict[int, float]]:
    entropy_by_step: dict[int, float] = {}
    max_diff_by_step: dict[int, float] = {}

    max_diff_keys = (
        "action_entropy_max_diff",
        "action_entropy_sample_max_diff",
        "action_sample_max_diff",
    )

    for step, metric_row in failure_metrics.items():
        entropy_value = metric_row.get("action_entropy")
        if isinstance(entropy_value, (int, float)):
            entropy_by_step[int(step)] = float(entropy_value)

        for key in max_diff_keys:
            max_diff_value = metric_row.get(key)
            if isinstance(max_diff_value, (int, float)):
                max_diff_by_step[int(step)] = float(max_diff_value)
                break

    return entropy_by_step, max_diff_by_step


def _save_action_entropy_npz(
    npz_path: Path,
    entropy_by_step: dict[int, float],
    max_diff_by_step: dict[int, float],
):
    steps = np.array(sorted(entropy_by_step.keys()), dtype=np.int64)
    entropy_vals = np.array([entropy_by_step[int(step)] for step in steps], dtype=np.float64)
    max_diff_vals = np.array([max_diff_by_step.get(int(step), np.nan) for step in steps], dtype=np.float64)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, step=steps, entropy=entropy_vals, max_diff=max_diff_vals)


def _precompute_entropy_peak_cp_by_step(
    entropy_by_step: dict[int, float],
    entropy_labels_by_step: dict[int, int],
    min_interval_len: int = 25,
) -> dict[int, int | None]:
    """For each step, return the Peak Entropy Point (step with the highest entropy)
    from the most recently *completed* free interval (label >= 1)
    that ended strictly before the current step.

    A free interval is a maximal run of consecutive steps whose entropy label is
    in the free/red region (label >= 1). An interval that is still open at the
    last observed step is
    intentionally excluded – it has not yet "ended" so it cannot be the
    *previous* interval.
    """
    if not entropy_by_step or not entropy_labels_by_step:
        return {}

    sorted_steps = sorted(entropy_by_step.keys())

    # ── Pass 1: identify completed free intervals ─────────────────────────────
    completed_intervals: list[tuple[int, int, int]] = []  # (start, end, peak_step)
    in_free_run: list[int] = []

    for step in sorted_steps:
        if entropy_labels_by_step.get(step, -1) >= 1:
            in_free_run.append(step)
        else:
            if in_free_run:
                if len(in_free_run) >= min_interval_len:
                    peak_step = max(in_free_run, key=lambda s: entropy_by_step[s])
                    completed_intervals.append((in_free_run[0], in_free_run[-1], peak_step))
                in_free_run = []
    # Intentionally do NOT add an open run at the end – it is not yet "previous".

    # ── Pass 2: for every step, find the latest completed interval ending before it
    peak_cp_by_step: dict[int, int | None] = {}
    interval_ptr = 0
    current_peak: int | None = None

    for step in sorted_steps:
        while interval_ptr < len(completed_intervals) and completed_intervals[interval_ptr][1] < step:
            current_peak = completed_intervals[interval_ptr][2]
            interval_ptr += 1
        peak_cp_by_step[step] = current_peak

    return peak_cp_by_step


def _precompute_entropy_peak_cp_by_threshold(
    entropy_by_step: dict[int, float],
    threshold: float,
    min_interval_len: int = 15,
    smooth_window: int = 5,
) -> dict[int, int | None]:
    """
    通过硬阈值 (Threshold) 寻找 Safe Region，并提取该区域的 Peak Checkpoint。
    完全模拟真机在线状态机的判断逻辑。
    """
    if not entropy_by_step:
        return {}

    sorted_steps = sorted(entropy_by_step.keys())

    # Pass 1: 对原始 Entropy 进行简单的滑动窗口平滑 (防抖)
    smoothed_entropy: dict[int, float] = {}
    buffer: list[float] = []
    for step in sorted_steps:
        val = entropy_by_step[step]
        if not math.isnan(val):
            buffer.append(val)
        if len(buffer) > smooth_window:
            buffer.pop(0)

        if buffer:
            smoothed_entropy[step] = sum(buffer) / len(buffer)
        else:
            smoothed_entropy[step] = float("nan")

    # Pass 2: 基于平滑后的阈值划分 Safe Region 并寻找 Peak
    completed_intervals: list[tuple[int, int, int]] = []  # (start, end, peak_step)
    in_free_run: list[int] = []

    for step in sorted_steps:
        val = smoothed_entropy[step]
        if not math.isnan(val) and val > threshold:
            in_free_run.append(step)
        else:
            if in_free_run:
                if len(in_free_run) >= min_interval_len:
                    # 找 Peak 时仍使用原始 Raw Entropy 取最真实极值点。
                    peak_step = max(in_free_run, key=lambda s: entropy_by_step[s])
                    completed_intervals.append((in_free_run[0], in_free_run[-1], peak_step))
                in_free_run = []
    # 故意不添加末尾未闭合区间，因为它还没结束。

    # Pass 3: 为每个 Step 映射最新已闭合区间的 Peak 作为 Checkpoint
    peak_cp_by_step: dict[int, int | None] = {}
    interval_ptr = 0
    current_peak: int | None = None

    for step in sorted_steps:
        while interval_ptr < len(completed_intervals) and completed_intervals[interval_ptr][1] < step:
            current_peak = completed_intervals[interval_ptr][2]
            interval_ptr += 1
        peak_cp_by_step[step] = current_peak

    return peak_cp_by_step


def _compute_entropy_labels_by_step(
    entropy_by_step: dict[int, float],
    episodes: list[dict],
    pipeline: str = "aloha",
) -> dict[int, int]:
    """Compute per-step HDBSCAN cluster labels from entropy values.

    Uses the same clustering pipeline as ``compute_action_entropy_safe_threshold``
    (either ``cluster_entropy_hdbscan_aloha`` or ``cluster_entropy_hdbscan_robobase``).

    Returns:
        dict mapping global_step → label where:
          - 0: precision region (low entropy, contact / manipulation)
          - 1: free region (high entropy / casual movement)
        Steps that are noise or unlabeled (-1 before abs) are omitted.
    """
    try:
        from tools.failure.metrics_compute import (
            cluster_entropy_hdbscan_aloha,
            cluster_entropy_hdbscan_robobase,
        )
    except ModuleNotFoundError:
        from metrics_compute import (
            cluster_entropy_hdbscan_aloha,
            cluster_entropy_hdbscan_robobase,
        )

    cluster_fn = cluster_entropy_hdbscan_aloha if pipeline == "aloha" else cluster_entropy_hdbscan_robobase
    label_by_step: dict[int, int] = {}

    for ep_meta in episodes:
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
        global_steps = list(range(from_idx, to_idx))
        episode_entropy = np.array([entropy_by_step.get(s, float("nan")) for s in global_steps], dtype=float)

        valid_mask = np.isfinite(episode_entropy)
        valid_steps = [global_steps[k] for k in range(len(global_steps)) if valid_mask[k]]
        valid_entropy = episode_entropy[valid_mask]

        if len(valid_steps) < 5:
            continue

        labels = cluster_fn(valid_entropy)
        for step, label in zip(valid_steps, labels, strict=False):
            label_by_step[step] = int(label)

    return label_by_step


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _concatenate_three_camera_images(images_dict: dict, camera_names: list, item: dict) -> np.ndarray | None:
    """Concatenate left, middle, right camera images horizontally."""
    # Priority order: left, middle, right
    priority_names = ["left", "middle", "right"]

    arrays = []
    for name in priority_names:
        # Find matching camera
        matching_cameras = [cam for cam in camera_names if name in cam.lower()]
        if not matching_cameras:
            continue

        cam_key = matching_cameras[0]
        img_key = f"observation.images.{cam_key}"

        if img_key not in item:
            continue

        img_data = item[img_key]
        arr = img_data.numpy() if hasattr(img_data, "numpy") else img_data

        # Handle channel-first format
        if arr.ndim == 3 and arr.shape[0] <= 4:
            arr = np.transpose(arr, (1, 2, 0))

        # Ensure 3 channels (RGB)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]  # Drop alpha channel if present

        arrays.append(arr)

    # Concatenate horizontally if we have at least 2 images
    if len(arrays) >= 2:
        concatenated = np.concatenate(arrays, axis=1)
        return concatenated
    elif len(arrays) == 1:
        return arrays[0]

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


def _log_checkpoint_windows(image_cache: dict, recent_checkpoints_by_step: dict, step: int):
    cps = recent_checkpoints_by_step.get(step, [])
    cps = cps[-5:]
    padded_cps = [None] * (5 - len(cps)) + cps
    for cp_idx, cp_step in enumerate(padded_cps):
        if cp_step is not None and cp_step in image_cache:
            rr.log(f"checkpoints/cp_{cp_idx}", image_cache[cp_step])
        else:
            rr.log(f"checkpoints/cp_{cp_idx}", rr.Clear(recursive=False))


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
    num_episode=None,
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
    if num_episode is None:
        episodes_to_visualize = total_episodes
    else:
        if int(num_episode) <= 0:
            raise ValueError("ERROR: num_episode must be a positive integer.")
        episodes_to_visualize = min(int(num_episode), total_episodes)

    print(f"Visualizing {episodes_to_visualize}/{total_episodes} episode(s).")
    dataset_root = Path(dataset.root)
    has_vlm_dir = (dataset_root / "vlm").exists()
    use_vlm_panels = False
    if has_vlm_dir:
        vlm_records = _load_vlm_message_records(dataset_root)
        if vlm_records:
            use_vlm_panels = True
            print(f"[INFO] Loaded {len(vlm_records)} VLM debug record(s) from dataset directory.")
        else:
            print(
                "[WARN] No usable VLM debug records found under <dataset_root>/vlm/debug_records. "
                "Falling back to checkpoint image windows computed from failure_metrics.jsonl."
            )
    else:
        vlm_records = []
        print(
            "[INFO] No <dataset_root>/vlm folder found. "
            "Falling back to checkpoint image windows computed from failure_metrics.jsonl."
        )

    failure_metrics = load_failure_metrics_jsonl(dataset.root)

    smoothed_td_by_step = {}
    previous_checkpoint_by_step = {}
    checkpoint_flag_by_step = {}
    detect_failure_by_step = {}

    td_failed_steps = []
    td_failed_step_set = set()
    vlm_trigger_step_set = set()

    failure_handling_cfg = load_failure_handling_json(dataset.root, required=True)
    failure_cfg = load_failure_config(dataset.root, required=True)
    checkpoint_metric_source = (
        str(failure_handling_cfg.get("checkpoint_metric_source", "temporal_disagreement")).strip().lower()
    )
    checkpoint_marker_origin = _checkpoint_marker_origin_for_source(checkpoint_metric_source)
    checkpoint_curve_labels = _checkpoint_curve_label_by_origin(checkpoint_marker_origin)
    print(f"Checkpoint metric source: {checkpoint_metric_source}")
    print(f"Checkpoint marker chart: {checkpoint_marker_origin}")

    if (
        "metrics" not in failure_handling_cfg
        or "temporal_disagreement" not in failure_handling_cfg["metrics"]
    ):
        raise ValueError(
            "ERROR: failure_handling.json must contain 'metrics' -> 'temporal_disagreement' section"
        )

    metrics_cfg = failure_handling_cfg["metrics"]
    td_config = metrics_cfg["temporal_disagreement"]

    td_cp_threshold = td_config.get("cp_threshold")
    if td_cp_threshold is None:
        raise ValueError(
            "ERROR: 'cp_threshold' not found in failure_handling.json temporal_disagreement config"
        )

    td_cp_threshold = float(td_cp_threshold)
    print(f"Loaded cp_threshold from failure_handling.json: {td_cp_threshold:.6f}")

    # Load action entropy safe threshold (SAFE = entropy > threshold → free space)
    ae_cfg_json = metrics_cfg.get("action_entropy", {})
    ae_safe_threshold_raw = ae_cfg_json.get("safe_threshold")
    if ae_safe_threshold_raw is not None:
        ae_safe_threshold: float | None = float(ae_safe_threshold_raw)
        print(f"Loaded action_entropy.safe_threshold from failure_handling.json: {ae_safe_threshold:.6f}")
        print(f"  SAFE   state: entropy > {ae_safe_threshold:.4f}  (high entropy → free / casual movement)")
        print(f"  PRECISE state: entropy ≤ {ae_safe_threshold:.4f}  (low entropy → contact / precision)")
    else:
        ae_safe_threshold = None
        print(
            "[WARN] action_entropy.safe_threshold not in failure_handling.json; SAFE region markers disabled."
        )

    safety_margin = td_config.get("safety_margin", DEFAULT_SAFETY_MARGIN)

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

        for step in sorted(failure_metrics.keys()):
            if detect_failure_by_step.get(step, False):
                td_failed_steps.append(step)
                td_failed_step_set.add(step)
            if bool(failure_metrics[step].get("vlm_request", False)):
                vlm_trigger_step_set.add(step)

    camera_names = [key.replace("observation.images.", "") for key in dataset.meta.camera_keys]
    entropy_npz_path = dataset_root / "meta" / ACTION_ENTROPY_NPZ_FILENAME
    entropy_model_path = _resolve_pretrained_path_from_dataset_root(dataset_root)
    action_entropy_policy = None
    action_entropy_preprocessor = None
    entropy_jsonl_by_step: dict[int, float] = {}
    entropy_jsonl_max_diff_by_step: dict[int, float] = {}
    entropy_offline_by_step: dict[int, float] = {}
    entropy_offline_max_diff_by_step: dict[int, float] = {}
    stream_indices: list[int] = []
    for ep_meta in dataset.meta.episodes:
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
        stream_indices.extend(range(from_idx, to_idx, stride))

    entropy_jsonl_by_step, entropy_jsonl_max_diff_by_step = _load_action_entropy_from_failure_metrics(
        failure_metrics
    )
    if entropy_jsonl_by_step:
        missing_steps = [idx for idx in stream_indices if idx not in entropy_jsonl_by_step]
        if missing_steps:
            print(
                f"[WARN] JSONL action_entropy has {len(entropy_jsonl_by_step)} step(s), "
                f"missing {len(missing_steps)} streamed step(s)."
            )
        else:
            print("[INFO] Loaded complete action entropy series from failure_metrics.jsonl.")
    else:
        print("[WARN] No action_entropy found in failure_metrics.jsonl.")

    loaded = _load_action_entropy_npz(entropy_npz_path)
    if loaded is not None:
        entropy_offline_by_step, entropy_offline_max_diff_by_step = loaded
        missing_steps = [idx for idx in stream_indices if idx not in entropy_offline_by_step]
        if missing_steps:
            print(
                f"[WARN] Cached offline action entropy missing {len(missing_steps)} streamed step(s); "
                "will recompute."
            )
            entropy_offline_by_step = {}
            entropy_offline_max_diff_by_step = {}
        else:
            print(f"[INFO] Loaded cached offline action entropy from: {entropy_npz_path}")

    if not entropy_offline_by_step and entropy_model_path is not None:
        print(
            f"[INFO] Offline action entropy cache missing required series. "
            f"Will precompute and save to: {entropy_npz_path}"
        )
        print(f"[INFO] Loading ACT policy for offline action entropy from: {entropy_model_path}")
        action_entropy_policy = ACTPolicy.from_pretrained(str(entropy_model_path))
        action_entropy_policy.eval()
        action_entropy_preprocessor, _ = make_pre_post_processors(
            action_entropy_policy.config,
            pretrained_path=str(entropy_model_path),
        )
        print(
            f"[INFO] Offline action entropy enabled: num_samples=1, "
            f"chunk_size={action_entropy_policy.config.chunk_size}"
        )
        print(
            f"[INFO] Precomputing offline action entropy for {len(stream_indices)} streamed steps "
            "with num_samples=1..."
        )
        entropy_offline_by_step, entropy_offline_max_diff_by_step = precompute_action_entropy_in_memory(
            dataset=dataset,
            policy=action_entropy_policy,
            preprocessor=action_entropy_preprocessor,
            indices=stream_indices,
            batch_size=4,
        )
        _save_action_entropy_npz(
            entropy_npz_path,
            entropy_by_step=entropy_offline_by_step,
            max_diff_by_step=entropy_offline_max_diff_by_step,
        )
        print(f"[INFO] Saved offline action entropy cache to: {entropy_npz_path}")

    action_entropy_enabled = bool(entropy_jsonl_by_step or entropy_offline_by_step)

    # Compute per-step HDBSCAN cluster labels (0=precision / 1=free) for both entropy sources.
    entropy_labels_jsonl: dict[int, int] = {}
    entropy_labels_offline: dict[int, int] = {}
    if action_entropy_enabled and dataset.meta.episodes:
        if entropy_jsonl_by_step:
            print("[INFO] Computing HDBSCAN cluster labels for JSONL action entropy...")
            entropy_labels_jsonl = _compute_entropy_labels_by_step(
                entropy_jsonl_by_step, dataset.meta.episodes
            )
            _prec = sum(1 for v in entropy_labels_jsonl.values() if v == 0)
            _free = sum(1 for v in entropy_labels_jsonl.values() if v >= 1)
            print(f"[INFO] JSONL entropy labels: {_prec} precision, {_free} free steps")
        if entropy_offline_by_step:
            print("[INFO] Computing HDBSCAN cluster labels for offline action entropy...")
            entropy_labels_offline = _compute_entropy_labels_by_step(
                entropy_offline_by_step, dataset.meta.episodes
            )
            _prec = sum(1 for v in entropy_labels_offline.values() if v == 0)
            _free = sum(1 for v in entropy_labels_offline.values() if v >= 1)
            print(f"[INFO] Offline entropy labels: {_prec} precision, {_free} free steps")

    if action_entropy_enabled:
        pass

    # Pre-compute entropy-based peak checkpoints independently for JSONL and offline.
    entropy_peak_cp_jsonl_by_step: dict[int, int | None] = {}
    entropy_peak_steps_jsonl: set[int] = set()
    entropy_peak_cp_offline_by_step: dict[int, int | None] = {}
    entropy_peak_steps_offline: set[int] = set()

    if action_entropy_enabled and ae_safe_threshold is not None:
        # 处理 JSONL 来源的 Entropy
        if entropy_jsonl_by_step:
            entropy_peak_cp_jsonl_by_step = _precompute_entropy_peak_cp_by_threshold(
                entropy_by_step=entropy_jsonl_by_step,
                threshold=ae_safe_threshold,
                min_interval_len=15,
                smooth_window=5,
            )
            entropy_peak_steps_jsonl = {
                int(v) for v in entropy_peak_cp_jsonl_by_step.values() if v is not None
            }
            n_peaks_jsonl = len(entropy_peak_steps_jsonl)
            print(
                "[INFO] (Threshold Mode) Pre-computed JSONL peak checkpoints: "
                f"found {n_peaks_jsonl} valid safe regions."
            )

        # 处理 Offline 来源的 Entropy
        if entropy_offline_by_step:
            entropy_peak_cp_offline_by_step = _precompute_entropy_peak_cp_by_threshold(
                entropy_by_step=entropy_offline_by_step,
                threshold=ae_safe_threshold,
                min_interval_len=15,
                smooth_window=5,
            )
            entropy_peak_steps_offline = {
                int(v) for v in entropy_peak_cp_offline_by_step.values() if v is not None
            }
            n_peaks_offline = len(entropy_peak_steps_offline)
            print(
                "[INFO] (Threshold Mode) Pre-computed Offline peak checkpoints: "
                f"found {n_peaks_offline} valid safe regions."
            )
    else:
        print(
            "[WARN] ae_safe_threshold is None or entropy disabled. Cannot compute threshold-based checkpoints."
        )

    if action_entropy_enabled and not (entropy_peak_steps_jsonl or entropy_peak_steps_offline):
        print(
            "[WARN] No completed free-labeled peaks found for marker rendering. "
            "Consider lowering min_interval_len or checking entropy labels."
        )

    if action_entropy_enabled:
        if entropy_offline_max_diff_by_step and max(entropy_offline_max_diff_by_step.values()) < 1e-8:
            print(
                "[WARN] Offline action entropy sampling seems deterministic for num_samples=1 "
                "(all max diff < 1e-8)."
            )
        print("[INFO] Action entropy comparison data ready (JSONL + offline).")
    elif entropy_model_path is None:
        print(
            "[WARN] Could not resolve pretrained_path and JSONL has no action_entropy; "
            "action entropy visualization is disabled."
        )

    if failure_metrics:
        camera_views = [rrb.Spatial2DView(origin=f"cameras/{cam}") for cam in camera_names]
        id_overlay_view = rrb.TextDocumentView(
            name="Episode ID",
            origin="overlay/episode_id",
        )

        if use_vlm_panels:
            text_views = [
                rrb.TextDocumentView(
                    name=f"Window 3.{idx} - Request Pair Text",
                    origin=f"vlm_message/pairs/{idx}/text",
                )
                for idx in range(MAX_VLM_PAIR_SLOTS)
            ]
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

            if action_entropy_enabled and (entropy_peak_cp_jsonl_by_step or entropy_peak_cp_offline_by_step):
                middle_right_panel = rrb.Vertical(
                    rrb.Spatial2DView(
                        name="JSONL Peak Checkpoint",
                        origin="checkpoints/entropy_cp_jsonl",
                    ),
                    rrb.Spatial2DView(
                        name="Offline Peak Checkpoint",
                        origin="checkpoints/entropy_cp_offline",
                    ),
                    row_shares=[1, 1],
                )

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.TimeSeriesView(
                            name="Temporal Disagreement", origin="metrics/temporal_disagreement"
                        ),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/temporal_disagreement_smoothed"],
                            origin="metrics/temporal_disagreement_smoothed",
                        ),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/following_error"],
                            origin="metrics/following_error",
                        ),
                        rrb.TimeSeriesView(
                            name="Previous Checkpoint Step", origin="metrics/previous_checkpoint_step"
                        ),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/attention_entropy"],
                            origin="metrics/attention_entropy",
                        ),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/endpoint_shift"],
                            origin="metrics/endpoint_shift",
                        ),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/action_jerk"],
                            origin="metrics/action_jerk",
                        ),
                        rrb.TimeSeriesView(name="Checkpoint Flag", origin="metrics/checkpoint_flag"),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/action_entropy_compare"],
                            origin="metrics/action_entropy_compare",
                        ),
                        rrb.TimeSeriesView(
                            name=checkpoint_curve_labels["metrics/action_entropy_max_diff_compare"],
                            origin="metrics/action_entropy_max_diff_compare",
                        ),
                    ),
                    middle_right_panel,
                    column_shares=[1, 1],
                ),
                rrb.Vertical(*camera_views, id_overlay_view, row_shares=[6] * len(camera_views) + [1]),
                column_shares=[4, 1],
            ),
            collapse_panels=True,
        )
    elif action_entropy_enabled:
        camera_views = [rrb.Spatial2DView(origin=f"cameras/{cam}") for cam in camera_names]
        id_overlay_view = rrb.TextDocumentView(
            name="Episode ID",
            origin="overlay/episode_id",
        )
        entropy_cp_jsonl_view = rrb.Spatial2DView(
            name="JSONL Peak Checkpoint",
            origin="checkpoints/entropy_cp_jsonl",
        )
        entropy_cp_offline_view = rrb.Spatial2DView(
            name="Offline Peak Checkpoint",
            origin="checkpoints/entropy_cp_offline",
        )
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TimeSeriesView(
                        name=checkpoint_curve_labels["metrics/action_entropy_compare"],
                        origin="metrics/action_entropy_compare",
                    ),
                    rrb.TimeSeriesView(
                        name=checkpoint_curve_labels["metrics/action_entropy_max_diff_compare"],
                        origin="metrics/action_entropy_max_diff_compare",
                    ),
                    entropy_cp_jsonl_view,
                    entropy_cp_offline_view,
                    row_shares=[3, 3, 2, 2],
                ),
                rrb.Vertical(*camera_views, id_overlay_view, row_shares=[6] * len(camera_views) + [1]),
                column_shares=[1, 2],
            ),
            collapse_panels=True,
        )
    else:
        blueprint = None

    rr.init("LeRobot Dataset Visualizer", spawn=True)
    if blueprint:
        rr.send_blueprint(blueprint)

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
        # Green markers for checkpoint steps in temporal_disagreement_smoothed
        rr.log(
            f"{checkpoint_marker_origin}/checkpoint_markers",
            rr.SeriesPoints(colors=[0, 255, 0], markers="circle", marker_sizes=5.0),
            static=True,
        )

    if ae_safe_threshold is not None:
        # Red dots for SAFE region (entropy > threshold → robot in free/casual space)
        # Same style as failed_markers on temporal_disagreement_smoothed
        rr.log(
            "metrics/action_entropy_compare/jsonl/safe_region",
            rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
            static=True,
        )
        rr.log(
            "metrics/action_entropy_compare/offline/safe_region",
            rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
            static=True,
        )

    if action_entropy_enabled:
        rr.log(
            "metrics/action_entropy_compare/jsonl/precision",
            rr.SeriesPoints(colors=[100, 150, 255], markers="circle", marker_sizes=3.0),
            static=True,
        )
        rr.log(
            "metrics/action_entropy_compare/jsonl/free",
            rr.SeriesPoints(colors=[255, 140, 0], markers="diamond", marker_sizes=3.0),
            static=True,
        )
        rr.log(
            "metrics/action_entropy_compare/offline/precision",
            rr.SeriesPoints(colors=[0, 200, 100], markers="circle", marker_sizes=3.0),
            static=True,
        )
        rr.log(
            "metrics/action_entropy_compare/offline/free",
            rr.SeriesPoints(colors=[255, 80, 80], markers="diamond", marker_sizes=3.0),
            static=True,
        )
        rr.log(
            "metrics/action_entropy_compare/jsonl/peak_cp_markers",
            rr.SeriesPoints(colors=[0, 230, 60], markers="circle", marker_sizes=8.0),
            static=True,
        )
        rr.log(
            "metrics/action_entropy_compare/offline/peak_cp_markers",
            rr.SeriesPoints(colors=[0, 230, 60], markers="circle", marker_sizes=8.0),
            static=True,
        )

    global_step = 0
    prev_attention_entropy = None
    prev_attention_step = None

    image_cache = {}

    for episode_idx in range(episodes_to_visualize):
        print(f"Streaming Episode {episode_idx}/{episodes_to_visualize}...", end="\r")
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

        episode_indices = list(range(from_idx, to_idx, stride))

        def _collate_fn(batch):
            return batch[0]

        # Default checkpoint for this episode: first frame.
        # Superseded once a qualified free-interval peak is found.
        episode_start_cp: int = from_idx

        loader = DataLoader(
            Subset(dataset, episode_indices),
            batch_size=1,
            num_workers=4,
            prefetch_factor=2,
            collate_fn=_collate_fn,
            shuffle=False,
        )

        for i, item in zip(episode_indices, loader, strict=True):
            rr.set_time_sequence("global_step", global_step)
            global_step += 1
            rr.log("overlay/episode_id", rr.TextDocument(f"{episode_idx}"), static=False)

            if item is None:
                continue

            for img_key in [k for k in item if "image" in k]:
                img_data = item[img_key]
                clean_key = img_key.replace("observation.images.", "")

                rr_img_obj = None

                if isinstance(img_data, dict) and "bytes" in img_data:
                    try:
                        rr_img_obj = rr.ImageEncoded(contents=img_data["bytes"])
                    except AttributeError:
                        if hasattr(rr, "EncodedImage"):
                            rr_img_obj = rr.EncodedImage(contents=img_data["bytes"])
                        else:
                            rr_img_obj = rr.Image(Image.open(io.BytesIO(img_data["bytes"])))
                else:
                    arr = img_data.numpy() if hasattr(img_data, "numpy") else img_data
                    if arr.ndim == 3 and arr.shape[0] <= 4:
                        arr = np.transpose(arr, (1, 2, 0))
                    rr_img_obj = rr.Image(arr)

                rr.log(f"cameras/{clean_key}", rr_img_obj)

            # Create concatenated image for checkpoint display (left, middle, right)
            concatenated_arr = _concatenate_three_camera_images({}, camera_names, item)
            if concatenated_arr is not None:
                image_cache[i] = rr.Image(concatenated_arr)

            # Entropy-based peak checkpoints: previous completed free-labeled interval peaks
            # for JSONL and offline sources, each falling back to the episode start frame.
            entropy_peak_cp_jsonl: int | None = entropy_peak_cp_jsonl_by_step.get(i)
            entropy_peak_cp_offline: int | None = entropy_peak_cp_offline_by_step.get(i)
            effective_cp_jsonl: int = (
                entropy_peak_cp_jsonl if entropy_peak_cp_jsonl is not None else episode_start_cp
            )
            effective_cp_offline: int = (
                entropy_peak_cp_offline if entropy_peak_cp_offline is not None else episode_start_cp
            )

            protected_checkpoint_steps: set[int] = {
                episode_start_cp,
                effective_cp_jsonl,
                effective_cp_offline,
            }
            if entropy_peak_cp_jsonl is not None:
                protected_checkpoint_steps.add(entropy_peak_cp_jsonl)
            if entropy_peak_cp_offline is not None:
                protected_checkpoint_steps.add(entropy_peak_cp_offline)
            if failure_metrics:
                protected_checkpoint_steps |= {
                    int(cp_step) for cp_step in recent_checkpoints_by_step.get(i, [])[-5:]
                }

            old_keys = [
                k for k in list(image_cache.keys()) if k < i - 150 and k not in protected_checkpoint_steps
            ]
            for k in old_keys:
                del image_cache[k]

            if action_entropy_enabled:
                entropy_value_jsonl = entropy_jsonl_by_step.get(i)
                sample_max_diff_jsonl = entropy_jsonl_max_diff_by_step.get(i)
                entropy_value_offline = entropy_offline_by_step.get(i)
                sample_max_diff_offline = entropy_offline_max_diff_by_step.get(i)

                is_entropy_peak_cp_jsonl = i in entropy_peak_steps_jsonl
                is_entropy_peak_cp_offline = i in entropy_peak_steps_offline

                if ae_safe_threshold is not None:
                    rr.log("metrics/action_entropy_compare/threshold", rr.Scalars(ae_safe_threshold))

                if entropy_value_jsonl is not None:
                    rr.log("metrics/action_entropy_compare/jsonl", rr.Scalars(entropy_value_jsonl))
                    _lbl_jsonl = entropy_labels_jsonl.get(i, -1)
                    if _lbl_jsonl == 0:
                        rr.log(
                            "metrics/action_entropy_compare/jsonl/precision",
                            rr.Scalars(entropy_value_jsonl),
                        )
                    elif _lbl_jsonl >= 1:
                        rr.log(
                            "metrics/action_entropy_compare/jsonl/free",
                            rr.Scalars(entropy_value_jsonl),
                        )
                    if is_entropy_peak_cp_jsonl and _lbl_jsonl >= 1:
                        rr.log(
                            "metrics/action_entropy_compare/jsonl/peak_cp_markers",
                            rr.Scalars(entropy_value_jsonl),
                        )
                if entropy_value_offline is not None:
                    rr.log("metrics/action_entropy_compare/offline", rr.Scalars(entropy_value_offline))
                    _lbl_offline = entropy_labels_offline.get(i, -1)
                    if _lbl_offline == 0:
                        rr.log(
                            "metrics/action_entropy_compare/offline/precision",
                            rr.Scalars(entropy_value_offline),
                        )
                    elif _lbl_offline >= 1:
                        rr.log(
                            "metrics/action_entropy_compare/offline/free",
                            rr.Scalars(entropy_value_offline),
                        )
                    if is_entropy_peak_cp_offline and _lbl_offline >= 1:
                        rr.log(
                            "metrics/action_entropy_compare/offline/peak_cp_markers",
                            rr.Scalars(entropy_value_offline),
                        )
                if sample_max_diff_jsonl is not None:
                    rr.log(
                        "metrics/action_entropy_max_diff_compare/jsonl",
                        rr.Scalars(sample_max_diff_jsonl),
                    )
                if sample_max_diff_offline is not None:
                    rr.log(
                        "metrics/action_entropy_max_diff_compare/offline",
                        rr.Scalars(sample_max_diff_offline),
                    )

                if effective_cp_jsonl in image_cache:
                    rr.log("checkpoints/entropy_cp_jsonl", image_cache[effective_cp_jsonl])
                else:
                    rr.log("checkpoints/entropy_cp_jsonl", rr.Clear(recursive=False))

                if effective_cp_offline in image_cache:
                    rr.log("checkpoints/entropy_cp_offline", image_cache[effective_cp_offline])
                else:
                    rr.log("checkpoints/entropy_cp_offline", rr.Clear(recursive=False))

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

                is_td_failed = i in td_failed_step_set
                is_checkpoint = checkpoint_flag_by_step.get(i, 0.0) > 0.5

                if is_td_failed:
                    rr.log("metrics/temporal_disagreement_smoothed/failed_markers", rr.Scalars(smooth_td))
                    rr.log(
                        "metrics/previous_checkpoint_step/failed_markers",
                        rr.Scalars(previous_checkpoint_by_step.get(i, np.nan)),
                    )

                if is_checkpoint:
                    marker_value = _checkpoint_marker_value_for_step(
                        source=checkpoint_metric_source,
                        metric_row=m,
                        smoothed_td=smooth_td,
                        step=i,
                        entropy_jsonl_by_step=entropy_jsonl_by_step,
                        entropy_jsonl_max_diff_by_step=entropy_jsonl_max_diff_by_step,
                    )
                    if marker_value is not None:
                        rr.log(
                            f"{checkpoint_marker_origin}/checkpoint_markers",
                            rr.Scalars(marker_value),
                        )

                if use_vlm_panels:
                    matched_record = _select_vlm_record_for_step(
                        vlm_records,
                        episode_idx,
                        i,
                        max_step_gap=0 if vlm_trigger_step_set else 3,
                    )
                    should_show_vlm = (i in vlm_trigger_step_set) if vlm_trigger_step_set else is_td_failed
                    if should_show_vlm:
                        _log_vlm_record_windows(matched_record)
                    else:
                        _log_vlm_record_windows(None)
                else:
                    if action_entropy_enabled and (
                        entropy_peak_cp_jsonl_by_step or entropy_peak_cp_offline_by_step
                    ):
                        # Entropy checkpoint panels are already updated above.
                        pass
                    else:
                        cps = recent_checkpoints_by_step.get(i, [])
                        if cps:
                            _log_checkpoint_windows(image_cache, recent_checkpoints_by_step, i)
                        else:
                            for cp_idx in range(5):
                                rr.log(f"checkpoints/cp_{cp_idx}", rr.Clear(recursive=False))

    print("\nDone streaming to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LeRobot dataset directly to Rerun.")
    parser.add_argument("--repo_id", type=str, help="Dataset repository ID")
    parser.add_argument("--root", type=str, default=None, help="Dataset root")
    parser.add_argument("--stride", type=int, default=7, help="Visualization stride (speed)")
    parser.add_argument(
        "--num_episode",
        type=int,
        default=None,
        help="Only visualize the first N episodes. Default: all episodes.",
    )
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
        num_episode=args.num_episode,
        save_timing_plot=args.save_timing_plot,
        timing_plot_path=args.timing_plot_path,
        timing_npz_path=args.timing_npz_path,
    )
