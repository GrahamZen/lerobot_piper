#!/usr/bin/env python
"""Compute CP threshold, update/create failure_handling.json, and export demo video.

This script:
1) Reads calibration scores from <repo_id>/failure_metrics.jsonl.
2) Computes CP threshold for temporal_disagreement:
         q_level = ceil((n + 1) * (1 - alpha)) / n
         cp_threshold = quantile(scores, min(q_level, 1.0))
3) Loads action entropy from <repo_id>/failure_metrics.jsonl when available,
   otherwise falls back to <repo_id>/meta/action_entropy.npz.
4) Loads <repo_id>/meta/record_config.json and resolves pretrained_path.
5) Creates or updates <pretrained_path>/failure_handling.json:
    - If missing, initialize from tools/failure/examples/pick_up_markers/failure_handling.json.
    - Always set metrics.temporal_disagreement.cp_threshold.
6) Exports first episode (episode_index=0) demo video to pretrained_path and updates
    demo_video_path in failure_handling.json with the absolute video path.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
from contextlib import suppress
from pathlib import Path

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute and save CP threshold from temporal_disagreement, "
            "create/update failure_handling.json, and export first-episode demo video."
        )
    )
    parser.add_argument("--repo_id", type=str, required=True, help="Repo id, e.g. eval/eval_failure_metrics")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Miscoverage level alpha. Default 0.01 (confidence 99%%).",
    )
    parser.add_argument("--fps", type=int, default=240, help="Output demo video FPS. Default: 240")
    parser.add_argument("--scale", type=float, default=0.5, help="Output demo video scale. Default: 0.5")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite demo video even if existing resolution already matches target.",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot").expanduser(),
        help="Lerobot cache root. Default: ~/.cache/huggingface/lerobot",
    )
    parser.add_argument(
        "--entropy_percentile",
        type=float,
        default=95.0,
        help="Percentile on clustered precision set for action entropy safe threshold. Default: 95",
    )
    parser.add_argument(
        "--entropy_min_cluster_size",
        type=int,
        default=50,
        help="HDBSCAN min_cluster_size for action entropy clustering. Default: 50",
    )
    parser.add_argument(
        "--entropy_min_samples",
        type=int,
        default=10,
        help="HDBSCAN min_samples for action entropy clustering. Default: 10",
    )
    parser.add_argument(
        "--trim_episode_frames",
        type=int,
        default=30,
        help="Ignore the first and last N frames of each episode when extracting thresholds. Default: 30",
    )
    return parser.parse_args()


def _load_metrics_rows(metrics_path: Path) -> list[dict]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"failure_metrics.jsonl not found: {metrics_path}")

    rows: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metrics_path} at line {line_no}") from exc

            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at line {line_no} in {metrics_path}")
            rows.append(row)

    return rows


def _trim_metrics_rows_by_episode(rows: list[dict], trim_episode_frames: int) -> list[dict]:
    if trim_episode_frames <= 0 or not rows:
        return rows

    grouped_rows: dict[int, list[dict]] = {}
    for row in rows:
        episode = row.get("episode")
        if not isinstance(episode, int):
            return rows
        grouped_rows.setdefault(int(episode), []).append(row)

    trimmed_rows: list[dict] = []
    for episode in sorted(grouped_rows):
        episode_rows = grouped_rows[episode]
        if len(episode_rows) <= 2 * trim_episode_frames:
            continue
        trimmed_rows.extend(episode_rows[trim_episode_frames:-trim_episode_frames])

    return trimmed_rows


def _extract_numeric_metric(rows: list[dict], metric_key: str, metrics_path: Path) -> np.ndarray:
    scores = []
    for line_no, row in enumerate(rows, start=1):
        value = row.get(metric_key)
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{metric_key} must be numeric, got {type(value).__name__} "
                f"at filtered row {line_no} in {metrics_path}"
            )
        if np.isfinite(value):
            scores.append(float(value))

    if not scores:
        raise ValueError(f"No valid {metric_key} values found in {metrics_path}")

    return np.array(scores, dtype=np.float64)


def load_temporal_disagreement(metrics_path: Path, trim_episode_frames: int = 30) -> np.ndarray:
    rows = _load_metrics_rows(metrics_path)
    rows = _trim_metrics_rows_by_episode(rows, trim_episode_frames)
    return _extract_numeric_metric(rows, "temporal_disagreement", metrics_path)


def load_action_entropy(metrics_path: Path, trim_episode_frames: int = 30) -> np.ndarray:
    rows = _load_metrics_rows(metrics_path)
    rows = _trim_metrics_rows_by_episode(rows, trim_episode_frames)
    return _extract_numeric_metric(rows, "action_entropy", metrics_path)


def load_action_entropy_from_npz(
    npz_path: Path,
    episode_ranges: list[tuple[int, int]] | None = None,
    trim_episode_frames: int = 30,
) -> np.ndarray:
    if not npz_path.exists():
        raise FileNotFoundError(f"action_entropy.npz not found: {npz_path}")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Failed to read npz file: {npz_path}") from exc

    key_candidates = ("entropy", "action_entropy", "entropies")
    entropy_arr = None
    for key in key_candidates:
        if key in data.files:
            entropy_arr = np.asarray(data[key], dtype=np.float64).reshape(-1)
            break

    if entropy_arr is None:
        raise KeyError(
            f"None of expected entropy keys {key_candidates} found in {npz_path}. "
            f"available keys: {list(data.files)}"
        )

    step_arr = None
    if "step" in data.files:
        step_arr = np.asarray(data["step"], dtype=np.int64).reshape(-1)
        if step_arr.shape[0] != entropy_arr.shape[0]:
            raise ValueError(
                f"step and entropy length mismatch in {npz_path}: "
                f"{step_arr.shape[0]} vs {entropy_arr.shape[0]}"
            )

    if episode_ranges and trim_episode_frames > 0 and step_arr is not None:
        keep_mask = np.zeros(step_arr.shape[0], dtype=bool)
        for from_idx, to_idx in episode_ranges:
            inner_start = int(from_idx) + trim_episode_frames
            inner_end = int(to_idx) - trim_episode_frames
            if inner_start >= inner_end:
                continue
            keep_mask |= (step_arr >= inner_start) & (step_arr < inner_end)
        entropy_arr = entropy_arr[keep_mask]

    entropy_arr = entropy_arr[np.isfinite(entropy_arr)]
    if entropy_arr.size == 0:
        raise ValueError(f"No finite entropy values found in {npz_path}")

    return entropy_arr


def load_action_entropy_with_fallback(
    metrics_path: Path,
    entropy_npz_path: Path,
    episode_ranges: list[tuple[int, int]] | None = None,
    trim_episode_frames: int = 30,
) -> tuple[np.ndarray, str]:
    try:
        arr = load_action_entropy(metrics_path, trim_episode_frames=trim_episode_frames)
        return arr, "failure_metrics.jsonl"
    except (FileNotFoundError, ValueError, KeyError) as jsonl_exc:
        print(
            "[WARN] action_entropy not available in failure_metrics.jsonl, "
            f"fallback to npz: {entropy_npz_path}"
        )
        try:
            arr = load_action_entropy_from_npz(
                entropy_npz_path,
                episode_ranges=episode_ranges,
                trim_episode_frames=trim_episode_frames,
            )
            print(
                "[INFO] Loaded action_entropy from npz fallback. "
                f"source={entropy_npz_path}, samples={arr.size}"
            )
            return arr, "meta/action_entropy.npz"
        except Exception as npz_exc:
            raise RuntimeError(
                "Failed to load action entropy from both failure_metrics.jsonl and npz fallback. "
                f"jsonl_error={jsonl_exc}; npz_error={npz_exc}"
            ) from npz_exc


def compute_cp_threshold(calibration_scores: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(calibration_scores)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(calibration_scores, q_level))
    return cp_threshold, q_level, n


def compute_action_entropy_safe_threshold(
    entropies: np.ndarray,
    percentile: float,
    min_cluster_size: int,
    min_samples: int,
) -> tuple[float, int, int]:
    try:
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required to compute action entropy safe threshold. "
            "Please install scikit-learn in the current environment."
        ) from exc

    hdbscan = sklearn_cluster.HDBSCAN
    standard_scaler = sklearn_preprocessing.StandardScaler

    if entropies.ndim != 1:
        raise ValueError(f"entropies must be a 1D array, got shape {entropies.shape}")
    if entropies.size == 0:
        raise ValueError("entropies is empty")
    if not 0 < percentile <= 100:
        raise ValueError(f"entropy_percentile must be in (0, 100], got {percentile}")
    if min_cluster_size < 2:
        raise ValueError(f"entropy_min_cluster_size must be >= 2, got {min_cluster_size}")
    if min_samples < 1:
        raise ValueError(f"entropy_min_samples must be >= 1, got {min_samples}")

    all_entropies = entropies.reshape(-1, 1)
    scaler = standard_scaler()
    normalized_entropies = scaler.fit_transform(all_entropies)

    clusterer = hdbscan(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(normalized_entropies)

    precision_set_entropies = all_entropies[labels >= 0]
    if precision_set_entropies.size == 0:
        # Fallback when all points are considered noise: use global percentile.
        precision_set_entropies = all_entropies

    safe_threshold = float(np.percentile(precision_set_entropies, percentile))
    return safe_threshold, int(precision_set_entropies.size), int(all_entropies.size)


def read_record_config(record_config_path: Path) -> dict:
    if not record_config_path.exists():
        raise FileNotFoundError(f"record_config.json not found: {record_config_path}")
    with record_config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_pretrained_path(record_config: dict) -> Path:
    policy_cfg = record_config.get("policy", {})
    pretrained_path = policy_cfg.get("pretrained_path") or record_config.get("pretrained_path")
    if not pretrained_path:
        raise KeyError("pretrained_path not found in record_config.json")
    return Path(pretrained_path).expanduser()


def load_or_create_failure_handling_config(failure_handling_path: Path) -> dict:
    return {
        "demo_video_path": "demo.mp4",
        "enable_logging": True,
        "enable_failure_handling": False,
        "flush_metrics_every_step": False,
        "checkpoint_queue_size": 5,
        "metrics": {
            "temporal_disagreement": {
                "enabled": True,
                "failure_threshold": 0.3,
                "cp_threshold": 0.23238854094630587,
                "window_size": 31,
                "eval_delay": 15,
                "smoothing_sigma": 5.0,
                "valley_lookback": 8,
                "valley_lookahead": 8,
                "valley_prominence": 0.0,
                "rho": 1.0,
            },
            "following_error": {
                "enabled": True,
                "threshold": 0.05,
            },
            "attention_entropy": {
                "enabled": True,
            },
            "mahalanobis_distance": {
                "enabled": True,
            },
            "endpoint_shift": {
                "enabled": True,
            },
            "action_jerk": {
                "enabled": True,
            },
            "action_entropy": {
                "enabled": True,
                "safe_threshold": -1.0,
                "min_bandwidth": 1e-5,
                "min_density": 1e-35,
                "min_overlap_samples": 3,
            },
        },
    }


def _set_cp_threshold(config: dict, cp_threshold: float) -> None:
    metrics = config.setdefault("metrics", {})
    td_cfg = metrics.setdefault("temporal_disagreement", {})
    td_cfg["cp_threshold"] = float(cp_threshold)


def _set_action_entropy_safe_threshold(config: dict, safe_threshold: float) -> None:
    metrics = config.setdefault("metrics", {})
    ae_cfg = metrics.setdefault("action_entropy", {})
    ae_cfg["safe_threshold"] = float(safe_threshold)


def _save_failure_handling_config(failure_handling_path: Path, config: dict) -> None:
    failure_handling_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_handling_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _to_uint8_image_array(arr) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        pass
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] > 4:
            arr = arr[..., :3]
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

    if arr.dtype == np.uint8:
        return arr

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        max_val = float(np.max(arr)) if arr.size else 0.0
        min_val = float(np.min(arr)) if arr.size else 0.0
        if max_val <= 1.0 and min_val >= 0.0:
            arr = arr * 255.0
        return np.clip(arr, 0.0, 255.0).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.uint8)

    return np.clip(arr.astype(np.float32), 0.0, 255.0).astype(np.uint8)


def _get_image(item: dict, camera_prefix: str = "observation.images.") -> dict[str, Image.Image]:
    cameras = ["left", "middle", "right"]
    images: dict[str, Image.Image] = {}
    for cam in cameras:
        key = f"{camera_prefix}{cam}"
        if key in item:
            img_data = item[key]
            if isinstance(img_data, dict) and "bytes" in img_data:
                images[cam] = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
            else:
                array = img_data.numpy() if hasattr(img_data, "numpy") else img_data
                images[cam] = Image.fromarray(_to_uint8_image_array(array)).convert("RGB")
        else:
            images[cam] = Image.new("RGB", (100, 100), color=(0, 0, 0))
    return images


def _get_episode_bounds(dataset: LeRobotDataset, episode_index: int) -> tuple[int, int]:
    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    ep_meta = dataset.meta.episodes[episode_index]
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
    return from_idx, to_idx


def _build_triplet_frame_image(item: dict, scale: float) -> Image.Image:
    imgs = _get_image(item)
    frame = np.concatenate(
        [
            np.array(imgs["left"]),
            np.array(imgs["middle"]),
            np.array(imgs["right"]),
        ],
        axis=1,
    )
    frame_img = Image.fromarray(frame).convert("RGB")
    if scale != 1.0:
        new_w = max(1, int(round(frame_img.width * scale)))
        new_h = max(1, int(round(frame_img.height * scale)))
        frame_img = frame_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return frame_img


def _read_video_resolution(video_path: Path) -> tuple[int, int] | None:
    try:
        reader = imageio.get_reader(video_path)
    except Exception:
        return None

    try:
        meta = reader.get_meta_data()
        size = meta.get("size") if isinstance(meta, dict) else None
        if isinstance(size, tuple) and len(size) == 2:
            return int(size[0]), int(size[1])

        frame0 = reader.get_data(0)
        if frame0.ndim >= 2:
            return int(frame0.shape[1]), int(frame0.shape[0])
        return None
    except Exception:
        return None
    finally:
        with suppress(Exception):
            reader.close()


def export_first_episode_video(
    repo_id: str,
    cache_root: Path,
    output_dir: Path,
    fps: int,
    scale: float,
    force: bool,
) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    dataset = LeRobotDataset(repo_id, root=cache_root / repo_id)
    from_idx, to_idx = _get_episode_bounds(dataset, episode_index=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    scale_str = str(scale).replace(".", "p")
    video_path = output_dir / f"demo_episode0_fps{fps}_scale{scale_str}.mp4"

    if from_idx >= to_idx:
        raise ValueError(f"Episode 0 has no frames: from_idx={from_idx}, to_idx={to_idx}")

    first_frame_img = _build_triplet_frame_image(dataset[from_idx], scale)
    target_resolution = first_frame_img.size

    if video_path.exists() and not force:
        existing_resolution = _read_video_resolution(video_path)
        if existing_resolution == target_resolution:
            return video_path.resolve()

    writer = imageio.get_writer(video_path, fps=fps)
    try:
        total_frames = to_idx - from_idx
        with tqdm(total=total_frames, desc="Exporting video", unit="frame") as pbar:
            for step_idx in range(from_idx, to_idx):
                if step_idx == from_idx:
                    frame_img = first_frame_img
                else:
                    frame_img = _build_triplet_frame_image(dataset[step_idx], scale)
                writer.append_data(np.array(frame_img, dtype=np.uint8))
                pbar.update(1)
    finally:
        writer.close()

    return video_path.resolve()


def main() -> None:
    args = parse_args()

    repo_dir = args.cache_root / args.repo_id
    metrics_path = repo_dir / "failure_metrics.jsonl"
    entropy_npz_path = repo_dir / "meta" / "action_entropy.npz"
    record_config_path = repo_dir / "meta" / "record_config.json"
    dataset = LeRobotDataset(args.repo_id, root=repo_dir)
    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)
    episode_ranges = [
        _get_episode_bounds(dataset, episode_index=i) for i in range(len(dataset.meta.episodes))
    ]

    calibration_scores = load_temporal_disagreement(
        metrics_path, trim_episode_frames=args.trim_episode_frames
    )
    cp_threshold, q_level, n = compute_cp_threshold(calibration_scores, args.alpha)
    action_entropies, action_entropy_source = load_action_entropy_with_fallback(
        metrics_path,
        entropy_npz_path,
        episode_ranges=episode_ranges,
        trim_episode_frames=args.trim_episode_frames,
    )
    ae_safe_threshold, p_count, ae_total = compute_action_entropy_safe_threshold(
        action_entropies,
        percentile=args.entropy_percentile,
        min_cluster_size=args.entropy_min_cluster_size,
        min_samples=args.entropy_min_samples,
    )

    print(f"repo_id: {args.repo_id}")
    print(f"samples (n): {n}")
    print(f"alpha: {args.alpha}")
    print(f"trim_episode_frames: {args.trim_episode_frames}")
    print(f"q_level: {q_level}")
    print(f"cp_threshold: {cp_threshold}")
    print(f"action_entropy_source: {action_entropy_source}")
    print(f"action_entropy_samples: {ae_total}")
    print(f"action_entropy_precision_set_samples: {p_count}")
    print(f"action_entropy_safe_threshold_p{args.entropy_percentile}: {ae_safe_threshold}")
    if not record_config_path.exists():
        print(
            f"❌ record_config.json not found at {record_config_path}. Cannot write CP threshold without it."
        )
        return
    record_config = read_record_config(record_config_path)
    pretrained_path = resolve_pretrained_path(record_config)
    failure_handling_path = pretrained_path / "failure_handling.json"
    config = load_or_create_failure_handling_config(failure_handling_path)
    _set_cp_threshold(config, cp_threshold)
    _set_action_entropy_safe_threshold(config, ae_safe_threshold)

    demo_video_path = export_first_episode_video(
        repo_id=args.repo_id,
        cache_root=args.cache_root,
        output_dir=pretrained_path,
        fps=args.fps,
        scale=args.scale,
        force=args.force,
    )
    config["demo_video_path"] = str(demo_video_path)

    _save_failure_handling_config(failure_handling_path, config)
    print(f"written_to: {failure_handling_path}")
    print(f"demo_video_path: {demo_video_path}")


if __name__ == "__main__":
    main()
