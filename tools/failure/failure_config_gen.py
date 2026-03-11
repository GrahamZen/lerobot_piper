#!/usr/bin/env python
"""Compute CP threshold, update/create failure_handling.json, and export demo video.

This script:
1) Reads calibration scores from <repo_id>/failure_metrics.jsonl.
2) Computes CP threshold for temporal_disagreement:
         q_level = ceil((n + 1) * (1 - alpha)) / n
         cp_threshold = quantile(scores, min(q_level, 1.0))
3) Loads <repo_id>/meta/record_config.json and resolves pretrained_path.
4) Creates or updates <pretrained_path>/failure_handling.json:
    - If missing, initialize from tools/failure/examples/pick_up_markers/failure_handling.json.
    - Always set metrics.temporal_disagreement.cp_threshold.
5) Exports first episode (episode_index=0) demo video to pretrained_path and updates
    demo_video_path in failure_handling.json with the absolute video path.
"""

from __future__ import annotations

import argparse
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
    return parser.parse_args()


def load_temporal_disagreement(metrics_path: Path) -> np.ndarray:
    if not metrics_path.exists():
        raise FileNotFoundError(f"failure_metrics.jsonl not found: {metrics_path}")

    scores = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {metrics_path} at line {line_no}") from exc

            value = row.get("temporal_disagreement")
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"temporal_disagreement must be numeric, got {type(value).__name__} "
                    f"at line {line_no} in {metrics_path}"
                )
            scores.append(float(value))

    if not scores:
        raise ValueError(f"No valid temporal_disagreement values found in {metrics_path}")

    return np.array(scores, dtype=np.float64)


def compute_cp_threshold(calibration_scores: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(calibration_scores)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(calibration_scores, q_level))
    return cp_threshold, q_level, n


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
    if failure_handling_path.exists():
        with failure_handling_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    # Default configuration
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
        },
    }


def _set_cp_threshold(config: dict, cp_threshold: float) -> None:
    metrics = config.setdefault("metrics", {})
    td_cfg = metrics.setdefault("temporal_disagreement", {})
    td_cfg["cp_threshold"] = float(cp_threshold)


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
    record_config_path = repo_dir / "meta" / "record_config.json"

    calibration_scores = load_temporal_disagreement(metrics_path)
    cp_threshold, q_level, n = compute_cp_threshold(calibration_scores, args.alpha)

    print(f"repo_id: {args.repo_id}")
    print(f"samples (n): {n}")
    print(f"alpha: {args.alpha}")
    print(f"q_level: {q_level}")
    print(f"cp_threshold: {cp_threshold}")
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
