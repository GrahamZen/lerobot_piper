#!/usr/bin/env python3
"""
Convert a LeRobot dataset that stores images as raw tensors (dtype='image' in parquet)
into one that stores them as mp4 video files (dtype='video'), matching the format used
by real-robot datasets.

Usage:
  python convert_images_to_videos.py <repo_id>
  python convert_images_to_videos.py local/libero_my_task --fps 10 --vcodec libx264 --pix-fmt yuv420p

The script modifies the dataset in-place:
  1. Encodes video files under videos/{key}/chunk-{c:03d}/file-{f:03d}.mp4
  2. Updates info.json  (dtype image→video, adds video_keys)
  3. Rewrites episodes parquet files with videos/* timestamp columns
"""

import argparse
import io
import json
import logging
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def decode_image(raw) -> np.ndarray:
    """Decode a stored image value to an HWC uint8 numpy array.

    Handles:
      - dict with 'bytes' key  (HuggingFace Image feature serialisation)
      - raw bytes
      - numpy array (already decoded)
    """
    if isinstance(raw, dict):
        raw = raw.get("bytes") or raw.get("path")
        if raw is None:
            raise ValueError("Image dict has neither 'bytes' nor 'path'")
    if isinstance(raw, (bytes, bytearray)):
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img)
    if isinstance(raw, np.ndarray):
        if raw.dtype != np.uint8:
            raw = (np.clip(raw, 0, 1) * 255).astype(np.uint8)
        return raw
    raise TypeError(f"Unsupported image type: {type(raw)}")


def find_data_files(dataset_root: Path) -> list[tuple[int, int, Path]]:
    """Return sorted list of (chunk_index, file_index, path) for all data parquet files."""
    result = []
    data_dir = dataset_root / "data"
    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        chunk_idx = int(chunk_dir.name.split("-")[1])
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            file_idx = int(parquet_file.stem.split("-")[1])
            result.append((chunk_idx, file_idx, parquet_file))
    return result


def find_episodes_files(dataset_root: Path) -> list[tuple[int, int, Path]]:
    """Return sorted list of (chunk_index, file_index, path) for all episodes parquet files."""
    result = []
    ep_dir = dataset_root / "meta" / "episodes"
    for chunk_dir in sorted(ep_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        chunk_idx = int(chunk_dir.name.split("-")[1])
        for parquet_file in sorted(chunk_dir.glob("file-*.parquet")):
            file_idx = int(parquet_file.stem.split("-")[1])
            result.append((chunk_idx, file_idx, parquet_file))
    return result


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------


def encode_video_for_file(
    frames_by_episode: dict[int, list[tuple[float, np.ndarray]]],
    output_path: Path,
    fps: float,
    vcodec: str,
    pix_fmt: str,
) -> dict[int, tuple[float, float]]:
    """
    Encode all frames from multiple episodes into one mp4 file.

    Args:
        frames_by_episode: {episode_index: [(timestamp, hwc_uint8_array), ...]}
                           Episodes must be in ascending order of episode_index.
        output_path: Path to write the .mp4 file.
        fps: Video frame rate.
        vcodec: FFmpeg video codec name.
        pix_fmt: Output pixel format.

    Returns:
        {episode_index: (from_timestamp, to_timestamp)}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten all frames in episode order
    ordered_episodes = sorted(frames_by_episode.keys())
    all_frames: list[tuple[int, float, np.ndarray]] = []
    for ep_idx in ordered_episodes:
        for ts, frame in frames_by_episode[ep_idx]:
            all_frames.append((ep_idx, ts, frame))

    if not all_frames:
        raise ValueError("No frames to encode")

    first_frame = all_frames[0][2]
    height, width = first_frame.shape[:2]

    # Compute from_timestamp / to_timestamp as actual time offsets in the output video.
    # pts counter is global across all episodes; time = pts / fps.
    episode_timestamps: dict[int, tuple[float, float]] = {}
    pts_counter = 0
    for ep_idx in ordered_episodes:
        ep_frames = frames_by_episode[ep_idx]
        ep_start_pts = pts_counter
        ep_end_pts = pts_counter + len(ep_frames) - 1
        episode_timestamps[ep_idx] = (ep_start_pts / fps, ep_end_pts / fps)
        pts_counter += len(ep_frames)

    container = av.open(str(output_path), mode="w")
    stream = container.add_stream(vcodec, rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = pix_fmt
    stream.options = {"crf": "23", "preset": "fast"}

    for pts, (_ep_idx, _ts, frame_arr) in enumerate(all_frames):
        img = Image.fromarray(frame_arr, mode="RGB")
        av_frame = av.VideoFrame.from_image(img)
        av_frame.pts = pts
        for packet in stream.encode(av_frame):
            container.mux(packet)

    # Flush
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return episode_timestamps


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def load_info(dataset_root: Path) -> dict:
    with open(dataset_root / "meta" / "info.json") as f:
        return json.load(f)


def save_info(dataset_root: Path, info: dict) -> None:
    with open(dataset_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    log.info("Updated meta/info.json")


def update_info_for_videos(info: dict, image_keys: list[str]) -> dict:
    """Change dtype from 'image' to 'video' for the given keys and update video_keys."""
    for key in image_keys:
        if key in info["features"]:
            info["features"][key]["dtype"] = "video"
            # Reorder names to channel-first for video convention if needed
            feat = info["features"][key]
            if feat.get("names") == ["height", "width", "channel"]:
                feat["names"] = ["channel", "height", "width"]
            # shape: keep as-is (lerobot uses [H, W, C] for image, [C, H, W] for video)
    info["video_keys"] = image_keys
    return info


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert_dataset(
    dataset_root: Path,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    overwrite: bool = False,
) -> None:
    info = load_info(dataset_root)
    fps = info["fps"]

    # Identify image keys to convert
    image_keys = [k for k, v in info["features"].items() if v.get("dtype") == "image"]
    if not image_keys:
        log.info("No 'image' dtype features found — already converted or no images.")
        return

    log.info(f"Converting image keys: {image_keys}")
    log.info(f"FPS: {fps}, codec: {vcodec}, pix_fmt: {pix_fmt}")

    # Ensure video_path template exists in info
    video_path_template = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    if not info.get("video_path"):
        info["video_path"] = video_path_template

    data_files = find_data_files(dataset_root)
    log.info(f"Found {len(data_files)} data parquet file(s)")

    # Group data files by chunk so all episodes in a chunk go into one video file.
    # This matches real-robot dataset structure: one .mp4 per chunk per camera.
    chunks: dict[int, list[tuple[int, Path]]] = {}
    for chunk_idx, _file_idx, parquet_path in data_files:
        chunks.setdefault(chunk_idx, []).append((_file_idx, parquet_path))

    # episode_index -> {chunk_index, file_index, from_ts, to_ts} per video_key
    video_ep_meta: dict[str, dict[int, dict]] = {k: {} for k in image_keys}

    for chunk_idx, chunk_files in tqdm(sorted(chunks.items()), desc="Encoding chunks"):
        # Load all data in this chunk (sorted by file index = episode order)
        chunk_df = pd.concat(
            [pd.read_parquet(p) for _, p in sorted(chunk_files)],
            ignore_index=True,
        )

        present_keys = [k for k in image_keys if k in chunk_df.columns]
        if not present_keys:
            log.warning(f"No image columns in chunk {chunk_idx}, skipping")
            continue

        episodes_in_chunk = sorted(chunk_df["episode_index"].unique())

        for key in present_keys:
            # Always file_index=0 per chunk (one video per chunk per camera)
            video_path = dataset_root / info["video_path"].format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=0,
            )

            if video_path.exists() and not overwrite:
                log.info(f"  Video exists, skipping: {video_path}")
                # Re-read timestamps from video to recover from_timestamp
                # Reconstruct from_timestamp by counting frames per episode
                pts_counter = 0
                for ep_idx in episodes_in_chunk:
                    ep_df = chunk_df[chunk_df["episode_index"] == ep_idx]
                    n = len(ep_df)
                    video_ep_meta[key][int(ep_idx)] = {
                        "chunk_index": chunk_idx,
                        "file_index": 0,
                        "from_timestamp": pts_counter / fps,
                        "to_timestamp": (pts_counter + n - 1) / fps,
                    }
                    pts_counter += n
                continue

            frames_by_episode: dict[int, list[tuple[float, np.ndarray]]] = {}
            for ep_idx in episodes_in_chunk:
                ep_df = chunk_df[chunk_df["episode_index"] == ep_idx].sort_values("frame_index")
                frames = []
                for _, row in ep_df.iterrows():
                    arr = decode_image(row[key])
                    frames.append((float(row["timestamp"]), arr))
                frames_by_episode[int(ep_idx)] = frames

            log.info(
                f"  Encoding {key} chunk-{chunk_idx:03d}: {len(episodes_in_chunk)} episodes → {video_path.name}"
            )
            ep_timestamps = encode_video_for_file(frames_by_episode, video_path, fps, vcodec, pix_fmt)

            for ep_idx, (from_ts, to_ts) in ep_timestamps.items():
                video_ep_meta[key][ep_idx] = {
                    "chunk_index": chunk_idx,
                    "file_index": 0,
                    "from_timestamp": from_ts,
                    "to_timestamp": to_ts,
                }

    # --- Update episodes parquet files ---
    log.info("Updating episodes metadata parquet files...")
    ep_files = find_episodes_files(dataset_root)

    for _chunk_idx, _file_idx, ep_parquet in tqdm(ep_files, desc="Updating episodes"):
        df_ep = pd.read_parquet(ep_parquet)

        for key in image_keys:
            col_chunk = f"videos/{key}/chunk_index"
            col_file = f"videos/{key}/file_index"
            col_from = f"videos/{key}/from_timestamp"
            col_to = f"videos/{key}/to_timestamp"

            # Skip if already present
            if col_chunk in df_ep.columns and not overwrite:
                continue

            chunk_vals, file_vals, from_ts_list, to_ts_list = [], [], [], []
            for ep_idx in df_ep["episode_index"]:
                meta = video_ep_meta[key].get(int(ep_idx))
                if meta is None:
                    raise ValueError(f"Missing video metadata for episode {ep_idx}, key {key}")
                chunk_vals.append(meta["chunk_index"])
                file_vals.append(meta["file_index"])
                from_ts_list.append(meta["from_timestamp"])
                to_ts_list.append(meta["to_timestamp"])

            df_ep[col_chunk] = chunk_vals
            df_ep[col_file] = file_vals
            df_ep[col_from] = from_ts_list
            df_ep[col_to] = to_ts_list

        df_ep.to_parquet(ep_parquet, index=False)

    # --- Update info.json ---
    info = update_info_for_videos(info, image_keys)
    save_info(dataset_root, info)

    log.info("Done! Dataset now uses video files.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_dataset_root(repo_id: str) -> Path:
    """Resolve repo_id to a local path under HF_LEROBOT_HOME."""
    parts = repo_id.split("/")
    candidate = HF_LEROBOT_HOME / Path(*parts)
    if candidate.exists():
        return candidate
    # Try as absolute path
    p = Path(repo_id)
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Dataset not found at {candidate}. Make sure it is downloaded locally under {HF_LEROBOT_HOME}."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo_id", help="Dataset repo ID (e.g. local/my_split) or absolute path")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: from info.json)")
    parser.add_argument("--vcodec", default="libx264", help="Video codec (default: libx264)")
    parser.add_argument("--pix-fmt", default="yuv420p", help="Pixel format (default: yuv420p)")
    parser.add_argument("--overwrite", action="store_true", help="Re-encode even if video files exist")
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.repo_id)
    log.info(f"Dataset root: {dataset_root}")

    if args.fps is not None:
        # Will be picked up by overriding info fps temporarily; handled inside convert_dataset
        # via load_info — just pass it through
        info = load_info(dataset_root)
        info["fps"] = args.fps

    convert_dataset(
        dataset_root=dataset_root,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
