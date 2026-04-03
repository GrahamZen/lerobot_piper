"""VLM (Gemini) pipeline and video utilities for checkpoint verification.

Public API:
    run_vlm_pipeline(training_dataset, training_dataset_root, ep_df, ...) -> dict[int, list[int]]
    load_vlm_checkpoints(pretrained_path) -> dict[int, list[int]] | None
    save_vlm_checkpoints(pretrained_path, verified_cp_ts_by_ep) -> None
    VLM_CHECKPOINTS_FILENAME: str
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

VLM_CHECKPOINTS_FILENAME = "vlm_checkpoints.json"


# ---------------------------------------------------------------------------
# Video clip extraction (for VLM upload)
# ---------------------------------------------------------------------------


def _find_middle_cam_video_file(dataset_root: Path) -> Path | None:
    cam_dir = dataset_root / "videos" / "observation.images.middle"
    if not cam_dir.exists():
        return None
    for p in sorted(cam_dir.glob("chunk-*/file-*.mp4")):
        return p
    return None


def _find_per_episode_video(dataset_root: Path, episode_index: int) -> Path | None:
    cam_dir = dataset_root / "videos" / "observation.images.middle"
    if not cam_dir.exists():
        return None
    patterns = [
        f"**/episode_{episode_index:06d}.mp4",
        f"chunk-*/episode_{episode_index:06d}.mp4",
        f"**/ep{episode_index:03d}_middle.mp4",
    ]
    for pat in patterns:
        for p in cam_dir.glob(pat):
            return p
    return None


def extract_episode_video_clip(
    dataset_root: Path,
    episode_index: int,
    from_idx: int,
    to_idx: int,
    fps: float,
    output_dir: Path,
) -> Path | None:
    """Extract a 1fps video clip for the given episode using ffmpeg."""
    output_path = output_dir / f"ep{episode_index:03d}_middle_1fps.mp4"
    if output_path.exists():
        return output_path

    source_video = _find_middle_cam_video_file(dataset_root)
    if source_video is not None:
        start_time = from_idx / fps
        duration = (to_idx - from_idx) / fps
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(source_video),
            "-t",
            str(duration),
            "-vf",
            "fps=1",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-loglevel",
            "error",
            str(output_path),
        ]
    else:
        per_ep = _find_per_episode_video(dataset_root, episode_index)
        if per_ep is None:
            return None
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(per_ep),
            "-vf",
            "fps=1",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-loglevel",
            "error",
            str(output_path),
        ]

    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed for ep {episode_index}: {e}")
        return None


# ---------------------------------------------------------------------------
# VLM (Gemini) pipeline
# ---------------------------------------------------------------------------


def _build_gemini_client(api_key: str | None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("Gemini API key required. Pass --gemini_api_key or set GEMINI_API_KEY env var.")
    return genai.Client(api_key=key)


def _upload_video_with_retry(client, video_path: Path, ep_id: int, max_retries: int = 5):
    from google.api_core import exceptions
    # from google.api_core import errors as google_errors

    print(f"  Uploading video for ep {ep_id}: {video_path.name}")
    vfile = client.files.upload(file=str(video_path))

    retries = 0
    while vfile.state.name == "PROCESSING":
        time.sleep(10)
        try:
            vfile = client.files.get(name=vfile.name)
        except exceptions.ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"  [WARN] ep {ep_id}: server 503, backing off...")
                time.sleep(5)
            else:
                raise
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise
            print(f"  [WARN] ep {ep_id}: network error ({e}), retry {retries}/{max_retries}")
            time.sleep(5)

    return vfile


def _analyze_episode(
    ep_id: int, vfile, client, gemini_model: str, dataset_fps: float
) -> tuple[int, list[int]]:
    """Query Gemini VLM for safe checkpoint timestamps in one episode."""
    from google.genai import types
    from pydantic import BaseModel, Field

    class EpisodeAnalysis(BaseModel):
        reasoning: str = Field(
            description="Analyze the bimanual action state second by second, describing subtasks and the relevant checkpoints."
        )
        safe_checkpoints: list[int] = Field(
            description="A list of integer second timestamps that meet the definition of a checkpoint. Return an empty list if none exist."
        )

    prompt = (
        "You are an expert robotic demonstration annotator analyzing a 1 FPS video. "
        "This video shows a bimanual robot performing a manipulation task. "
        "Your goal is to identify ALL time points (in integer seconds) where the robot is in a 'checkpoint' state. "
        "A checkpoint is defined by meeting these conditions:\n\n"
        "1. Imagine if the robot fails to complete a subtask (e.g., performing a grasp/manipulation with the end-effectors to an object). "
        "The checkpoint would be a prior point in history where a rewind-and-resume would increase the chance of success.\n"
        "2. Checkpoints must represent the start of an approach trajectory. Select the moment the arm is in transit or just "
        "entering the workspace, providing at least a 1-2 second buffer before it reaches a pre-manipulation hover or begins "
        "fine-grained alignment with the object.\n"
        "3. EXCLUSION: The final state where the entire overall task is fully completed should NOT be considered a safe checkpoint. "
        "We only want intermediate subtask checkpoints.\n\n"
        "There are typically multiple subtasks in a task. First, provide your reasoning by describing the timeline of the robot's actions. "
        "Then, extract the exact integer seconds where the safe checkpoints occur."
    )

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=[vfile, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EpisodeAnalysis,
                temperature=0.1,
            ),
        )
        result = json.loads(response.text)
        seconds_list: list[int] = result.get("safe_checkpoints", [])
        ts_list = [int(sec * dataset_fps) for sec in seconds_list]
        print(
            f"  ep {ep_id}: reasoning='{result.get('reasoning', '')[:80]}...' checkpoints_sec={seconds_list}"
        )
        return ep_id, ts_list
    except Exception as e:
        print(f"  [ERROR] ep {ep_id}: VLM query failed: {e}")
        return ep_id, []


def run_vlm_pipeline(
    training_dataset: LeRobotDataset,
    training_dataset_root: Path,
    ep_df,
    gemini_api_key: str | None,
    gemini_model: str,
    vlm_workers: int,
) -> dict[int, list[int]]:
    """Run the full VLM pipeline: clip videos → upload → query → parse.

    Returns verified_cp_ts_by_ep: {ep_id -> [timestep_offset, ...]}
    """
    client = _build_gemini_client(gemini_api_key)
    dataset_fps = float(training_dataset.fps)

    clip_dir = training_dataset_root / "vlm_clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[VLM] Extracting 1fps episode video clips to {clip_dir}...")

    episode_video_paths: dict[int, Path] = {}
    for _, ep_row in tqdm(ep_df.iterrows(), total=len(ep_df), desc="Clipping episodes"):
        ep_id = int(ep_row["episode_index"])
        from_idx = int(ep_row["dataset_from_index"])
        to_idx = int(ep_row["dataset_to_index"])
        video_path = extract_episode_video_clip(
            training_dataset_root, ep_id, from_idx, to_idx, dataset_fps, clip_dir
        )
        if video_path is not None:
            episode_video_paths[ep_id] = video_path
        else:
            tqdm.write(f"  [WARN] Could not extract video for ep {ep_id}, skipping.")

    if not episode_video_paths:
        raise RuntimeError("No episode videos could be extracted. Cannot run VLM pipeline.")

    print(f"\n[VLM] Uploading {len(episode_video_paths)} videos to Gemini...")
    video_files: dict[int, object] = {}
    for ep_id, video_path in sorted(episode_video_paths.items()):
        try:
            vfile = _upload_video_with_retry(client, video_path, ep_id)
            video_files[ep_id] = vfile
            print(f"  ep {ep_id}: uploaded → {vfile.name}")
        except Exception as e:
            print(f"  [ERROR] ep {ep_id}: upload failed: {e}")

    print(f"\n[VLM] Querying VLM (workers={vlm_workers})...")
    verified_cp_ts_by_ep: dict[int, list[int]] = {}
    with ThreadPoolExecutor(max_workers=vlm_workers) as executor:
        future_to_ep = {
            executor.submit(_analyze_episode, ep_id, vfile, client, gemini_model, dataset_fps): ep_id
            for ep_id, vfile in video_files.items()
        }
        for future in as_completed(future_to_ep):
            ep_id_result, ts_list = future.result()
            verified_cp_ts_by_ep[ep_id_result] = ts_list

    print(f"\n[VLM] Done. Checkpoints found for {len(verified_cp_ts_by_ep)} episodes.")
    return verified_cp_ts_by_ep


# ---------------------------------------------------------------------------
# VLM checkpoint cache (vlm_checkpoints.json)
# ---------------------------------------------------------------------------


def load_vlm_checkpoints(pretrained_path: Path) -> dict[int, list[int]] | None:
    cache_path = pretrained_path / VLM_CHECKPOINTS_FILENAME
    if not cache_path.exists():
        return None
    with cache_path.open() as f:
        raw: dict = json.load(f)
    return {int(k): [int(t) for t in v] for k, v in raw.items()}


def save_vlm_checkpoints(pretrained_path: Path, verified_cp_ts_by_ep: dict[int, list[int]]) -> None:
    cache_path = pretrained_path / VLM_CHECKPOINTS_FILENAME
    serializable = {str(k): v for k, v in sorted(verified_cp_ts_by_ep.items())}
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"[VLM] Saved verified_cp_ts_by_ep to {cache_path}")
