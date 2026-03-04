import io
from pathlib import Path
from typing import Any

import imageio
import numpy as np
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


class EpisodeFailureExtractor:
    SAFETY_MARGIN = 40

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.dataset_path = Path(repo_id).expanduser()
        self.dataset_name = (
            self.dataset_path.name if self.dataset_path.exists() else repo_id.replace("/", "_")
        )
        self.dataset = LeRobotDataset(repo_id, root=None)

        if self.dataset.meta.episodes is None:
            from lerobot.datasets.utils import load_episodes

            self.dataset.meta.episodes = load_episodes(self.dataset.root)

        self.failure_metrics = load_failure_metrics_jsonl(self.dataset.root)
        self.failure_handling_cfg = load_failure_handling_json(self.dataset.root)
        self.failure_config = load_failure_config(self.dataset.root)
        self.td_cp_threshold = self._resolve_cp_threshold(self.failure_handling_cfg)
        (
            self.smoothed_td_by_step,
            self.previous_checkpoint_by_step,
            self.checkpoint_flag_by_step,
            self.recent_checkpoints_by_step,
        ) = replay_checkpoint_series(
            self.failure_metrics,
            self.failure_config,
            safety_margin=self.SAFETY_MARGIN,
        )

    def save_failure_images(self, episode_index: int, save_dir: str | Path) -> dict:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        from_idx, to_idx = self.get_episode_bounds(episode_index)
        failed_step = None
        failed_td = None

        for step_idx in range(from_idx, to_idx):
            if step_idx in self.failure_metrics:
                raw_td = float(self.failure_metrics[step_idx].get("temporal_disagreement", 0.0))
                if raw_td > self.td_cp_threshold:
                    failed_step = step_idx
                    failed_td = raw_td
                    break

        result: dict[str, Any] = {
            "repo_id": self.repo_id,
            "episode_index": int(episode_index),
            "from_idx": int(from_idx),
            "to_idx": int(to_idx),
            "threshold": float(self.td_cp_threshold),
            "failed_step": failed_step,
            "failed_temporal_disagreement": failed_td,
            "status": "failure_found" if failed_step is not None else "no_failure",
            "current_images": {},
            "checkpoint_images": [],
            "checkpoint_steps": [],
        }

        if failed_step is None:
            return result

        curr_item = self.dataset[failed_step]
        curr_imgs = self._get_image(curr_item)

        checkpoint_steps = self.recent_checkpoints_by_step.get(failed_step, [])[-5:]
        result["checkpoint_steps"] = [int(cp) for cp in checkpoint_steps]

        failure_step_dir = save_dir / "failure" / str(failed_step)
        failure_step_dir.mkdir(parents=True, exist_ok=True)

        for cam in ["left", "middle", "right"]:
            curr_path = failure_step_dir / f"{cam}.png"
            curr_imgs[cam].save(curr_path)
            result["current_images"][cam] = str(curr_path)

        for cp_idx, cp_step in enumerate(reversed(checkpoint_steps), start=1):
            cp_item = self.dataset[int(cp_step)]
            cp_imgs = self._get_image(cp_item)
            cp_entry: dict[str, Any] = {"offset": -cp_idx, "step": int(cp_step), "images": {}}
            checkpoint_step_dir = save_dir / "checkpoints" / str(cp_step)
            checkpoint_step_dir.mkdir(parents=True, exist_ok=True)
            for cam in ["left", "middle", "right"]:
                cp_path = checkpoint_step_dir / f"{cam}.png"
                cp_imgs[cam].save(cp_path)
                cp_entry["images"][cam] = str(cp_path)
            result["checkpoint_images"].append(cp_entry)
        return result

    def save_episode_video(
        self,
        episode_index: int,
        save_dir: str | Path,
        fps: int,
        scale: float,
    ) -> dict:
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        from_idx, to_idx = self.get_episode_bounds(episode_index)
        video_path = save_dir / f"episode_{episode_index}.mp4"

        writer = imageio.get_writer(video_path, fps=fps)
        frame_count = 0
        original_size = None
        scaled_size = None

        try:
            for step_idx in range(from_idx, to_idx):
                item = self.dataset[step_idx]
                imgs = self._get_image(item)
                frame = np.concatenate(
                    [
                        np.array(imgs["left"]),
                        np.array(imgs["middle"]),
                        np.array(imgs["right"]),
                    ],
                    axis=1,
                )

                frame_img = Image.fromarray(frame).convert("RGB")
                original_size = frame_img.size

                if scale != 1.0:
                    new_w = max(1, int(round(frame_img.width * scale)))
                    new_h = max(1, int(round(frame_img.height * scale)))
                    frame_img = frame_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

                scaled_size = frame_img.size
                writer.append_data(np.array(frame_img, dtype=np.uint8))
                frame_count += 1
        finally:
            writer.close()

        return {
            "repo_id": self.repo_id,
            "episode_index": int(episode_index),
            "from_idx": int(from_idx),
            "to_idx": int(to_idx),
            "fps": int(fps),
            "scale": float(scale),
            "frame_count": int(frame_count),
            "original_size_wh": list(original_size) if original_size is not None else None,
            "scaled_size_wh": list(scaled_size) if scaled_size is not None else None,
            "video_path": str(video_path),
        }

    def get_episode_bounds(self, episode_index: int) -> tuple[int, int]:
        ep_meta = self.dataset.meta.episodes[episode_index]
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

    def _resolve_cp_threshold(self, failure_handling_cfg: dict) -> float:
        td_cp_threshold = None

        if "metrics" in failure_handling_cfg and "temporal_disagreement" in failure_handling_cfg["metrics"]:
            td_cp_threshold = failure_handling_cfg["metrics"]["temporal_disagreement"].get("cp_threshold")

        if td_cp_threshold is None and "cp_threshold" in failure_handling_cfg:
            td_cp_threshold = failure_handling_cfg.get("cp_threshold")

        if td_cp_threshold is None:
            return float("inf")
        return float(td_cp_threshold)

    def _to_uint8_image_array(self, arr):
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

        arr = np.clip(arr.astype(np.float32), 0.0, 255.0).astype(np.uint8)
        return arr

    def _get_image(self, item, camera_prefix="observation.images."):
        cameras = ["left", "middle", "right"]
        imgs = {}
        for cam in cameras:
            key = f"{camera_prefix}{cam}"
            if key in item:
                img_data = item[key]
                if isinstance(img_data, dict) and "bytes" in img_data:
                    imgs[cam] = Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
                else:
                    arr = img_data.numpy() if hasattr(img_data, "numpy") else img_data
                    arr = self._to_uint8_image_array(arr)
                    imgs[cam] = Image.fromarray(arr).convert("RGB")
            else:
                imgs[cam] = Image.new("RGB", (100, 100), color=(0, 0, 0))
        return imgs
