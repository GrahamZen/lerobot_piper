"""
Usage: uv run python tools/viz/browse_first_frames.py --repo_id local/lerobot_pick_cups_aug
"""

import argparse
import io

import numpy as np
import rerun as rr
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def browse_first_frames(repo_id: str, root: str | None = None):
    print(f"Loading dataset: {repo_id}")
    try:
        dataset = LeRobotDataset(repo_id, root=root)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Ensure episodes metadata is loaded
    if dataset.meta.episodes is None:
        try:
            from lerobot.datasets.utils import load_episodes

            dataset.meta.episodes = load_episodes(dataset.root)
        except Exception as e:
            print(f"Warning: Could not load episodes metadata: {e}")

    episodes = dataset.meta.episodes
    if episodes is None:
        print("Error: episodes metadata not found or failed to load.")
        return

    num_episodes = len(episodes)
    print(f"Found {num_episodes} episodes.")

    print("Initializing Rerun...")
    rr.init(f"First Frame Browser: {repo_id}", spawn=True)

    for episode_idx in range(num_episodes):
        ep_meta = episodes[episode_idx]

        # Get start index (dataset_from_index can be int or list)
        from_idx = ep_meta["dataset_from_index"]
        # Get start index (dataset_from_index can be int or list)
        from_idx = ep_meta["dataset_from_index"]
        from_idx = int(from_idx[0]) if isinstance(from_idx, list) else int(from_idx)

        # Use episode index as the timeline step
        rr.set_time_sequence("episode", episode_idx)
        rr.log("overlay/episode_id", rr.TextDocument(f"# Episode {episode_idx}"), static=False)

        try:
            item = dataset[from_idx]
        except Exception as e:
            print(f"Error loading frame at index {from_idx} (Episode {episode_idx}): {e}")
            continue

        # Log images
        image_keys = [k for k in item if "image" in k]
        for img_key in image_keys:
            img_data = item[img_key]
            clean_key = img_key.replace("observation.images.", "")

            # Handle different image formats (dict with bytes vs tensor/array)
            if isinstance(img_data, dict) and "bytes" in img_data:
                img = Image.open(io.BytesIO(img_data["bytes"]))
                rr.log(f"cameras/{clean_key}", rr.Image(img))
            else:
                arr = img_data
                if hasattr(arr, "numpy"):
                    arr = arr.numpy()
                # If CHW (channels first), convert to HWC (height, width, channels)
                # Typically 3 channels (RGB) or 1? checking for common CHW shapes like (3, H, W)
                if arr.ndim == 3 and arr.shape[0] <= 4:
                    arr = np.transpose(arr, (1, 2, 0))

                rr.log(f"cameras/{clean_key}", rr.Image(arr))

    print(f"Finished logging {num_episodes} first frames to Rerun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Browse the first frame of every episode in a LeRobot dataset."
    )
    parser.add_argument(
        "--repo_id", type=str, required=True, help="Dataset repository ID (e.g. local/lerobot_pick_cups_aug)"
    )
    parser.add_argument("--root", type=str, default=None, help="Dataset root directory (optional)")

    args = parser.parse_args()
    browse_first_frames(args.repo_id, args.root)
