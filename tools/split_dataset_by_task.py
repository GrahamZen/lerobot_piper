#!/usr/bin/env python3
"""
Split a LeRobot dataset by task name.

Usage:
  python split_dataset_by_task.py <repo_id>                    # list all tasks with episode counts
  python split_dataset_by_task.py <repo_id> --taskname <name>  # split dataset for given task
"""

import argparse
import re
import sys

from lerobot.datasets.dataset_tools import split_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME


def get_task_episodes(dataset):
    """Return dict mapping taskname -> list of episode indices."""
    task_episodes = {}
    for i, ep in enumerate(dataset.meta.episodes):
        taskname = ep["tasks"]
        if isinstance(taskname, list):
            taskname = taskname[0] if len(taskname) == 1 else str(taskname)
        key = str(taskname)
        task_episodes.setdefault(key, []).append(i)
    return task_episodes


def build_libero_task_index():
    """Build a mapping from language description (lowercase) -> (suite, task_id).

    Returns an empty dict if libero is not installed.
    """
    try:
        from libero.libero import benchmark
    except ImportError:
        return {}

    index = {}
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
    bench = benchmark.get_benchmark_dict()
    for suite_name in suites:
        try:
            suite = bench[suite_name]()
            for i, task in enumerate(suite.tasks):
                key = task.language.lower().strip()
                # keep first match (earlier suites take priority)
                if key not in index:
                    index[key] = (suite_name, i)
        except Exception:  # noqa: BLE001  # nosec B110
            pass
    return index


def sanitize_task_name(task_name):
    """Convert task name to a valid repo-name slug."""
    name = task_name.lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    # Truncate to avoid overly long names
    if len(name) > 60:
        name = name[:60].rstrip("_")
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Split a LeRobot dataset by task name.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("repo_id", help="Source dataset repo ID (e.g. HuggingFaceVLA/libero)")
    parser.add_argument(
        "--taskname",
        default=None,
        help="Task name to split out. Use the exact string shown in --list output.",
    )
    parser.add_argument(
        "--new-repo-id",
        default=None,
        help="Override the generated new repo ID (default: local/<base>_<task_slug>)",
    )
    parser.add_argument(
        "--no-encode-videos",
        action="store_true",
        help="Don't encode videos after splitting (useful if you just want to inspect the split dataset or encode videos later).",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id} ...")
    dataset = LeRobotDataset(args.repo_id)
    task_episodes = get_task_episodes(dataset)

    if args.taskname is None:
        # List all tasks sorted by episode count descending
        sorted_tasks = sorted(task_episodes.items(), key=lambda x: len(x[1]), reverse=True)

        print("Building LIBERO task index ...")
        libero_index = build_libero_task_index()
        has_libero = bool(libero_index)

        if has_libero:
            print(f"\n{'Episodes':>10}  {'env.task':<16}  {'task_id':>7}  Task Name")
            print("-" * 110)
            for taskname, episodes in sorted_tasks:
                match = libero_index.get(taskname.lower().strip())
                suite = match[0] if match else "?"
                tid = str(match[1]) if match else "?"
                print(f"{len(episodes):>10}  {suite:<16}  {tid:>7}  {taskname}")
        else:
            print(f"\n{'Episodes':>10}  Task Name")
            print("-" * 80)
            for taskname, episodes in sorted_tasks:
                print(f"{len(episodes):>10}  {taskname}")

        print(f"\nTotal tasks: {len(sorted_tasks)}")
        return

    # Find matching task (exact or substring)
    matched_key = None
    if args.taskname in task_episodes:
        matched_key = args.taskname
    else:
        # Try substring match
        candidates = [k for k in task_episodes if args.taskname.lower() in k.lower()]
        if len(candidates) == 1:
            matched_key = candidates[0]
            print(f"Matched task: {matched_key}")
        elif len(candidates) > 1:
            print("Multiple tasks match the given taskname:")
            for c in candidates:
                print(f"  - {c}  ({len(task_episodes[c])} episodes)")
            sys.exit(1)
        else:
            print(f"No task found matching: {args.taskname!r}")
            sys.exit(1)

    episode_ids = task_episodes[matched_key]
    print(f"Task: {matched_key}")
    print(f"Episodes ({len(episode_ids)}): {episode_ids}")

    # Generate split key and output directory
    base_name = args.repo_id.split("/")[-1]
    task_slug = sanitize_task_name(matched_key)

    if args.new_repo_id:
        parts = args.new_repo_id.rsplit("/", 1)
        split_key = parts[-1]
        output_dir = HF_LEROBOT_HOME / parts[0] if len(parts) == 2 else HF_LEROBOT_HOME
        new_repo_id = args.new_repo_id
    else:
        split_key = f"{base_name}_{task_slug}"
        output_dir = HF_LEROBOT_HOME / "local"
        new_repo_id = f"local/{split_key}"

    save_path = output_dir / split_key

    # Check if dataset already exists
    if save_path.exists() and any(save_path.iterdir()):
        print(f"Dataset already exists at {save_path}, skipping split.")
    else:
        print(f"Saving to: {save_path}")
        print(f"\nSplitting dataset with key '{split_key}' ({len(episode_ids)} episodes) ...")
        result_datasets = split_dataset(
            dataset,
            splits={split_key: episode_ids},
            output_dir=output_dir,
        )
        new_ds = result_datasets[split_key]
        print(f"Done: {new_ds.meta.total_episodes} episodes, {new_ds.meta.total_frames} frames")
        print(f"Saved to: {save_path}")

    if not args.no_encode_videos:
        import sys as _sys
        from pathlib import Path as _Path

        _tools_dir = _Path(__file__).parent
        _sys.path.insert(0, str(_tools_dir))
        from convert_images_to_videos import convert_dataset

        print("\nEncoding videos ...")
        convert_dataset(save_path)

    # Look up LIBERO env config for this task
    libero_index = build_libero_task_index()
    match = libero_index.get(matched_key.lower().strip())
    env_task = match[0] if match else "<env.task>"
    task_id = str(match[1]) if match else "<task_id>"

    job_name = f"act_{split_key}"
    output_train_dir = f"outputs/train/{split_key}"

    print(f"""
Training command:

uv run lerobot-train \\
  --policy.type=act \\
  --env.type=libero \\
  --env.task={env_task} \\
  --env.task_ids="[{task_id}]" \\
  --dataset.repo_id={new_repo_id} \\
  --output_dir={output_train_dir} \\
  --job_name={job_name} \\
  --wandb.mode=offline \\
  --policy.push_to_hub=false \\
  --dataset.image_transforms.enable=true \\
  --policy.use_amp=false \\
  --batch_size=16 \\
  --steps=15000 \\
  --eval_freq=1500 \\
  --save_freq=1500 \\
  --eval.batch_size=20 \\
  --eval.n_episodes=20""")


if __name__ == "__main__":
    main()
