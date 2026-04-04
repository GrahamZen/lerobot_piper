#!/usr/bin/env python3
"""
Download, convert, and optionally filter a RoboCasa dataset.

The output is always a v3.0 LeRobot dataset.

Modes
-----
list     (--list)
    Download the raw dataset if needed, then print the episode ↔
    layout_id / style_id distribution table. No files are written.

convert  (no --layout-ids / --style-ids)
    Download the raw v2.1 dataset, convert it to v3.0 in-place, and
    symlink it into the local HuggingFace cache.

filter   (--layout-ids and/or --style-ids)
    Same as convert, then extract the matching episodes into a new
    repo with an auto-generated or user-supplied repo id.

Usage examples
--------------
  # List episode distribution
  python tools/filter_robocasa_by_layout.py CoffeeServeMug --list

  # Full convert (no filtering)
  python tools/filter_robocasa_by_layout.py CoffeeServeMug
  python tools/filter_robocasa_by_layout.py CoffeeServeMug --split pretrain

  # Filter by layout / style
  python tools/filter_robocasa_by_layout.py CoffeeServeMug --layout-ids 2 3 --style-ids 5 6
  python tools/filter_robocasa_by_layout.py CoffeeServeMug --layout-ids 0 --dry-run
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Robocasa registry helpers
# ---------------------------------------------------------------------------


def _get_task_info(task_name: str) -> tuple[str, str]:
    """Return (kind, split) for a task name. Raises ValueError if not found."""
    from robocasa.utils.dataset_registry import (
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
        PRETRAINING_TASKS,
        TARGET_TASKS,
    )

    if task_name in ATOMIC_TASK_DATASETS:
        kind = "atomic"
    elif task_name in COMPOSITE_TASK_DATASETS:
        kind = "composite"
    else:
        raise ValueError(f"Task '{task_name}' not found in robocasa registry.")

    in_target = any(task_name in tasks for tasks in TARGET_TASKS.values())
    in_pretrain = any(task_name in tasks for tasks in PRETRAINING_TASKS.values())

    if in_target and in_pretrain:
        split = "both"
    elif in_target:
        split = "target"
    elif in_pretrain:
        split = "pretrain"
    else:
        split = "unknown"

    return kind, split


def _get_robocasa_dataset_base() -> Path:
    """Return the configured robocasa dataset base path."""
    import robocasa

    robocasa_pkg = Path(robocasa.__file__).parent
    macros_private = robocasa_pkg / "macros_private.py"
    if macros_private.exists():
        spec = importlib.util.spec_from_file_location("_macros_private", macros_private)
        if spec is None or spec.loader is None:
            return robocasa_pkg.parent / "datasets"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        base = getattr(mod, "DATASET_BASE_PATH", None)
        if base:
            return Path(base)
    return robocasa_pkg.parent / "datasets"


# ---------------------------------------------------------------------------
# Download + convert
# ---------------------------------------------------------------------------


def _download_raw(task_name: str, split: str) -> Path:
    """Download the raw v2.1 dataset and return its lerobot/ path."""
    base = _get_robocasa_dataset_base()

    from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS

    all_tasks = {**ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS}
    meta = all_tasks[task_name]
    human_path = meta[split]["human_path"]
    raw_path = base / human_path / "lerobot"

    if raw_path.exists():
        print(f"Raw dataset already cached at: {raw_path}")
        return raw_path

    print(f"Downloading {task_name} ({split}) ...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robocasa.scripts.download_datasets",
            "--tasks",
            task_name,
            "--split",
            split,
            "--source",
            "human",
        ],
        input=b"y\n",
        check=True,
    )
    if not raw_path.exists():
        raise RuntimeError(f"Download completed but expected path not found: {raw_path}")
    return raw_path


def _convert_to_v30(raw_path: Path, repo_id: str) -> Path:
    """Convert v2.1 dataset at raw_path to v3.0 in-place. Returns the v3.0 path."""
    info_path = raw_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") == "v3.0":
        print(f"Dataset already at v3.0: {raw_path}")
        return raw_path

    print(f"Converting {raw_path} from v2.1 to v3.0 ...")

    import shutil

    tmp_root = Path(tempfile.mkdtemp(prefix="_robocasa_convert_"))

    parts = repo_id.split("/")
    fake_dataset_path = tmp_root / parts[-1]
    if fake_dataset_path.is_symlink():
        fake_dataset_path.unlink()
    fake_dataset_path.symlink_to(raw_path)

    subprocess.run(
        [
            sys.executable,
            "src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py",
            f"--repo-id={parts[-1]}",
            f"--root={tmp_root}",
            "--push-to-hub=false",
        ],
        check=True,
        cwd=Path(__file__).parent.parent,
    )

    v30_tmp = tmp_root / parts[-1]
    if v30_tmp.exists() and not v30_tmp.is_symlink():
        old_backup = raw_path.parent / (raw_path.name + "_old")
        if old_backup.exists():
            shutil.rmtree(str(old_backup))
        raw_path.rename(old_backup)
        shutil.move(str(v30_tmp), str(raw_path))
        print(f"Converted dataset at: {raw_path} (original backed up to {old_backup})")

    return raw_path


def _ensure_cache_link(v3_path: Path, split: str, task_name: str) -> str:
    """Symlink the v3.0 dataset into the local HF cache. Returns the repo_id."""
    lerobot_cache = Path("/home/droplab/.cache/huggingface/lerobot/local/robocasa")
    lerobot_cache.mkdir(parents=True, exist_ok=True)
    cache_name = f"{split}_{task_name}"
    cache_link = lerobot_cache / cache_name
    if not cache_link.exists():
        cache_link.symlink_to(v3_path)
        print(f"Linked v3.0 dataset to: {cache_link}")
    return f"local/robocasa/{cache_name}"


# ---------------------------------------------------------------------------
# Extras / layout-style metadata
# ---------------------------------------------------------------------------


def _find_extras_dir(raw_path: Path) -> Path | None:
    """Find extras/ directory inside or alongside the dataset."""
    for candidate in [
        raw_path / "extras",
        raw_path.parent / "extras",
        raw_path.parent / (raw_path.name + "_old") / "extras",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_episode_layout_style(extras_dir: Path) -> dict[int, dict]:
    """Return {episode_index: {"layout_id": int, "style_id": int}}."""
    result = {}
    for ep_dir in sorted(extras_dir.iterdir()):
        if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
            continue
        meta_file = ep_dir / "ep_meta.json"
        if not meta_file.exists():
            continue
        try:
            ep_idx = int(ep_dir.name.split("_")[1])
            with open(meta_file) as f:
                meta = json.load(f)
            result[ep_idx] = {
                "layout_id": meta.get("layout_id"),
                "style_id": meta.get("style_id"),
            }
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return result


def _print_distribution_table(ep_meta: dict[int, dict], total_label: str = "") -> None:
    """Print a layout × style episode count table."""
    from collections import Counter

    counts: Counter = Counter()
    for meta in ep_meta.values():
        counts[(meta["layout_id"], meta["style_id"])] += 1

    all_layouts = sorted({k[0] for k in counts})
    all_styles = sorted({k[1] for k in counts})

    col_w = 6
    lbl = "L\\S"
    header = f"{lbl:>5} " + " ".join(f"{s:>{col_w}}" for s in all_styles) + f"  {'total':>6}"
    title = f"Episode counts per (layout_id, style_id) — {len(ep_meta)} total"
    if total_label:
        title += f"  [{total_label}]"
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for layout in all_layouts:
        row_vals = [counts.get((layout, s), 0) for s in all_styles]
        row = f"{layout:>5} " + " ".join(f"{v:>{col_w}}" for v in row_vals) + f"  {sum(row_vals):>6}"
        print(row)
    col_totals = [sum(counts.get((layout_id, s), 0) for layout_id in all_layouts) for s in all_styles]
    print("-" * len(header))
    print(f"{'total':>5} " + " ".join(f"{v:>{col_w}}" for v in col_totals) + f"  {len(ep_meta):>6}")


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def mode_list(task_name: str, split: str) -> None:
    raw_path = _download_raw(task_name, split)
    extras_dir = _find_extras_dir(raw_path)
    if extras_dir is None:
        print(f"ERROR: Could not find extras/ directory near {raw_path}", file=sys.stderr)
        sys.exit(1)
    ep_meta = _load_episode_layout_style(extras_dir)
    if not ep_meta:
        print("ERROR: No ep_meta.json files found in extras directory.", file=sys.stderr)
        sys.exit(1)
    _print_distribution_table(ep_meta, total_label=f"{task_name} / {split}")


def mode_convert(task_name: str, split: str) -> None:
    raw_path = _download_raw(task_name, split)
    convert_repo_id = f"robocasa/{split}_{task_name}"
    v3_path = _convert_to_v30(raw_path, convert_repo_id)
    repo_id = _ensure_cache_link(v3_path, split, task_name)
    print(f"\nDataset available as: {repo_id}")
    _print_train_cmd(task_name, split, repo_id)


def mode_filter(
    task_name: str,
    split: str,
    layout_ids: list[int] | None,
    style_ids: list[int] | None,
    new_repo_id: str | None,
    dry_run: bool,
    no_encode_videos: bool,
) -> None:
    # Build output repo_id
    if new_repo_id:
        out_repo_id = new_repo_id
    else:
        suffix_parts = []
        if layout_ids is not None:
            suffix_parts.append("layout" + "_".join(str(i) for i in sorted(layout_ids)))
        if style_ids is not None:
            suffix_parts.append("style" + "_".join(str(i) for i in sorted(style_ids)))
        out_repo_id = f"local/robocasa/{split}_{task_name}_{'_'.join(suffix_parts)}"

    output_root = Path("/home/droplab/.cache/huggingface/lerobot/loca/robocasa")
    split_key = out_repo_id.split("/")[-1]
    save_path = output_root / split_key

    if save_path.exists() and any(save_path.iterdir()):
        print(f"Filtered dataset already exists at {save_path}")
        print(f"Output dataset: {out_repo_id}")
        return

    # Download raw
    raw_path = _download_raw(task_name, split)

    # Load extras metadata
    extras_dir = _find_extras_dir(raw_path)
    if extras_dir is None:
        print(f"ERROR: Could not find extras/ directory near {raw_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading layout/style metadata from: {extras_dir}")
    ep_meta = _load_episode_layout_style(extras_dir)
    if not ep_meta:
        print("ERROR: No ep_meta.json files found in extras directory.", file=sys.stderr)
        sys.exit(1)
    print(f"Found metadata for {len(ep_meta)} episodes.")

    # Filter
    layout_set = set(layout_ids) if layout_ids is not None else None
    style_set = set(style_ids) if style_ids is not None else None
    matched_episodes = [
        ep_idx
        for ep_idx in range(len(ep_meta))
        if (meta := ep_meta.get(ep_idx)) is not None
        and (layout_set is None or meta["layout_id"] in layout_set)
        and (style_set is None or meta["style_id"] in style_set)
    ]

    filter_desc = []
    if layout_set is not None:
        filter_desc.append(f"layout_ids={sorted(layout_set)}")
    if style_set is not None:
        filter_desc.append(f"style_ids={sorted(style_set)}")
    print(f"\nFilter: {', '.join(filter_desc)}")
    print(f"Matching episodes: {len(matched_episodes)} / {len(ep_meta)}")

    if not matched_episodes:
        print("No episodes match the given filter. Exiting.")
        sys.exit(0)

    if dry_run:
        _print_distribution_table({i: ep_meta[i] for i in matched_episodes}, total_label="dry-run match")
        print("\n[dry-run] No files written.")
        return

    # Convert to v3.0
    convert_repo_id = f"robocasa/{split}_{task_name}"
    v3_path = _convert_to_v30(raw_path, convert_repo_id)
    src_repo_id = _ensure_cache_link(v3_path, split, task_name)

    # Load and split
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print("\nLoading v3.0 dataset ...")
    dataset = LeRobotDataset(src_repo_id)
    print(f"Total episodes: {dataset.meta.total_episodes}")

    print(f"Splitting {len(matched_episodes)} episodes into: {save_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    result_datasets = split_dataset(
        dataset,
        splits={split_key: matched_episodes},
        output_dir=output_root,
    )
    new_ds = result_datasets[split_key]
    print(f"Done: {new_ds.meta.total_episodes} episodes, {new_ds.meta.total_frames} frames")

    if not no_encode_videos:
        sys.path.insert(0, str(Path(__file__).parent))
        from convert_images_to_videos import convert_dataset

        print("\nEncoding videos ...")
        convert_dataset(save_path)

    print(f"\nOutput dataset: {out_repo_id}")
    _print_train_cmd(task_name, split, out_repo_id)


def _print_train_cmd(task_name: str, split: str, repo_id: str) -> None:
    job_name = repo_id.split("/")[-1]
    print(f"""
Training command:

uv run --extra robocasa lerobot-train \\
  --policy.type=act \\
  --env.type=robocasa \\
  --env.task={task_name} \\
  --env.split={split} \\
  --dataset.repo_id={repo_id} \\
  --output_dir=outputs/train/{job_name} \\
  --job_name={job_name} \\
  --wandb.mode=offline \\
  --policy.push_to_hub=false \\
  --dataset.image_transforms.enable=true \\
  --policy.use_amp=true \\
  --batch_size=64 \\
  --num_workers=24 \\
  --policy.optimizer_lr=1e-4 \\
  --steps=200000 \\
  --save_freq=25000""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download, convert, and optionally filter a RoboCasa dataset to v3.0.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("task_name", help="RoboCasa task name (e.g. CoffeeServeMug)")
    parser.add_argument(
        "--split",
        default=None,
        choices=["target", "pretrain"],
        help="Which split to use when the task exists in both (default: target).",
    )

    # Mode flags
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print episode ↔ layout/style distribution table. No files are written.",
    )

    # Filter options (filter mode only)
    filter_group = parser.add_argument_group("filter mode options")
    filter_group.add_argument(
        "--layout-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Layout IDs to include (0-9). Activates filter mode.",
    )
    filter_group.add_argument(
        "--style-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Style IDs to include (0-10). Activates filter mode.",
    )
    filter_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show matching episode count and table without writing any files.",
    )
    filter_group.add_argument(
        "--new-repo-id",
        default=None,
        help="Override output repo ID (default: auto-generated from task/layout/style).",
    )
    filter_group.add_argument(
        "--no-encode-videos",
        action="store_true",
        help="Skip video encoding step after filtering.",
    )

    args = parser.parse_args()

    # Validate mutual exclusions
    has_filter = args.layout_ids is not None or args.style_ids is not None
    if args.list and has_filter:
        parser.error("--list cannot be combined with --layout-ids / --style-ids")
    if args.list and args.dry_run:
        parser.error("--list and --dry-run are mutually exclusive")
    if args.list and args.new_repo_id:
        parser.error("--list and --new-repo-id are mutually exclusive")

    # Resolve task info
    try:
        kind, split = _get_task_info(args.task_name)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    if split == "both":
        split = args.split or "target"
        print(f"Task: {args.task_name}  kind={kind}  split={split} (in both splits; use --split to override)")
    else:
        if args.split and args.split != split:
            parser.error(
                f"Task '{args.task_name}' only has split '{split}', but --split={args.split} was given"
            )
        print(f"Task: {args.task_name}  kind={kind}  split={split}")

    # Dispatch
    if args.list:
        mode_list(args.task_name, split)
    elif has_filter:
        mode_filter(
            task_name=args.task_name,
            split=split,
            layout_ids=args.layout_ids,
            style_ids=args.style_ids,
            new_repo_id=args.new_repo_id,
            dry_run=args.dry_run,
            no_encode_videos=args.no_encode_videos,
        )
    else:
        if args.dry_run:
            parser.error("--dry-run only applies in filter mode (use --layout-ids or --style-ids)")
        if args.no_encode_videos:
            parser.error("--no-encode-videos only applies in filter mode")
        mode_convert(args.task_name, split)


if __name__ == "__main__":
    main()
