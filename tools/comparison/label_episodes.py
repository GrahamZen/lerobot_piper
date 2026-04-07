"""Interactive episode labeling tool.

Labels each episode in a dataset as success (1) or failure (0) via user input,
and saves the result to:
    {root}/{repo_id}/comparison/episode_labels.json

When batch_runs.json exists in the same directory, iterates over every entry
and labels each dataset in sequence.  Pass --repo_id to label a single dataset.

Usage::

    # Batch mode (reads batch_runs.json)
    python tools/comparison/label_episodes.py

    # Batch mode with Rerun visualisation before each labelling prompt
    python tools/comparison/label_episodes.py --display_data

    # Single dataset
    python tools/comparison/label_episodes.py --repo_id eval/eval_pick_and_place_act
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "failure"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Rerun visualisation
# ---------------------------------------------------------------------------


def _load_failure_threshold(fh_json: Path) -> float | None:
    try:
        with fh_json.open() as f:
            fh = json.load(f)
        det = fh.get("detector") or fh
        v = float(det.get("failure_threshold", det.get("threshold", 0.0)))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return v or None


def _visualize_dataset(dataset: LeRobotDataset, dataset_root: Path, stride: int = 5) -> None:
    """Stream TD curve, failure markers, threshold, and camera frames to Rerun."""
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    from offline_utils import _extract_pretrained_path, get_episode_bounds, load_failure_metrics_jsonl
    from torch.utils.data import DataLoader, Subset

    # --- Load failure metrics ---
    recorded_metrics = load_failure_metrics_jsonl(dataset_root)
    if not recorded_metrics:
        print(f"[WARN] No failure_metrics.jsonl found in {dataset_root} — skipping visualisation.")
        return
    print(f"[Rerun] Loaded {len(recorded_metrics)} metric rows.")

    # --- Resolve failure threshold ---
    failure_threshold: float | None = None
    record_cfg_path = dataset_root / "meta" / "record_config.json"
    if record_cfg_path.exists():
        with record_cfg_path.open() as f:
            record_cfg = json.load(f)
        pretrained_path = _extract_pretrained_path(record_cfg)
        if pretrained_path:
            fh_json = pretrained_path / "failure_handling.json"
            if fh_json.exists():
                failure_threshold = _load_failure_threshold(fh_json)
    if failure_threshold is not None:
        print(f"[Rerun] failure_threshold={failure_threshold:.6f}")
    else:
        print("[Rerun] No failure_threshold found — threshold line will be omitted.")

    # --- Blueprint ---
    camera_keys = dataset.meta.camera_keys
    ts_contents = ["metrics/td_smoothed/**"]
    if failure_threshold is not None:
        ts_contents.append("metrics/failure_threshold")

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.TimeSeriesView(name="TD Smoothed", contents=ts_contents),
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TextDocumentView(name="Episode", origin="overlay/episode_id"),
                    rrb.Spatial2DView(name="Camera", origin="cameras/stitched"),
                    row_shares=[1, 8],
                ),
            ),
            row_shares=[1, 1.5],
        ),
        collapse_panels=True,
    )

    rr.init(f"Label: {dataset.repo_id}", spawn=True)
    rr.send_blueprint(blueprint)
    rr.log(
        "metrics/td_smoothed/failed_markers",
        rr.SeriesPoints(colors=[255, 0, 0], markers="diamond", marker_sizes=5.0),
        static=True,
    )
    rr.log(
        "metrics/td_smoothed/score",
        rr.SeriesLines(colors=[52, 152, 219], names="TD smoothed"),
        static=True,
    )
    if failure_threshold is not None:
        rr.log(
            "metrics/failure_threshold",
            rr.SeriesLines(colors=[255, 165, 0], names="threshold"),
            static=True,
        )

    # --- Build frame list ---
    n_eps = len(dataset.meta.episodes)
    all_frames: list[tuple[int, int]] = []
    episode_from_idx: dict[int, int] = {}
    for ep_idx in range(n_eps):
        from_idx, to_idx = get_episode_bounds(dataset, ep_idx)
        episode_from_idx[ep_idx] = from_idx
        for frame_idx in range(from_idx, to_idx, stride):
            all_frames.append((ep_idx, frame_idx))

    loader = DataLoader(
        Subset(dataset, [f for _, f in all_frames]),
        batch_size=1,
        num_workers=4,
        prefetch_factor=2,
        collate_fn=lambda b: b[0],
        shuffle=False,
    )

    # --- Stream ---
    from tqdm import tqdm

    current_ep = -1
    for global_step, ((ep_idx, frame_idx), item) in enumerate(
        tqdm(zip(all_frames, loader, strict=True), total=len(all_frames), desc="[Rerun] Streaming")
    ):
        if ep_idx != current_ep:
            current_ep = ep_idx

        rr.set_time("step", sequence=frame_idx)
        rr.set_time("global_step", sequence=global_step)
        rr.log("overlay/episode_id", rr.TextDocument(f"Episode {ep_idx}"), static=False)

        # Camera
        cam_imgs: list[np.ndarray] = []
        for key in camera_keys:
            img = item.get(key)
            if img is None:
                continue
            arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] <= 4:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            cam_imgs.append(arr)
        if cam_imgs:
            rr.log("cameras/stitched", rr.Image(np.concatenate(cam_imgs, axis=1)))

        # TD metrics
        row = recorded_metrics.get(frame_idx)
        if row:
            td_smoothed = float(row.get("td_smoothed", 0.0))
            rr.log("metrics/td_smoothed/score", rr.Scalars(td_smoothed))
            if failure_threshold is not None:
                rr.log("metrics/failure_threshold", rr.Scalars(failure_threshold))
                if td_smoothed > failure_threshold:
                    rr.log("metrics/td_smoothed/failed_markers", rr.Scalars(td_smoothed))

    print("[Rerun] Done streaming.")


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def _auto_label(dataset_root: Path, n_episodes: int) -> set[int] | None:
    """Detect failed episodes by checking if any step's td_smoothed exceeds the threshold.

    Returns a set of failed episode indices, or None if auto-labelling is not possible
    (missing failure_metrics.jsonl or failure_handling.json).
    """
    from offline_utils import _extract_pretrained_path, load_failure_metrics_jsonl

    recorded_metrics = load_failure_metrics_jsonl(dataset_root)
    if not recorded_metrics:
        print("[Auto] No failure_metrics.jsonl found — cannot auto-label.")
        return None

    # Resolve threshold
    failure_threshold: float | None = None
    record_cfg_path = dataset_root / "meta" / "record_config.json"
    if record_cfg_path.exists():
        with record_cfg_path.open() as f:
            record_cfg = json.load(f)
        pretrained_path = _extract_pretrained_path(record_cfg)
        if pretrained_path:
            fh_json = pretrained_path / "failure_handling.json"
            if fh_json.exists():
                failure_threshold = _load_failure_threshold(fh_json)

    if failure_threshold is None:
        print("[Auto] No failure_threshold found in failure_handling.json — cannot auto-label.")
        return None

    print(f"[Auto] threshold={failure_threshold:.6f}")

    # Group td_smoothed by episode.
    # failure_metrics.jsonl uses 1-based episode indices (recorder starts at 0
    # but reset_episode() is called once before the first episode, making it 1).
    # episode_labels.json uses 0-based indices, so subtract 1 here.
    ep_max_td: dict[int, float] = {}
    for row in recorded_metrics.values():
        ep = int(row.get("episode", 0)) - 1  # convert 1-based → 0-based
        if ep < 0:
            continue
        td = row.get("td_smoothed")
        if td is not None:
            ep_max_td[ep] = max(ep_max_td.get(ep, 0.0), float(td))

    failed = {ep for ep, max_td in ep_max_td.items() if max_td > failure_threshold}
    print(f"[Auto] {len(failed)} failure(s) detected out of {n_episodes} episodes: {sorted(failed)}")
    return failed


def label_one(
    repo_id: str,
    root: Path,
    display_data: bool = False,
    stride: int = 5,
    auto: bool = False,
    check: bool = False,
    force: bool = False,
) -> None:
    dataset = LeRobotDataset(repo_id)
    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)

    n_episodes = dataset.num_episodes
    print(f"\nDataset: {repo_id}  ({n_episodes} episodes)")

    output_path = root / repo_id / "comparison" / "episode_labels.json"

    if check:
        # Show current labels and let the user optionally overwrite
        if output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)
            current_failed = sorted(int(k) for k, v in existing.items() if v == 0)
            print(f"  Current failed episodes: {current_failed}")
        else:
            existing = {}
            print("  No labels yet.")

        if display_data:
            _visualize_dataset(dataset, Path(dataset.root), stride=stride)

        raw = input("Enter new failed indices to overwrite, or press Enter to skip: ").strip()
        if not raw:
            print("  Skipped.")
            return

        try:
            checked_failed = set(json.loads(raw))
            if not all(isinstance(x, int) and 0 <= x < n_episodes for x in checked_failed):
                print("  Invalid indices — skipping.")
                return
        except (json.JSONDecodeError, TypeError):
            print("  Invalid format — skipping.")
            return

        labels = {str(i): (0 if i in checked_failed else 1) for i in range(n_episodes)}
        print(f"  {len(checked_failed)} failures, {n_episodes - len(checked_failed)} successes.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labels, f, indent=4)
        print(f"Saved {len(labels)} labels to {output_path}")
        return

    # Skip if already fully labeled (unless --auto --force)
    if output_path.exists():
        with open(output_path) as f:
            labels = json.load(f)
        if len(labels) == n_episodes:
            if auto and force:
                print("Already labeled — re-generating with auto (--force).")
            else:
                print(f"Already labeled ({len(labels)} episodes) — skipping.")
                return
        else:
            print(f"Resuming from {output_path} ({len(labels)}/{n_episodes} labeled)")
    else:
        labels = {}

    if display_data:
        _visualize_dataset(dataset, Path(dataset.root), stride=stride)

    failed: set[int] | None = None

    if auto:
        failed = _auto_label(Path(dataset.root), n_episodes)
        if failed is None:
            print("[Auto] Falling back to manual input.")
            auto = False

    if not auto:
        while True:
            raw = input(f"Enter failed episode indices (0-{n_episodes - 1}), e.g. [2,7,12,16]: ").strip()
            try:
                parsed_failed = json.loads(raw)
                if not isinstance(parsed_failed, list):
                    print("  Invalid format. Please enter a JSON list like [2,7,12,16].")
                    continue
                if not all(isinstance(x, int) and 0 <= x < n_episodes for x in parsed_failed):
                    print(f"  All indices must be integers in range [0, {n_episodes - 1}].")
                    continue
                failed = set(parsed_failed)
                break
            except (json.JSONDecodeError, TypeError):
                print("  Invalid format. Please enter a JSON list like [2,7,12,16].")

    assert failed is not None

    for i in range(n_episodes):
        labels[str(i)] = 0 if i in failed else 1

    print(f"  {len(failed)} failures, {n_episodes - len(failed)} successes.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labels, f, indent=4)
    print(f"Saved {len(labels)} labels to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Label episodes as success/failure.")
    parser.add_argument(
        "--repo_id", type=str, default=None, help="Dataset repo id. Omit to use batch_runs.json."
    )
    parser.add_argument(
        "--root", type=str, default="~/.cache/huggingface/lerobot", help="Dataset root directory"
    )
    parser.add_argument(
        "--display_data",
        action="store_true",
        help="Stream each dataset to Rerun (TD curve + failure markers + camera) before prompting for labels.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Frame stride for Rerun streaming (default: 5). Higher = faster but less smooth.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-label episodes by checking if any step's td_smoothed exceeds the threshold. "
        "Falls back to manual input if failure_metrics.jsonl or threshold is unavailable.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Review and optionally overwrite existing labels. Shows current failed episodes, "
        "then prompts for new indices (Enter to skip).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --auto: overwrite already-labeled datasets instead of skipping.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    batch_file = Path(__file__).parent / "batch_runs.json"

    kwargs = {
        "display_data": args.display_data,
        "stride": args.stride,
        "auto": args.auto,
        "check": args.check,
        "force": args.force,
    }

    if args.repo_id:
        label_one(args.repo_id, root, **kwargs)
    elif batch_file.exists():
        with open(batch_file) as f:
            entries = json.load(f)
        print(f"Batch mode: {len(entries)} dataset(s) from {batch_file}")
        for i, entry in enumerate(entries):
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(entries)}]")
            label_one(entry["repo_id"], root, **kwargs)
        print(f"\n{'=' * 60}")
        print("All datasets labeled.")
    else:
        parser.error("--repo_id is required when batch_runs.json is not present.")


if __name__ == "__main__":
    main()
