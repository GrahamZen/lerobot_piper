#!/usr/bin/env python
"""Run the Gemini VLM pipeline on a random sample of training episodes and cache results.

Reads train_config.json from --pretrained_path to locate the training dataset automatically
(same logic as failure_config_gen.py).  After this script completes,
failure_config_gen.py will find vlm_checkpoints.json in pretrained_path and skip VLM entirely.

Cache behaviour:
  - No cache         : sample num_episodes, run VLM on all, save.
  - Cache < target   : sample (target - cached) more episodes from the remainder, run VLM,
                       merge with existing cache, save.
  - Cache >= target  : nothing to do, exit early.
  - --force_vlm      : discard cache, sample fresh, run VLM on all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from vlm_pipeline import (
    load_vlm_checkpoints,
    run_vlm_pipeline,
    save_vlm_checkpoints,
)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample training episodes, run VLM checkpoint detection, and cache results to "
            "pretrained_path/vlm_checkpoints.json for use by failure_config_gen.py."
        )
    )
    parser.add_argument(
        "--pretrained_path",
        type=Path,
        required=True,
        help="Model folder containing train_config.json. vlm_checkpoints.json will be saved here.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        required=True,
        help="Target number of episodes to have VLM results for.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode sampling. Default: 42",
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("~/.cache/huggingface/lerobot").expanduser(),
        help="Lerobot cache root used when train_config.json has no explicit root. "
        "Default: ~/.cache/huggingface/lerobot",
    )
    parser.add_argument(
        "--force_vlm",
        action="store_true",
        help="Discard existing vlm_checkpoints.json and re-run from scratch.",
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key. Falls back to GEMINI_API_KEY env var.",
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-pro-preview-03-25",
        help="Gemini model name for VLM analysis.",
    )
    parser.add_argument(
        "--vlm_workers",
        type=int,
        default=1,
        help="Number of concurrent VLM workers. Default: 1",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training dataset discovery (identical to failure_config_gen.py)
# ---------------------------------------------------------------------------


def resolve_training_dataset_root(pretrained_path: Path, cache_root: Path) -> tuple[Path, str]:
    train_config_path = pretrained_path / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"train_config.json not found at {train_config_path}")
    with train_config_path.open() as f:
        train_config = json.load(f)
    training_repo_id: str = train_config["dataset"]["repo_id"]
    root = train_config["dataset"].get("root")
    dataset_root = Path(root).expanduser() if root else cache_root / training_repo_id
    return dataset_root, training_repo_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    pretrained_path = args.pretrained_path.expanduser().resolve()

    # ---- Locate training dataset ----
    print(f"[Setup] Resolving training dataset from {pretrained_path / 'train_config.json'}...")
    training_dataset_root, training_repo_id = resolve_training_dataset_root(pretrained_path, args.cache_root)
    print(f"[Setup] training_repo_id : {training_repo_id}")
    print(f"[Setup] training_dataset_root: {training_dataset_root}")

    training_dataset = LeRobotDataset(training_repo_id, root=training_dataset_root)
    if training_dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        training_dataset.meta.episodes = load_episodes(training_dataset.root)

    ep_parquet_dir = training_dataset_root / "meta" / "episodes"
    ep_df = pd.concat([pd.read_parquet(p) for p in sorted(ep_parquet_dir.glob("chunk-*"))])
    ep_df = ep_df.sort_values("episode_index").reset_index(drop=True)
    all_ep_ids = ep_df["episode_index"].tolist()
    print(f"[Setup] Training dataset: {len(all_ep_ids)} episodes total")

    if args.num_episodes > len(all_ep_ids):
        raise ValueError(f"--num_episodes {args.num_episodes} exceeds total episodes {len(all_ep_ids)}")

    # ---- Load existing cache ----
    cached: dict[int, list[int]] = {}
    if not args.force_vlm:
        loaded = load_vlm_checkpoints(pretrained_path)
        if loaded is not None:
            cached = loaded
            print(f"[Cache] Loaded {len(cached)} episodes from existing vlm_checkpoints.json.")

    if args.force_vlm and (pretrained_path / "vlm_checkpoints.json").exists():
        print("[Cache] --force_vlm set, discarding existing cache.")

    # ---- Determine which episodes still need VLM ----
    already_done = set(cached.keys())
    need = args.num_episodes - len(already_done)

    if need <= 0:
        print(
            f"[Cache] Already have {len(already_done)} episodes (>= target {args.num_episodes}). "
            "Nothing to do."
        )
        return

    remaining_ep_ids = [ep for ep in all_ep_ids if ep not in already_done]
    if len(remaining_ep_ids) < need:
        raise ValueError(
            f"Need {need} more episodes but only {len(remaining_ep_ids)} remain without VLM results."
        )

    rng = np.random.default_rng(args.seed)
    new_ep_ids = sorted(rng.choice(remaining_ep_ids, size=need, replace=False).tolist())
    print(f"[Sample] Sampling {need} new episodes (seed={args.seed}): {new_ep_ids}")

    sampled_ep_df = ep_df[ep_df["episode_index"].isin(new_ep_ids)].reset_index(drop=True)

    # ---- Run VLM pipeline on new episodes only ----
    print(f"\n[VLM] Running VLM pipeline on {len(new_ep_ids)} episodes...")
    new_results = run_vlm_pipeline(
        training_dataset=training_dataset,
        training_dataset_root=training_dataset_root,
        ep_df=sampled_ep_df,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        vlm_workers=args.vlm_workers,
    )

    # ---- Merge and save ----
    merged = {**cached, **new_results}
    save_vlm_checkpoints(pretrained_path, merged)
    print(
        f"\n[Done] vlm_checkpoints.json now contains {len(merged)} episodes "
        f"({len(cached)} from cache + {len(new_results)} new)."
    )


if __name__ == "__main__":
    main()
