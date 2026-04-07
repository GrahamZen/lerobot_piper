"""Offline utilities for loading and replaying failure-handling data.

Public API (signature-stable):
    get_episode_bounds(dataset, episode_index) -> (from_idx, to_idx)
    load_failure_metrics_jsonl(dataset_root) -> dict[int, dict]
    _extract_pretrained_path(record_config) -> Path | None
"""

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Episode bounds helpers
# ---------------------------------------------------------------------------


def get_episode_bounds(dataset, episode_index: int) -> tuple[int, int]:
    """Return (from_idx, to_idx) for a given episode, loading episode metadata if needed."""
    if dataset.meta.episodes is None:
        from lerobot.datasets.utils import load_episodes

        dataset.meta.episodes = load_episodes(dataset.root)
    ep_meta = dataset.meta.episodes[episode_index]
    from_idx = int(
        ep_meta["dataset_from_index"][0]
        if isinstance(ep_meta["dataset_from_index"], list)
        else ep_meta["dataset_from_index"]
    )
    to_idx = int(
        ep_meta["dataset_to_index"][0]
        if isinstance(ep_meta["dataset_to_index"], list)
        else ep_meta["dataset_to_index"]
    )
    return from_idx, to_idx


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def _extract_pretrained_path(record_config: dict[str, Any]) -> Path | None:
    pretrained_path = record_config.get("pretrained_path")
    if not pretrained_path and isinstance(record_config.get("policy"), dict):
        pretrained_path = record_config["policy"].get("pretrained_path")

    if not pretrained_path:
        for key in ("model", "train", "training"):
            section = record_config.get(key)
            if isinstance(section, dict) and section.get("pretrained_path"):
                pretrained_path = section["pretrained_path"]
                break

    if not pretrained_path:
        return None

    return Path(pretrained_path).expanduser()


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_failure_metrics_jsonl(dataset_root: str | Path) -> dict[int, dict[str, Any]]:
    """Load failure_metrics.jsonl into a dict keyed by ``global_step``.

    Returns an empty dict if the file does not exist.
    """
    metrics_path = Path(dataset_root) / "failure_metrics.jsonl"
    if not metrics_path.exists():
        return {}

    failure_metrics: dict[int, dict[str, Any]] = {}
    with metrics_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                # Support both "global_step" (new) and "step" (old) as the key
                key = row.get("global_step", row.get("step"))
                if key is not None:
                    failure_metrics[int(key)] = row
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
    return failure_metrics
