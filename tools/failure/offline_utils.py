import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.failure_handling.config import FailureConfig
from lerobot.policies.failure_handling.metrics import FailureMetrics


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


def _iter_failure_handling_candidates(pretrained_path: Path):
    direct = [
        pretrained_path / "failure_handling.json",
        pretrained_path / "pretrained_model" / "failure_handling.json",
    ]
    yield from direct

    search_roots = [pretrained_path]
    if pretrained_path.name == "pretrained_model":
        search_roots.extend([pretrained_path.parent, pretrained_path.parent.parent])
    elif pretrained_path.name == "last":
        search_roots.append(pretrained_path.parent)

    seen: set[Path] = set()

    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        if not root.exists() or not root.is_dir():
            continue

        for pattern in (
            "outputs/**/checkpoints/last/pretrained_model/failure_handling.json",
            "**/checkpoints/last/pretrained_model/failure_handling.json",
        ):
            yield from root.glob(pattern)


def resolve_failure_handling_json_path(
    dataset_root: str | Path, raise_on_missing: bool = False
) -> Path | None:
    """Resolve the path to failure_handling.json.

    Args:
        dataset_root: Root directory of the dataset
        raise_on_missing: If True, raise errors when config files are not found

    Returns:
        Path to failure_handling.json, or None if not found and raise_on_missing=False

    Raises:
        FileNotFoundError: If raise_on_missing=True and files are not found
        RuntimeError: If raise_on_missing=True and parsing fails
        ValueError: If raise_on_missing=True and pretrained_path is missing
    """
    dataset_root = Path(dataset_root).expanduser()
    record_config_path = dataset_root / "meta" / "record_config.json"
    if not record_config_path.exists():
        if raise_on_missing:
            raise FileNotFoundError(
                f"ERROR: record_config.json not found at {record_config_path}. "
                "Cannot load checkpoint signal config."
            )
        return None

    try:
        with open(record_config_path) as file:
            record_config = json.load(file)
    except Exception as e:
        if raise_on_missing:
            raise RuntimeError(
                f"ERROR: Failed to parse record_config.json at {record_config_path}: {e}"
            ) from e
        return None

    pretrained_path = _extract_pretrained_path(record_config)
    if not pretrained_path:
        if raise_on_missing:
            raise ValueError(f"ERROR: 'pretrained_path' not found in {record_config_path}")
        return None

    candidates = list(_iter_failure_handling_candidates(pretrained_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if raise_on_missing:
        tried_paths = "\n".join(f"  - {c}" for c in candidates[:5])
        raise FileNotFoundError(
            f"ERROR: failure_handling.json not found. Tried:\n{tried_paths}\n"
            "This file is required to visualize checkpoint detection and failure metrics."
        )
    return None


def load_failure_handling_json(dataset_root: str | Path, required: bool = False) -> dict[str, Any]:
    """Load failure_handling.json from dataset.

    Args:
        dataset_root: Root directory of the dataset
        required: If True, raise errors when config file is not found or invalid

    Returns:
        Dictionary with failure handling configuration

    Raises:
        FileNotFoundError/RuntimeError/ValueError: If required=True and loading fails
    """
    cfg_path = resolve_failure_handling_json_path(dataset_root, raise_on_missing=required)
    if cfg_path is None:
        if required:
            raise FileNotFoundError(
                f"ERROR: Could not resolve failure_handling.json path for dataset {dataset_root}"
            )
        return {}

    try:
        with open(cfg_path) as file:
            cfg = json.load(file)
            if required:
                print(f"Loaded failure handling config from {cfg_path}")
            return cfg
    except Exception as e:
        if required:
            raise RuntimeError(f"ERROR: Failed to parse failure_handling.json at {cfg_path}: {e}") from e
        return {}


def load_failure_config(dataset_root: str | Path, required: bool = False) -> FailureConfig:
    """Load FailureConfig from dataset.

    Args:
        dataset_root: Root directory of the dataset
        required: If True, raise errors when config file is not found

    Returns:
        FailureConfig instance

    Raises:
        FileNotFoundError/RuntimeError/ValueError: If required=True and loading fails
    """
    cfg_path = resolve_failure_handling_json_path(dataset_root, raise_on_missing=required)
    if cfg_path is None and required:
        raise FileNotFoundError(
            f"ERROR: Could not resolve failure_handling.json path for dataset {dataset_root}"
        )
    return FailureConfig.from_json(cfg_path)


def load_failure_metrics_jsonl(dataset_root: str | Path) -> dict[int, dict[str, Any]]:
    metrics_path = Path(dataset_root) / "failure_metrics.jsonl"
    if not metrics_path.exists():
        return {}

    failure_metrics: dict[int, dict[str, Any]] = {}
    with open(metrics_path) as file:
        for line in file:
            try:
                row = json.loads(line)
                if "step" in row:
                    failure_metrics[int(row["step"])] = row
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                continue
    return failure_metrics


def _checkpoint_metric_from_row(row: dict[str, Any], source: str) -> float:
    source = str(source).strip().lower()

    key_map = {
        "temporal_disagreement": "temporal_disagreement",
        "following_error": "following_error",
        "attention_entropy": "attention_entropy",
        "mahalanobis_distance": "mahalanobis_distance",
        "endpoint_shift": "endpoint_shift",
        "action_jerk": "action_jerk",
        "action_entropy": "action_entropy",
    }

    if source == "action_entropy_max_diff":
        for key in (
            "action_entropy_max_diff",
            "action_entropy_sample_max_diff",
            "action_sample_max_diff",
        ):
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        fallback = row.get("temporal_disagreement", 0.0)
        return float(fallback) if isinstance(fallback, (int, float)) else 0.0

    metric_key = key_map.get(source)
    if metric_key is None:
        metric_key = "temporal_disagreement"

    value = row.get(metric_key)
    if isinstance(value, (int, float)):
        return float(value)

    fallback = row.get("temporal_disagreement", 0.0)
    return float(fallback) if isinstance(fallback, (int, float)) else 0.0


def replay_checkpoint_series(
    failure_metrics: dict[int, dict[str, Any]],
    config: FailureConfig,
    *,
    safety_margin: int = 40,
    dataset_episodes: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, list[int]], dict[int, bool]]:
    if not failure_metrics:
        return {}, {}, {}, {}, {}

    replay_cfg = deepcopy(config)
    replay_cfg.enable_failure_handling = True
    metrics_engine = FailureMetrics(config=replay_cfg, output_dir=None)
    checkpoint_metric_source = str(replay_cfg.checkpoint_metric_source).strip().lower()

    steps = sorted(failure_metrics.keys())
    smoothed_by_step: dict[int, float] = {}
    previous_checkpoint_by_step: dict[int, float] = {}
    checkpoint_flag_by_step: dict[int, float] = {}
    recent_checkpoints_by_step: dict[int, list[int]] = {}
    detect_failure_by_step: dict[int, bool] = {}

    checkpoint_set: set[int] = set()
    checkpoint_history: list[int] = []
    checkpoint_set_in_episode: set[int] = set()

    episode_bounds: list[tuple[int, int]] = []
    if dataset_episodes is not None:
        for ep_meta in dataset_episodes:
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
            episode_bounds.append((from_idx, to_idx))
        episode_bounds.sort(key=lambda x: x[0])

    current_episode_idx = 0

    for step in steps:
        if episode_bounds:
            while (
                current_episode_idx < len(episode_bounds)
                and int(step) >= episode_bounds[current_episode_idx][1]
            ):
                current_episode_idx += 1
                metrics_engine = FailureMetrics(config=replay_cfg, output_dir=None)
                checkpoint_history = []
                checkpoint_set_in_episode = set()

        if bool(failure_metrics[step].get("is_recovery_wait", False)):
            detect_failure_by_step[int(step)] = False
            smoothed_by_step[int(step)] = float(metrics_engine.latest_smoothed_disagreement)

            safe_checkpoint = float("nan")
            for cp in reversed(checkpoint_history):
                if int(step) - cp >= safety_margin:
                    safe_checkpoint = float(cp)
                    break

            previous_checkpoint_by_step[int(step)] = safe_checkpoint
            recent_checkpoints_by_step[int(step)] = checkpoint_history[-5:]
            continue

        row = failure_metrics[step]
        disagreement = float(row.get("temporal_disagreement", 0.0))
        checkpoint_metric = _checkpoint_metric_from_row(row, checkpoint_metric_source)

        metrics_engine.process_step = int(step)
        metrics_engine.latest_temporal_disagreement = disagreement
        metrics_engine.append_state(
            torch.zeros(1, dtype=torch.float32),
            batch=None,
            checkpoint_metric=checkpoint_metric,
        )
        detect_failure_by_step[int(step)] = bool(metrics_engine.detect_failure())

        smoothed_by_step[int(step)] = float(metrics_engine.latest_smoothed_disagreement)

        queue_steps = [int(cp_step) for cp_step, _, _ in metrics_engine.checkpoint_action_queue]
        for cp_step in queue_steps:
            checkpoint_set.add(cp_step)
            if cp_step not in checkpoint_set_in_episode:
                checkpoint_set_in_episode.add(cp_step)
                checkpoint_history.append(cp_step)

        safe_checkpoint = float("nan")
        for cp in reversed(checkpoint_history):
            if int(step) - cp >= safety_margin:
                safe_checkpoint = float(cp)
                break

        previous_checkpoint_by_step[int(step)] = safe_checkpoint
        recent_checkpoints_by_step[int(step)] = checkpoint_history[-5:]

    for step in steps:
        checkpoint_flag_by_step[int(step)] = 1.0 if int(step) in checkpoint_set else 0.0

    return (
        smoothed_by_step,
        previous_checkpoint_by_step,
        checkpoint_flag_by_step,
        recent_checkpoints_by_step,
        detect_failure_by_step,
    )
