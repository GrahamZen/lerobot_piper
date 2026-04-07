"""Base interface for offline failure-detection detectors."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


class BaseDetector(ABC):
    """Stateful, episode-scoped anomaly detector.

    Lifecycle per episode::

        detector.reset()
        for step, chunks in episode:
            score = detector.update(step, chunks)
        # save / collect scores externally

    ``chunks`` is the batch of action-chunk samples from the policy at the
    current step, shape ``(B, horizon, action_dim)``.  For deterministic
    policies (ACT without stochastic latent) ``B`` will be 1.
    """

    #: Unique identifier used for file naming and Rerun path components.
    name: str

    def reset(self) -> None:  # noqa: B027
        """Reset all episode-local state.  Call at the start of every episode."""

    def calibration_score(self) -> float:
        """Episode-level score used for threshold calibration.

        Called once at the end of each calibration episode (after all
        ``update`` calls).  The default returns the last ``update`` output;
        subclasses should override to return a more meaningful episode summary
        (e.g. cumulative total or episode maximum).
        """
        return 0.0

    @abstractmethod
    def update(self, step: int, chunks: Tensor | None) -> float:
        """Ingest action chunks for *step* and return an anomaly score.

        Args:
            step: Episode-local step index (0-based).
            chunks: Policy samples, shape ``(B, horizon, action_dim)``.
                    May be ``None`` on steps where no new chunk is produced
                    (e.g. mid-horizon steps for methods that only re-predict
                    every *k* steps).

        Returns:
            Non-negative anomaly score.  Higher → more anomalous.
        """
        ...

    # ------------------------------------------------------------------
    # Persistence helpers (shared, override if needed)
    # ------------------------------------------------------------------

    @staticmethod
    def save_scores(scores: list[dict[str, Any]], path: Path) -> None:
        """Write a list of score-row dicts to a JSONL file.

        Each row must contain at least ``global_step``, ``episode``,
        ``step_in_episode``, and ``score``.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in scores:
                f.write(json.dumps(row) + "\n")

    @staticmethod
    def load_scores(path: Path) -> dict[int, float]:
        """Load a JSONL score file into a ``{global_step: score}`` dict."""
        if not path.exists():
            return {}
        out: dict[int, float] = {}
        with path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                    gs = row.get("global_step")
                    sc = row.get("score")
                    if gs is not None and sc is not None:
                        out[int(gs)] = float(sc)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return out


def make_policy_batch(
    item: dict[str, Any],
    device: torch.device | str,
    n_samples: int = 1,
) -> dict[str, Tensor]:
    """Convert a single dataset *item* into a policy-ready batch.

    Adds a batch dimension and repeats ``n_samples`` times so that a
    stochastic policy (e.g. diffusion with random latent) produces
    ``n_samples`` diverse action chunks in a single forward pass.

    Only observation keys (those starting with ``"observation."``) and
    ``"observation_state"`` are included; action / metadata keys are dropped.

    Args:
        item: A single sample from ``LeRobotDataset``.
        device: Target device for all tensors.
        n_samples: How many times to repeat the observation along dim 0.

    Returns:
        Dict of tensors with shape ``(n_samples, ...)``.
    """
    batch: dict[str, Tensor] = {}
    for key, val in item.items():
        if not (key.startswith("observation.") or key == "observation_state"):
            continue
        if not isinstance(val, Tensor):
            val = torch.as_tensor(val)
        # dataset item has no batch dim → add one, then repeat
        val = val.unsqueeze(0).to(device, non_blocking=True)
        if n_samples > 1:
            val = val.expand(n_samples, *val.shape[1:]).contiguous()
        batch[key] = val
    return batch
