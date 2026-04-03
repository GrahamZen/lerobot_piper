"""Fixed temporal-disagreement failure detector.

This module is intentionally non-extendable: the detection mechanism
(TD → causal Gaussian smoothing → threshold comparison) is locked.
"""

import collections
import logging

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.failure_handling.config import DetectorConfig

logger = logging.getLogger(__name__)


def _causal_gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """One-sided (causal) Gaussian smoothing — no future leakage.

    Args:
        values: 1-D array of raw metric values (oldest → newest).
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Smoothed array of the same length.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=np.float64)
    if sigma <= 0:
        return np.asarray(values, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    for t in range(n):
        w = min(t + 1, len(kernel))
        k = kernel[:w]
        k_norm = k / k.sum()
        out[t] = np.dot(k_norm, values[t - w + 1 : t + 1][::-1])

    return out


class FailureDetector:
    """Compute temporal disagreement, smooth it, and detect failures.

    Attributes:
        td_raw:      Raw (unsmoothed) TD at the current step. Useful for visualization.
        td_smoothed: Causal-Gaussian-smoothed TD at the current step.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._td_history: collections.deque[float] = collections.deque(maxlen=max(1, config.td_window_size))
        self._weights_cache: dict = {}

        self.td_raw: float = 0.0
        self.td_smoothed: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_raw_td(self, new_chunk: Tensor, policy) -> float:
        """Compute raw temporal disagreement between the old ensembled plan and *new_chunk*.

        Returns 0.0 when:
        - The policy has no ``temporal_ensembler`` attribute.
        - The ensembler has not yet accumulated any actions.
        - *new_chunk* is shorter than the current plan.
        """
        if policy is None:
            return 0.0

        ensembler = getattr(policy, "temporal_ensembler", None)
        if ensembler is None or ensembler.ensembled_actions is None:
            return 0.0

        old_plan: Tensor = ensembler.ensembled_actions  # (batch, T, action_dim)
        overlap = old_plan.shape[1]
        if overlap == 0 or new_chunk.shape[1] < overlap:
            return 0.0

        new_plan = new_chunk[:, :overlap]
        diff_sq = (old_plan - new_plan) ** 2  # (batch, T, action_dim)

        rho = self._config.td_rho
        if rho == 1.0:
            return float(diff_sq.mean())

        t = diff_sq.shape[1]
        cache_key = (t, rho, diff_sq.device, diff_sq.dtype)
        if cache_key not in self._weights_cache:
            exponents = torch.arange(t, dtype=diff_sq.dtype, device=diff_sq.device)
            weights = rho**exponents
            weights = weights / weights.sum()
            self._weights_cache[cache_key] = weights.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)

        return float((diff_sq * self._weights_cache[cache_key]).sum(dim=1).mean())

    def update(self, raw_td: float) -> float:
        """Append *raw_td* to the history deque and recompute the smoothed value.

        Returns:
            The new smoothed TD value (also stored in ``td_smoothed``).
        """
        self.td_raw = raw_td
        self._td_history.append(raw_td)

        sigma = self._config.td_smoothing_sigma
        arr = np.array(self._td_history, dtype=np.float64)

        if sigma <= 0 or len(arr) == 0:
            self.td_smoothed = float(arr[-1]) if len(arr) > 0 else 0.0
        else:
            smoothed = _causal_gaussian_smooth(arr, sigma)
            self.td_smoothed = float(smoothed[-1])

        return self.td_smoothed

    def is_failing(self) -> bool:
        """Return True when the smoothed TD exceeds the configured failure threshold."""
        return self.td_smoothed > self._config.failure_threshold

    def reset(self) -> None:
        """Clear history and reset accumulators (call after recovery or at episode boundary)."""
        self._td_history.clear()
        self.td_raw = 0.0
        self.td_smoothed = 0.0
