"""STAC — Spatio-Temporal Action Consistency detector.

Reference:
    "STAC: Failure Detection for Diffusion Policies via Spatio-Temporal
    Action Consistency" (2024).

Implementation follows sentinel/bc/ood_detection/error_utils.py exactly:
    - RBF kernel:  K(x,y) = exp(-gamma * ||x-y||²)
    - Gamma (median heuristic):  gamma = 1 / (2 * median(pairwise_sq_distances))
    - Returns MMD² (not sqrt), accumulated as cumulative score η_t.
    - Overlap: prev_action[:, k:h], curr_action[:, :h-k]
"""

from __future__ import annotations

import torch
from torch import Tensor

from tools.comparison.methods.base import BaseDetector


class STACDetector(BaseDetector):
    """Temporal-consistency monitor based on MMD² between action-chunk distributions.

    Parameters
    ----------
    execution_horizon:
        ``k`` — STAC's temporal gap between compared predictions.
        The two chunks overlap for ``chunk_size - k`` steps.
    chunk_size:
        ``h`` — total length of each predicted action chunk.
        Inferred from the first chunk if ``None``.
    gamma:
        RBF kernel parameter for ``K(x,y) = exp(-gamma * ||x-y||²)``.
        ``"median"`` (default) uses the median pairwise squared-distance
        heuristic as in sentinel: ``gamma = 1 / (2 * median(sq_dists))``.
        A float uses that value directly.
    threshold:
        ``γ`` — cumulative MMD² threshold for failure flagging.
        Set via offline calibration on nominal data.
    """

    name = "stac"

    def __init__(
        self,
        execution_horizon: int = 10,
        chunk_size: int | None = None,
        gamma: float | str = "median",
        threshold: float = 0.5,
    ) -> None:
        self.k = execution_horizon
        self.h = chunk_size
        self.gamma = gamma  # "median" or float
        self.threshold = threshold

        self._prev_chunks: Tensor | None = None
        self._prev_step: int | None = None
        self._cumulative: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_chunks = None
        self._prev_step = None
        self._cumulative = 0.0

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update(self, step: int, chunks: Tensor | None) -> float:
        """Ingest *chunks* at *step* and return the cumulative MMD² score η_t.

        Comparison happens only every ``k`` steps so STAC's temporal horizon
        is decoupled from the policy's execution horizon.
        """
        if chunks is None:
            return self._cumulative

        if self.h is None:
            self.h = chunks.shape[1]

        if step % self.k != 0:
            return self._cumulative

        if self._prev_chunks is not None:
            mmd2 = self._compute_mmd2(self._prev_chunks, chunks)
            self._cumulative += mmd2

        self._prev_chunks = chunks
        self._prev_step = step
        return self._cumulative

    def is_failure(self) -> bool:
        return self._cumulative > self.threshold

    def calibration_score(self) -> float:
        """Final cumulative MMD² at episode end (η_H in the paper)."""
        return self._cumulative

    # ------------------------------------------------------------------
    # MMD² (sentinel-faithful)
    # ------------------------------------------------------------------

    def _resolve_gamma(self, x: Tensor, y: Tensor) -> float:
        """Compute gamma from data (median heuristic) or return fixed value.

        Sentinel: ``gamma = 1.0 / (2 * median(pairwise_sq_distances))``
        where distances are computed over the combined x+y samples.
        """
        if self.gamma != "median":
            return float(self.gamma)

        z = torch.cat([x, y], dim=0)  # (N+M, D)
        diff = z.unsqueeze(1) - z.unsqueeze(0)  # (N+M, N+M, D)
        dist_sq = diff.pow(2).sum(dim=-1)  # (N+M, N+M)
        pos = dist_sq[dist_sq > 0]
        if pos.numel() == 0:
            return 1.0  # identical samples — fallback
        median_sq = float(pos.median())
        return 1.0 / (2.0 * median_sq)

    def _rbf_kernel(self, x: Tensor, y: Tensor, gamma: float) -> Tensor:
        """RBF kernel matrix K(x,y) = exp(-gamma * ||x-y||²).

        Args:
            x: ``(B_x, D)``
            y: ``(B_y, D)``
            gamma: kernel bandwidth parameter.

        Returns:
            ``(B_x, B_y)`` kernel matrix.
        """
        diff_sq = (x.unsqueeze(1) - y.unsqueeze(0)) ** 2  # (B_x, B_y, D)
        dist_sq = diff_sq.sum(dim=-1)  # (B_x, B_y)
        return torch.exp(-gamma * dist_sq)

    def _compute_mmd2(self, old_chunks: Tensor, new_chunks: Tensor) -> float:
        """Compute MMD² between overlapping portions of two chunk batches.

        Overlap extraction (sentinel convention):
            prev = old_chunks[:, k:h]       # a_{t+k : t+h-1 | t}
            curr = new_chunks[:, :h-k]      # a_{t+k : t+h-1 | t+k}

        Args:
            old_chunks: ``(B, h, action_dim)`` — chunks predicted at step *t*.
            new_chunks: ``(B, h, action_dim)`` — chunks predicted at step *t+k*.

        Returns:
            MMD² value (non-negative float).
        """
        h = old_chunks.shape[1]
        overlap = h - self.k

        if overlap <= 0:
            return 0.0

        old_overlap = old_chunks[:, self.k : h, :]
        new_overlap = new_chunks[:, : min(overlap, new_chunks.shape[1]), :]

        actual_overlap = min(old_overlap.shape[1], new_overlap.shape[1])
        if actual_overlap <= 0:
            return 0.0

        old_overlap = old_overlap[:, :actual_overlap, :]
        new_overlap = new_overlap[:, :actual_overlap, :]

        b_x = old_overlap.shape[0]
        b_y = new_overlap.shape[0]

        x = old_overlap.reshape(b_x, -1).float()  # (B_x, overlap*D)
        y = new_overlap.reshape(b_y, -1).float()  # (B_y, overlap*D)

        gamma = self._resolve_gamma(x, y)

        k_xx = self._rbf_kernel(x, x, gamma)
        k_yy = self._rbf_kernel(y, y, gamma)
        k_xy = self._rbf_kernel(x, y, gamma)

        mmd2 = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
        return max(mmd2, 0.0)
