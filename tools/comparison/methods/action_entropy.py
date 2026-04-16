"""Action Entropy detector — from AAC (CVPR 2026).

Reference:
    "Adaptive Action Chunking at Inference-time for
    Vision-Language-Action Models" (arXiv:2604.04161, CVPR 2026).

Algorithm
---------
Given B independent action-chunk samples ``(B, H, action_dim)`` from a
stochastic policy at one observation step:

1. For each time-step t in [0, H), compute the per-dimension variance
   across B samples, then sum the Gaussian differential entropies:

       entropy(t) = Σ_d  0.5 * log(2πe · max(σ²_{t,d}, ε))

2. Build the cumulative mean entropy curve:

       avg_entropy(h) = mean(entropy[0 : h+1])   h = 0 … H-1

3. Score per observation step:
   - ``"mean_entropy"`` (default): avg_entropy(H-1) — mean over the full
     chunk horizon.  Stable because it averages H individual entropies.
   - ``"max_jump"``: magnitude of the sharpest jump in avg_entropy.
   - ``"truncation_ratio"``: (H - h*) / H.

4. Detection signal: the cumulative running mean of per-step scores,
   returned by ``update()``.  Averaging across episode steps further
   reduces noise.  The first ``warmup_steps`` calls return ``-inf``
   (suppressed) so the score is not emitted before enough samples
   accumulate.

Score direction for manipulation tasks (ACT / ACT-FM)
------------------------------------------------------
Successful episodes tend to have *higher* entropy — the policy considers
multiple valid strategies.  Failing episodes lock into a single (wrong)
action → *lower* entropy.

Setting ``invert=True`` (default) negates the raw entropy score before
storing, so the direction matches the rest of the framework:
    high (less-negative) stored score → low original entropy → anomalous.

Requirements
------------
Requires B ≥ 2 samples per step.  For ACT (CVAE) policies pass
``sample_latent=True`` to ``predict_action_chunk``; ACT-FM / Diffusion
policies are already stochastic.

Use ``n_samples ≥ 8`` (16 recommended) in ``compute_comparison.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from tools.comparison.methods.base import BaseDetector

_LOG2PIE = math.log(2 * math.pi * math.e)  # constant part of Gaussian entropy


class ActionEntropyDetector(BaseDetector):
    """Failure detector based on intra-chunk action entropy.

    Parameters
    ----------
    score_mode:
        ``"mean_entropy"`` — mean Gaussian differential entropy over the
        full chunk horizon H (default, most stable).
        ``"max_jump"`` — maximum jump in the mean-entropy curve.
        ``"truncation_ratio"`` — (H - h*) / H.
    eps:
        Variance floor for numerical stability.
    min_samples:
        Minimum B required to compute entropy; raw score treated as 0.0.
    invert:
        If ``True`` (default), negate the raw score so that low-entropy
        (anomalous) steps produce high detection scores.
    warmup_steps:
        Number of initial episode steps for which ``update()`` returns
        ``-inf`` (suppressed).  Prevents noisy single-step readings from
        triggering false detections before the running mean stabilises.
    """

    name = "action_entropy"

    def __init__(
        self,
        score_mode: str = "mean_entropy",
        eps: float = 1e-8,
        min_samples: int = 2,
        invert: bool = True,
        warmup_steps: int = 3,
    ) -> None:
        if score_mode not in ("mean_entropy", "max_jump", "truncation_ratio"):
            raise ValueError(
                f"Unknown score_mode '{score_mode}'. Use 'mean_entropy', 'max_jump', or 'truncation_ratio'."
            )
        self.score_mode = score_mode
        self.eps = eps
        self.min_samples = min_samples
        self.invert = invert
        self.warmup_steps = warmup_steps
        self._raw_history: list[float] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._raw_history.clear()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibration_score(self) -> float:
        """Max running mean after warmup (used for CP threshold calibration).

        Using the maximum of the episode's running-mean curve (from step
        ``warmup_steps`` onward) matches what ``_detect_max_threshold``
        computes inline from the stored per-step scores, ensuring
        threshold consistency between ``run_calibration`` and inline eval.
        """
        n = len(self._raw_history)
        if n == 0:
            return 0.0

        # Sum of the first warmup_steps raw values (prefix before post-warmup)
        total = sum(self._raw_history[: self.warmup_steps])
        best = float("-inf")
        for i in range(self.warmup_steps, n):
            total += self._raw_history[i]
            rm = total / (i + 1)  # running mean at episode step i+1 (1-indexed)
            if rm > best:
                best = rm

        if best == float("-inf"):
            # Episode shorter than warmup — fall back to final running mean
            return sum(self._raw_history) / n
        return best

    def update(self, step: int, chunks: Tensor | None) -> float:  # noqa: ARG002
        """Compute per-step score and return the current running mean.

        Args:
            step: Global frame index (unused internally).
            chunks: ``(B, H, action_dim)`` independent policy samples, or
                ``None`` if not available.

        Returns:
            Running cumulative mean of all stored (negated) scores up to
            this episode step, or ``-inf`` during the warmup period.
        """
        if chunks is None or chunks.shape[0] < self.min_samples:
            raw = 0.0
        else:
            step_ent = self._step_entropy(chunks)  # (H,)
            avg_curve = self._mean_entropy_curve(step_ent)  # (H,)

            if self.score_mode == "mean_entropy":
                raw = float(avg_curve[-1].item())
            elif self.score_mode == "max_jump":
                _, max_jump = self._max_diff_point(avg_curve)
                raw = max_jump
            else:  # truncation_ratio
                h_star, _ = self._max_diff_point(avg_curve)
                horizon = chunks.shape[1]
                raw = (horizon - h_star) / horizon

            if self.invert:
                raw = -raw

        self._raw_history.append(raw)
        ep_step = len(self._raw_history)  # 1-indexed

        if ep_step <= self.warmup_steps:
            # Return a large-negative sentinel so max() in CP threshold
            # computation is unaffected and per-step detection never fires.
            return -1e9

        return sum(self._raw_history) / ep_step  # running cumulative mean

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_entropy(self, chunks: Tensor) -> Tensor:
        """Compute Gaussian differential entropy at each chunk time-step.

        Args:
            chunks: ``(B, H, action_dim)``

        Returns:
            ``(H,)`` entropy values (summed over action dimensions).
        """
        # Variance across B samples: (H, action_dim)
        var = chunks.float().var(dim=0)  # (H, D)
        var = var.clamp(min=self.eps)
        # Gaussian entropy per dimension: 0.5 * log(2πe σ²)
        # Summed over D → scalar per time-step
        entropy = 0.5 * (torch.log(var) + _LOG2PIE).sum(dim=-1)  # (H,)
        return entropy

    def _mean_entropy_curve(self, step_ent: Tensor) -> Tensor:
        """Cumulative mean entropy up to each time-step.

        Args:
            step_ent: ``(H,)``

        Returns:
            ``(H,)`` where element h = mean(step_ent[0 : h+1]).
        """
        cumsum = torch.cumsum(step_ent, dim=0)
        indices = torch.arange(1, len(step_ent) + 1, device=step_ent.device, dtype=step_ent.dtype)
        return cumsum / indices

    def _max_diff_point(self, avg_curve: Tensor) -> tuple[int, float]:
        """Find the maximum-difference point on the mean-entropy curve.

        Returns:
            ``(h_star, max_jump)`` where h_star is 1-indexed.
        """
        horizon = len(avg_curve)
        if horizon < 2:
            return 1, 0.0

        diff = avg_curve[1:] - avg_curve[:-1]  # (H-1,)
        idx = int(diff.argmax().item())  # 0-indexed in diff → step idx+1 in curve
        h_star = idx + 1  # 1-indexed
        max_jump = float(diff[idx].item())
        return h_star, max(max_jump, 0.0)
