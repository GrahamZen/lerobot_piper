import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot.policies.failure_handling.config import FailureConfig

logger = logging.getLogger(__name__)


def _compute_action_entropy(
    action_samples: torch.Tensor,
    min_bandwidth: float = 1e-5,
    min_density: float = 1e-35,
) -> torch.Tensor:
    """KDE-based action entropy from overlapping chunk predictions.

    Mirrors the paper-style procedure in tools/failure/action_entropy_memory.py
    but operates on (M, D) tensors already assembled by the caller.

    Args:
        action_samples: Shape (M, D) — M overlapping predictions for the same
            target timestep, each of dimension D (action_dim).
        min_bandwidth: Lower-bound for Silverman bandwidth to prevent degenerate kernels.
        min_density: Lower-bound for KDE density before log to avoid -inf.

    Returns:
        Scalar entropy estimate (torch.Tensor, 0-d).
    """
    sample_count, action_dim = action_samples.shape
    dtype = action_samples.dtype
    device = action_samples.device

    min_bw_t = torch.tensor(min_bandwidth, dtype=dtype, device=device)
    min_density_t = torch.tensor(min_density, dtype=dtype, device=device)
    gaussian_norm_t = torch.tensor(math.sqrt(2.0 * math.pi), dtype=dtype, device=device)

    sigma = torch.std(action_samples, dim=0, unbiased=sample_count > 1)
    sigma = torch.clamp(sigma, min=min_bw_t)
    bandwidth = torch.clamp(1.06 * sigma * (sample_count ** (-0.2)), min=min_bw_t)

    squared_mahalanobis = torch.zeros((sample_count, sample_count), dtype=dtype, device=device)
    for dim_idx in range(action_dim):
        vals = action_samples[:, dim_idx]
        diffs = vals.unsqueeze(1) - vals.unsqueeze(0)
        squared_mahalanobis = squared_mahalanobis + (diffs / bandwidth[dim_idx]) ** 2

    log_kernel_norm = torch.log(bandwidth * gaussian_norm_t).sum()
    kernel_vals = torch.exp(-0.5 * squared_mahalanobis - log_kernel_norm)
    densities = torch.clamp(kernel_vals.mean(dim=1), min=min_density_t)
    return -torch.log(densities).mean()


class FailureMetrics:
    """
    Manages state (deques), computes failure metrics, and tracks safe checkpoints.
    """

    def __init__(
        self,
        config: FailureConfig,
        output_dir: Path | None = None,
    ):
        self.config = config
        self.output_dir = output_dir
        self.mu = None
        self.inv_cov = None
        self._policy = None

        mahal_cfg = self.config.metrics.mahalanobis_distance
        if mahal_cfg.enabled and mahal_cfg.offline_mahalanobis_path is not None:
            try:
                dist = torch.load(mahal_cfg.offline_mahalanobis_path, weights_only=True)
                self.mu = dist["mean"]
                self.inv_cov = dist["inv_cov"]
                logger.info(f"Loaded Mahalanobis offline data from {mahal_cfg.offline_mahalanobis_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to load Mahalanobis offline data from {mahal_cfg.offline_mahalanobis_path}: {e}"
                )

        # State / Caches populated externally via the Postprocessor hooking mechanism
        self.cam_features_this_step: list[torch.Tensor] = []
        self.last_attn_weights: torch.Tensor | None = None
        self.last_chunk_endpoint: torch.Tensor | None = None

        # Tracking variables
        self.step: int = 0
        self.episode: int = 0
        self.process_step: int = 0
        self.latest_temporal_disagreement: float = 0.0
        self.latest_smoothed_disagreement: float = 0.0

        # Buffers for JSONL logs and mahalanobis feature dumps
        self.metrics_buffer: list[dict[str, Any]] = []
        self.feature_buffer: list[dict[str, Any]] = []
        self.last_logged_metrics: dict[str, Any] | None = None

        # Online checkpoint state
        td_cfg = self.config.metrics.temporal_disagreement
        self.recent_disagreements: deque = deque(maxlen=td_cfg.window_size)
        self.recent_actions: deque = deque(maxlen=td_cfg.window_size)
        self.recent_steps: deque = deque(maxlen=td_cfg.window_size)
        self.recent_views: deque = deque(maxlen=td_cfg.window_size)

        self.checkpoint_action_queue: deque[tuple[int, torch.Tensor, dict[str, torch.Tensor]]] = deque(
            maxlen=self.config.checkpoint_queue_size
        )
        self.checkpoint_step_set: set[int] = set()

        # Rolling buffer for action-entropy: stores (log_step, chunk_tensor) pairs
        # where chunk_tensor has shape (1, chunk_size, action_dim).
        self.recent_action_chunks: list[tuple[int, torch.Tensor]] = []

    def bind_policy(self, policy):
        """Used to fetch the temporal_ensembler internally."""
        self._policy = policy

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def get_temporal_disagreement(
        self, new_actions_chunk: torch.Tensor, rho: float | None = None
    ) -> float | torch.Tensor:
        if self.process_step == 0 or self._policy is None:
            return 0.0

        if not hasattr(self._policy, "temporal_ensembler"):
            return 0.0

        ensembler = self._policy.temporal_ensembler
        if ensembler.ensembled_actions is None:
            return 0.0

        old_plan = ensembler.ensembled_actions
        overlap_len = old_plan.shape[1]

        if overlap_len == 0 or new_actions_chunk.shape[1] < overlap_len:
            return 0.0

        if rho is None:
            rho = self.config.metrics.temporal_disagreement.rho

        new_plan = new_actions_chunk[:, :overlap_len]
        diff_sq = (old_plan - new_plan) ** 2  # (batch, T, action_dim)

        if rho == 1.0:
            return diff_sq.mean()

        # weights: [1, rho, rho^2, ..., rho^(T-1)], shape (T,)
        t = diff_sq.shape[1]
        exponents = torch.arange(t, dtype=diff_sq.dtype, device=diff_sq.device)
        weights = rho**exponents  # (T,)
        weights = weights / weights.sum()
        # broadcast over (batch, T, action_dim)
        weighted = diff_sq * weights.unsqueeze(0).unsqueeze(-1)
        return weighted.sum(dim=1).mean()

    def get_following_error(
        self, target_qpos: torch.Tensor, actual_qpos: torch.Tensor
    ) -> float | torch.Tensor:
        if target_qpos is None or actual_qpos is None:
            return 0.0

        target = torch.as_tensor(target_qpos, dtype=torch.float32).flatten()
        actual = torch.as_tensor(actual_qpos, dtype=torch.float32).flatten()

        min_dim = min(target.shape[0], actual.shape[0])
        target = target[:min_dim]
        actual = actual[:min_dim]

        return torch.norm(target - actual, p=2)

    def get_attention_entropy(self) -> float | torch.Tensor:
        if self.last_attn_weights is None:
            return 0.0

        p = self.last_attn_weights + 1e-9
        entropy = -torch.sum(p * torch.log(p), dim=-1)
        return entropy.mean()

    def get_mahalanobis_distance(self) -> float | torch.Tensor:
        if not self.cam_features_this_step or self.mu is None or self.inv_cov is None:
            return 0.0

        pooled = []
        for feat in self.cam_features_this_step:
            p = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            p = p.mean(dim=0)
            pooled.append(p)

        feat_vector = torch.cat(pooled, dim=0)
        mu = self.mu.to(feat_vector.device)
        inv_cov = self.inv_cov.to(feat_vector.device)

        diff = feat_vector - mu
        left = torch.matmul(diff, inv_cov)
        mahalanobis_sq = torch.matmul(left, diff)
        return torch.sqrt(torch.abs(mahalanobis_sq))

    def get_endpoint_shift(self, actions_chunk: torch.Tensor) -> float | torch.Tensor:
        if actions_chunk.dim() < 3 or actions_chunk.shape[1] < 2:
            return 0.0

        current_plan = actions_chunk.detach()

        if self.last_chunk_endpoint is None:
            self.last_chunk_endpoint = current_plan[:, -1]
            return 0.0

        shifted_endpoint = current_plan[:, -2]
        shift = torch.norm(shifted_endpoint - self.last_chunk_endpoint, p=2, dim=-1)
        self.last_chunk_endpoint = current_plan[:, -1]

        return shift.mean()

    def get_action_jerk(self, actions_chunk: torch.Tensor) -> float | torch.Tensor:
        if actions_chunk.dim() < 3 or actions_chunk.shape[1] < 4:
            return 0.0

        velocity = torch.diff(actions_chunk, dim=1)
        acceleration = torch.diff(velocity, dim=1)
        jerk = torch.diff(acceleration, dim=1)
        return torch.norm(jerk, dim=-1).mean()

    def get_action_chunk_entropy(
        self, actions_chunk: torch.Tensor
    ) -> tuple[float | torch.Tensor, float | torch.Tensor]:
        """Estimate action entropy and max sample diff from overlapping chunk predictions.

        Maintains a rolling buffer (``recent_action_chunks``) of past action chunks.
        For the current log-step ``s``, every chunk stored at source step ``t`` provides
        a prediction at horizon ``s - t``.  All such overlapping single-step predictions
        are collected and passed to the KDE-based entropy estimator.

        Returns:
            (entropy, max_diff): Both are 0.0 when the overlap count is below
                config.metrics.action_entropy.min_overlap_samples.
        """
        if actions_chunk.dim() < 3:
            return 0.0, 0.0

        ae_cfg = self.config.metrics.action_entropy
        chunk_size = actions_chunk.shape[1]
        current_step = self.step

        # Episode start should always reset the overlap buffer even if global step is continuous.
        if self.process_step == 0:
            self.recent_action_chunks.clear()

        # Inference/logging can have discontinuities (episode boundary, recovery wait,
        # dropped frames). Reset overlap buffer so stale chunks do not create fake spikes.
        if self.recent_action_chunks and current_step != self.recent_action_chunks[-1][0] + 1:
            self.recent_action_chunks.clear()

        # Register the current chunk
        self.recent_action_chunks.append((current_step, actions_chunk.detach()))

        # Discard entries that no longer overlap with the current step
        self.recent_action_chunks = [
            (s, c) for s, c in self.recent_action_chunks if current_step - s < chunk_size
        ]

        # Collect one prediction per overlapping source chunk
        overlapping: list[torch.Tensor] = []
        for source_step, source_chunk in self.recent_action_chunks:
            horizon = current_step - source_step
            if 0 <= horizon < source_chunk.shape[1]:
                overlapping.append(source_chunk[:, horizon, :])  # (1, action_dim)

        min_overlap = max(2, int(ae_cfg.min_overlap_samples))
        if len(overlapping) < min_overlap:
            return 0.0, 0.0

        action_samples = torch.cat(overlapping, dim=0)  # (M, action_dim)
        entropy = _compute_action_entropy(
            action_samples,
            min_bandwidth=ae_cfg.min_bandwidth,
            min_density=ae_cfg.min_density,
        )
        max_diff = torch.abs(action_samples - action_samples[0:1]).max()
        return entropy, max_diff

    # ------------------------------------------------------------------
    # State tracking and checkpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_tensor_for_checkpoint(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().cpu().clone()

    @staticmethod
    def _causal_gaussian_series(values: np.ndarray, *, sigma: float = 4.0) -> np.ndarray:
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

    def _extract_checkpoint_views(self, batch: dict[str, torch.Tensor] | None) -> dict[str, torch.Tensor]:
        if batch is None:
            return {}

        keys = (
            "observation.images.left",
            "observation.images.middle",
            "observation.images.right",
        )

        views: dict[str, torch.Tensor] = {}
        for key in keys:
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                views[key] = self._clone_tensor_for_checkpoint(value)
        return views

    def update_checkpoint_queue(self, smoothed_window: np.ndarray) -> None:
        td_cfg = self.config.metrics.temporal_disagreement
        if not td_cfg.enabled:
            return

        total = len(self.recent_disagreements)
        eval_idx = total - 1 - td_cfg.eval_delay

        eval_val = smoothed_window[eval_idx]
        past_vals = smoothed_window[eval_idx - td_cfg.valley_lookback : eval_idx]
        future_vals = smoothed_window[eval_idx + 1 : eval_idx + 1 + td_cfg.valley_lookahead]

        if len(past_vals) == 0 or len(future_vals) == 0:
            return

        current_prominence = td_cfg.valley_prominence
        if current_prominence <= 0.0:
            current_prominence = max(1e-6, 0.35 * float(np.std(smoothed_window)))

        is_valley = True

        if eval_val > min(past_vals) or eval_val >= min(future_vals):
            is_valley = False
        else:
            if (
                max(past_vals) - eval_val < current_prominence
                or max(future_vals) - eval_val < current_prominence
            ):
                is_valley = False

        if not is_valley:
            return

        checkpoint_step = self.recent_steps[eval_idx]
        if checkpoint_step in self.checkpoint_step_set:
            return

        checkpoint_action = self.recent_actions[eval_idx]
        checkpoint_views = self.recent_views[eval_idx] if len(self.recent_views) > eval_idx else {}

        if len(self.checkpoint_action_queue) == self.checkpoint_action_queue.maxlen:
            oldest_step, _, _ = self.checkpoint_action_queue[0]
            self.checkpoint_step_set.discard(oldest_step)

        self.checkpoint_action_queue.append(
            (checkpoint_step, checkpoint_action.detach().clone(), checkpoint_views)
        )
        self.checkpoint_step_set.add(checkpoint_step)

        logger.debug(f"Registered new safe Checkpoint at step {checkpoint_step}")

    def detect_failure(self) -> bool:
        td_cfg = self.config.metrics.temporal_disagreement
        if not td_cfg.enabled:
            return False

        val = self.latest_smoothed_disagreement
        if torch.is_tensor(val):
            val = val.item()
        logger.debug(
            f"Evaluating failure detection: smoothed_td={val:.4f}, threshold={td_cfg.cp_threshold:.4f}"
        )
        return val > td_cfg.cp_threshold

    def append_state(
        self, intended_action: torch.Tensor, batch: dict[str, torch.Tensor] | None = None
    ) -> None:
        td_cfg = self.config.metrics.temporal_disagreement
        if not td_cfg.enabled:
            return

        self.recent_steps.append(self.process_step)
        self.recent_actions.append(intended_action.detach().clone())
        self.recent_views.append(self._extract_checkpoint_views(batch))

        val = self.latest_temporal_disagreement
        if torch.is_tensor(val):
            val = val.item()
        self.recent_disagreements.append(val)

        disagreements_arr = np.array(self.recent_disagreements, dtype=np.float64)
        smoothed_window = self._causal_gaussian_series(disagreements_arr, sigma=td_cfg.smoothing_sigma)
        if len(smoothed_window) > 0:
            self.latest_smoothed_disagreement = float(smoothed_window[-1])

        min_required = td_cfg.eval_delay + td_cfg.valley_lookback + 1
        if len(self.recent_disagreements) >= min_required:
            self.update_checkpoint_queue(smoothed_window)

    def compute_and_log(self, actions_chunk, target_qpos=None, actual_qpos=None, temporal_disagreement=None):
        metrics = {
            "episode": self.episode,
            "step": self.step,
            "timestamp": time.time(),
            "is_recovery_wait": False,
            "vlm_request": False,
        }

        if self.config.metrics.temporal_disagreement.enabled:
            metrics["temporal_disagreement"] = (
                self.get_temporal_disagreement(actions_chunk)
                if temporal_disagreement is None
                else temporal_disagreement
            )
            for _rho in (0, 0.1, 0.5, 0.7, 0.8, 0.9, 0.95):
                metrics[f"temporal_disagreement_rho{_rho}"] = self.get_temporal_disagreement(
                    actions_chunk, rho=_rho
                )

        if self.config.metrics.following_error.enabled:
            metrics["following_error"] = self.get_following_error(target_qpos, actual_qpos)

        if self.config.metrics.attention_entropy.enabled:
            metrics["attention_entropy"] = self.get_attention_entropy()

        if self.config.metrics.mahalanobis_distance.enabled:
            metrics["mahalanobis_distance"] = self.get_mahalanobis_distance()

        if self.config.metrics.endpoint_shift.enabled:
            metrics["endpoint_shift"] = self.get_endpoint_shift(actions_chunk)

        if self.config.metrics.action_jerk.enabled:
            metrics["action_jerk"] = self.get_action_jerk(actions_chunk)

        if self.config.metrics.action_entropy.enabled:
            _ae, _ae_max_diff = self.get_action_chunk_entropy(actions_chunk)
            metrics["action_entropy"] = _ae
            metrics["action_entropy_max_diff"] = _ae_max_diff

        if (
            self.cam_features_this_step
            and self.output_dir is not None
            and self.config.metrics.mahalanobis_distance.enabled
        ):
            pooled = []
            for feat in self.cam_features_this_step:
                p = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
                p = p.mean(dim=0)
                pooled.append(p)
            feat_vector = torch.cat(pooled, dim=0)
            self.feature_buffer.append(
                {
                    "episode": self.episode,
                    "step": self.step,
                    "feature": feat_vector,
                }
            )

        if self.output_dir is not None:
            self.metrics_buffer.append(metrics)

        self.last_logged_metrics = metrics.copy()

        # Clear per-step caches
        self.cam_features_this_step.clear()
        self.last_attn_weights = None
        self.step += 1

        return metrics

    def mark_last_logged_vlm_request(self) -> bool:
        if self.last_logged_metrics is None:
            return False

        self.last_logged_metrics["vlm_request"] = True
        if self.metrics_buffer:
            self.metrics_buffer[-1]["vlm_request"] = True
        return True

    def log_recovery_wait_step(self) -> bool:
        """
        During recovery wait (no new inference), duplicate the previous metrics row
        and only refresh step/timestamp so metrics stay aligned with dataset frames.
        """
        source = None
        if self.metrics_buffer:
            source = self.metrics_buffer[-1]
        elif self.last_logged_metrics is not None:
            source = self.last_logged_metrics

        if source is None:
            return False

        metrics = source.copy()
        metrics["episode"] = self.episode
        metrics["step"] = self.step
        metrics["timestamp"] = time.time()
        metrics["is_recovery_wait"] = True
        metrics["vlm_request"] = False

        if self.output_dir is not None:
            self.metrics_buffer.append(metrics)

        self.last_logged_metrics = metrics.copy()
        self.step += 1
        return True

    def flush_metrics(self):
        if not self.metrics_buffer or self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fpath = self.output_dir / "failure_metrics.jsonl"
        with open(fpath, "a") as f:
            for metrics in self.metrics_buffer:
                for k, v in metrics.items():
                    if torch.is_tensor(v):
                        metrics[k] = v.item()
                f.write(json.dumps(metrics) + "\n")
        if len(self.metrics_buffer) > 1:
            logger.info(f"Saved {len(self.metrics_buffer)} metric rows to {fpath}")
        self.metrics_buffer.clear()

    def flush_features(self):
        if not self.feature_buffer or self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        fpath = self.output_dir / "backbone_features.pt"

        existing = []
        if fpath.exists():
            try:
                existing = torch.load(fpath, weights_only=True)
            except Exception:
                existing = []

        for item in self.feature_buffer:
            if torch.is_tensor(item["feature"]):
                item["feature"] = item["feature"].cpu()

        combined = existing + self.feature_buffer
        torch.save(combined, fpath)
        if len(self.feature_buffer) > 1:
            logger.info(
                f"Saved {len(self.feature_buffer)} feature vectors to {fpath} (total: {len(combined)})"
            )
        self.feature_buffer.clear()

    def finalize(self):
        if len(self.metrics_buffer) > 0 or len(self.feature_buffer) > 0:
            logger.info(
                f"Flushing {len(self.metrics_buffer)} metric rows and {len(self.feature_buffer)} feature vectors"
            )
        self.flush_metrics()
        self.flush_features()

    def clear_tracking_state(self, *, reset_process_step: bool = True) -> None:
        """
        This is used after a recovery jump to avoid consuming stale statistics
        collected after the selected checkpoint.
        """
        if reset_process_step:
            self.process_step = 0
        self.cam_features_this_step.clear()
        self.last_attn_weights = None
        self.last_chunk_endpoint = None
        self.latest_temporal_disagreement = 0.0
        self.latest_smoothed_disagreement = 0.0
        self.recent_disagreements.clear()
        self.recent_actions.clear()
        self.recent_steps.clear()
        self.recent_views.clear()
        self.recent_action_chunks.clear()
        self.checkpoint_action_queue.clear()
        self.checkpoint_step_set.clear()

    def reset(self):
        self.finalize()
        self.episode += 1
        self.clear_tracking_state(reset_process_step=True)
