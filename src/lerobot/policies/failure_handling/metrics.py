import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.ndimage import gaussian_filter1d

from lerobot.policies.failure_handling.config import FailureConfig

logger = logging.getLogger(__name__)


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

        # Buffers for JSONL logs and mahalanobis feature dumps
        self.metrics_buffer: list[dict[str, Any]] = []
        self.feature_buffer: list[dict[str, Any]] = []

        # Online checkpoint state
        td_cfg = self.config.metrics.temporal_disagreement
        self.recent_disagreements: deque = deque(maxlen=td_cfg.window_size)
        self.recent_actions: deque = deque(maxlen=td_cfg.window_size)
        self.recent_steps: deque = deque(maxlen=td_cfg.window_size)

        self.checkpoint_action_queue: deque[tuple[int, torch.Tensor]] = deque(
            maxlen=self.config.checkpoint_queue_size
        )
        self.checkpoint_step_set: set[int] = set()

    def bind_policy(self, policy):
        """Used to fetch the temporal_ensembler internally."""
        self._policy = policy

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def get_temporal_disagreement(self, new_actions_chunk: torch.Tensor) -> float | torch.Tensor:
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

        new_plan = new_actions_chunk[:, :overlap_len]
        return F.mse_loss(old_plan, new_plan)

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

    # ------------------------------------------------------------------
    # State tracking and checkpoints
    # ------------------------------------------------------------------

    def update_checkpoint_queue(self) -> None:
        td_cfg = self.config.metrics.temporal_disagreement
        if not td_cfg.enabled:
            return

        disagreements = list(self.recent_disagreements)
        total = len(disagreements)

        smoothed_window = gaussian_filter1d(np.array(disagreements), sigma=td_cfg.smoothing_sigma)
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

        if len(self.checkpoint_action_queue) == self.checkpoint_action_queue.maxlen:
            oldest_step, _ = self.checkpoint_action_queue[0]
            self.checkpoint_step_set.discard(oldest_step)

        self.checkpoint_action_queue.append((checkpoint_step, checkpoint_action.detach().clone()))
        self.checkpoint_step_set.add(checkpoint_step)

        logger.debug(f"Registered new safe Checkpoint at step {checkpoint_step}")

    def detect_failure(self) -> bool:
        td_cfg = self.config.metrics.temporal_disagreement
        if not td_cfg.enabled:
            return False

        val = self.latest_temporal_disagreement
        if torch.is_tensor(val):
            val = val.item()
        return val > td_cfg.cp_threshold

    def append_state(self, intended_action: torch.Tensor) -> None:
        td_cfg = self.config.metrics.temporal_disagreement
        if not td_cfg.enabled:
            return

        self.recent_steps.append(self.process_step)
        self.recent_actions.append(intended_action.detach().clone())

        val = self.latest_temporal_disagreement
        if torch.is_tensor(val):
            val = val.item()
        self.recent_disagreements.append(val)

        min_required = td_cfg.eval_delay + td_cfg.valley_lookback + 1
        if len(self.recent_disagreements) >= min_required:
            self.update_checkpoint_queue()

    def compute_and_log(self, actions_chunk, target_qpos=None, actual_qpos=None, temporal_disagreement=None):
        metrics = {
            "episode": self.episode,
            "step": self.step,
            "timestamp": time.time(),
        }

        if self.config.metrics.temporal_disagreement.enabled:
            metrics["temporal_disagreement"] = (
                self.get_temporal_disagreement(actions_chunk)
                if temporal_disagreement is None
                else temporal_disagreement
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

        # Clear per-step caches
        self.cam_features_this_step.clear()
        self.last_attn_weights = None
        self.step += 1

        return metrics

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
        logger.info(f"Saved {len(self.feature_buffer)} feature vectors to {fpath} (total: {len(combined)})")
        self.feature_buffer.clear()

    def finalize(self):
        logger.info(
            f"Flushing {len(self.metrics_buffer)} metric rows and {len(self.feature_buffer)} feature vectors"
        )
        self.flush_metrics()
        self.flush_features()

    def reset(self):
        self.finalize()
        self.episode += 1
        self.process_step = 0
        self.cam_features_this_step.clear()
        self.last_attn_weights = None
        self.last_chunk_endpoint = None
        self.latest_temporal_disagreement = 0.0
        self.recent_disagreements.clear()
        self.recent_actions.clear()
        self.recent_steps.clear()
        self.checkpoint_action_queue.clear()
        self.checkpoint_step_set.clear()
