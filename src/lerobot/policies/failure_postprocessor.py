"""
Failure detection postprocessor for ACT policy inference.

Computes 6 real-time failure detection metrics during inference and logs them
to a JSONL file for offline visualization:
  1. Temporal Ensembling Disagreement (MSE between old/new overlapping plans)
  2. Proprioceptive Following Error (L2 between target and actual qpos)
  3. Attention Entropy Spike (cross-attention weight entropy)
  4. Visual Feature Mahalanobis Distance (OOD detection via backbone features)
  5. Chunk Endpoint Shift (L2 drift of predicted trajectory endpoint)
  6. Predicted Action Jerk (3rd-order finite diff norm of action chunk)
"""

import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from scipy.ndimage import gaussian_filter1d

from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger(__name__)


class FailurePostprocessor:
    """
    Monitors ACT inference for signs of failure by computing lightweight diagnostic
    metrics each step and streaming them to a JSONL file.

    Usage:
        # After constructing ACTPolicy:
        policy.init_failure_postprocessor(output_dir="/path/to/dataset")
        # Then select_action will automatically compute & log metrics.
    """

    def __init__(
        self,
        policy,
        output_dir=None,
        offline_mu=None,
        offline_inv_cov=None,
        failure_handling_json_path: str | Path | None = None,
        enable_logging: bool | None = None,
    ):
        """
        Args:
            policy: ACTPolicy instance (fully constructed).
            output_dir: Directory to write failure_metrics.jsonl. If None, metrics
                        are computed but not saved.
            offline_mu: (C,) tensor — mean of training-set backbone features (for Mahalanobis).
            offline_inv_cov: (C, C) tensor — inverse covariance of training-set features.
            failure_handling_json_path: Path to failure_handling.json with runtime
                parameters for failure detection/recovery.
            enable_logging: If True, computes/logs full diagnostics; if False,
                only computes failure-handling-required signals.
        """
        self.policy = policy
        self.output_dir = Path(output_dir) if output_dir else None
        self.mu = offline_mu
        self.inv_cov = offline_inv_cov

        self._failure_handling_cfg = self._load_failure_handling_config(failure_handling_json_path)
        if enable_logging is None:
            self._enable_logging = bool(self._failure_handling_cfg["enable_logging"])
        else:
            self._enable_logging = bool(enable_logging)
        self._enable_failure_handling = bool(self._failure_handling_cfg["enable_failure_handling"])
        self._failure_threshold = float(self._failure_handling_cfg["failure_threshold"])
        cp_threshold_raw = self._failure_handling_cfg.get("cp_threshold", self._failure_threshold)
        try:
            self._cp_threshold = float(cp_threshold_raw)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid cp_threshold={cp_threshold_raw}; fallback to failure_threshold={self._failure_threshold}"
            )
            self._cp_threshold = self._failure_threshold

        checkpoint_queue_size = max(1, int(self._failure_handling_cfg["checkpoint_queue_size"]))

        self._eval_delay = max(1, int(self._failure_handling_cfg.get("eval_delay", 15)))
        self._smoothing_sigma = float(self._failure_handling_cfg.get("smoothing_sigma", 2.0))
        self._valley_lookback = max(1, int(self._failure_handling_cfg["valley_lookback"]))
        self._valley_lookahead = max(1, int(self._failure_handling_cfg["valley_lookahead"]))
        self._valley_prominence = float(self._failure_handling_cfg["valley_prominence"])

        self._eval_delay = max(self._eval_delay, self._valley_lookahead)

        min_required_window = self._eval_delay + self._valley_lookback + 1
        self._window_size = max(int(self._failure_handling_cfg.get("window_size", 31)), min_required_window)

        # Internal caches populated by hooks
        self._last_attn_weights = None
        self._cam_features_this_step = []  # accumulates per-camera features
        self._last_chunk_endpoint = None  # for endpoint shift metric

        # Bookkeeping
        self._step = 0
        self._episode = 0
        self._file_handle = None
        self._hooks = []

        # Buffer for backbone features (saved for offline Mahalanobis distance)
        self._feature_buffer = []

        # Online checkpoint tracking
        self._process_step = 0
        self._latest_temporal_disagreement = 0.0
        self._recent_disagreements = deque(maxlen=self._window_size)
        self._recent_actions = deque(maxlen=self._window_size)
        self._recent_steps = deque(maxlen=self._window_size)
        self._checkpoint_action_queue: deque[tuple[int, torch.Tensor]] = deque(maxlen=checkpoint_queue_size)
        self._checkpoint_step_set: set[int] = set()

        # Register hooks only when full diagnostics logging is enabled.
        if self._enable_logging:
            self._register_hooks()

        # Open output file
        if self._enable_logging and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fpath = self.output_dir / "failure_metrics.jsonl"
            self._file_handle = open(fpath, "a")  # noqa: SIM115
            logger.info(f"FailurePostprocessor logging to {fpath}")

    def _load_failure_handling_config(self, failure_handling_json_path: str | Path | None) -> dict:
        cfg = {
            "enable_logging": True,
            "enable_failure_handling": True,
            "failure_threshold": 0.3,
            "cp_threshold": 0.3,
            "checkpoint_queue_size": 5,
            "window_size": 31,
            "eval_delay": 15,
            "smoothing_sigma": 2.0,
            "valley_lookback": 8,
            "valley_lookahead": 8,
            "valley_prominence": 0.0,
        }
        if failure_handling_json_path is None:
            return cfg

        cfg_path = Path(failure_handling_json_path)
        if not cfg_path.exists():
            logger.warning(f"failure_handling.json not found at {cfg_path}, using defaults")
            return cfg

        try:
            with open(cfg_path) as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                logger.warning(f"failure_handling.json at {cfg_path} is not an object, using defaults")
                return cfg

            for key in cfg:
                if key in loaded:
                    cfg[key] = loaded[key]
            logger.info(f"Loaded failure handling config from {cfg_path}")
        except Exception as exc:
            logger.warning(f"Failed to read failure handling config at {cfg_path}: {exc}")

        return cfg

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Register forward hooks to capture internal tensors without modifying model code."""
        model = self.policy.model

        # 1. Cross-attention weights from the last decoder layer
        last_decoder_layer = model.decoder.layers[-1]

        def cross_attn_hook(module, input, output):  # noqa: A002
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights shape: (B, target_seq_len, source_seq_len)
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self._last_attn_weights = output[1].detach()

        h = last_decoder_layer.multihead_attn.register_forward_hook(cross_attn_hook)
        self._hooks.append(h)

        # 2. Backbone features (fires once per camera image in the forward loop)
        if self.policy.config.image_features:

            def backbone_hook(module, input, output):  # noqa: A002
                # IntermediateLayerGetter returns OrderedDict({"feature_map": tensor})
                feat = output["feature_map"].detach()
                self._cam_features_this_step.append(feat)

            h = model.backbone.register_forward_hook(backbone_hook)
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def get_temporal_disagreement(self, new_actions_chunk):
        """
        Metric 1: Temporal Ensembling Disagreement.
        MSE between the old ensembled plan and the new prediction's overlapping portion.
        """
        if not hasattr(self.policy, "temporal_ensembler"):
            return 0.0

        ensembler = self.policy.temporal_ensembler
        if ensembler.ensembled_actions is None:
            return 0.0

        # ensembled_actions: (B, chunk_size-1, action_dim) — the residual plan from last step
        old_plan = ensembler.ensembled_actions
        # new_actions_chunk: (B, chunk_size, action_dim) — take the first chunk_size-1
        overlap_len = old_plan.shape[1]
        new_plan = new_actions_chunk[:, :overlap_len]

        mse = F.mse_loss(old_plan, new_plan)
        return mse.item()

    def get_following_error(self, target_qpos, actual_qpos):
        """
        Metric 2: Proprioceptive Following Error.
        L2 norm between the commanded target and the actual robot joint positions.
        """
        if target_qpos is None or actual_qpos is None:
            return 0.0

        target = torch.as_tensor(target_qpos, dtype=torch.float32).flatten()
        actual = torch.as_tensor(actual_qpos, dtype=torch.float32).flatten()

        # Truncate to the shorter dimension (action_dim <= state_dim typically)
        min_dim = min(target.shape[0], actual.shape[0])
        target = target[:min_dim]
        actual = actual[:min_dim]

        error = torch.norm(target - actual, p=2)
        return error.item()

    def get_attention_entropy(self):
        """
        Metric 3: Cross-Attention Entropy.
        High entropy = model doesn't know where to look = potential OOD.
        """
        if self._last_attn_weights is None:
            return 0.0

        # attn_weights: (B, target_seq, source_seq)
        p = self._last_attn_weights + 1e-9
        entropy = -torch.sum(p * torch.log(p), dim=-1)  # (B, target_seq)
        return entropy.mean().item()

    def get_mahalanobis_distance(self):
        """
        Metric 4: Visual Feature Mahalanobis Distance.
        Large distance = backbone sees something it wasn't trained on.
        """
        if not self._cam_features_this_step or self.mu is None or self.inv_cov is None:
            return 0.0

        # Concatenate all camera features (preserves per-camera spatial info)
        # Each: (B, C, H, W) → GAP → (B, C) → mean over batch → (C,)
        pooled = []
        for feat in self._cam_features_this_step:
            p = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
            p = p.mean(dim=0)  # average over batch → (C,)
            pooled.append(p)

        # Concatenate cameras → (N_cameras * C,)
        feat_vector = torch.cat(pooled, dim=0)

        mu = self.mu.to(feat_vector.device)
        inv_cov = self.inv_cov.to(feat_vector.device)

        diff = feat_vector - mu
        left = torch.matmul(diff, inv_cov)
        mahalanobis_sq = torch.matmul(left, diff)
        return torch.sqrt(torch.abs(mahalanobis_sq)).item()

    def get_endpoint_shift(self, actions_chunk):
        """
        Metric 5: Chunk Endpoint Shift.
        L2 distance between the predicted trajectory for a fixed future point (t+H-1)
        as seen from two consecutive time steps.
        Large shift = the model is drastically re-planning its long-range goal.
        """
        if actions_chunk.dim() < 3 or actions_chunk.shape[1] < 2:
            return 0.0

        # current_plan: (B, chunk_size, action_dim)
        current_plan = actions_chunk.detach()

        if self._last_chunk_endpoint is None:
            # Store the current endpoint (t + chunk_size - 1) for comparison next step
            self._last_chunk_endpoint = current_plan[:, -1]
            return 0.0

        # At the next step, the previous endpoint (t-1 + H) corresponds to
        # the second-to-last item (index -2) in the current sequence.
        shifted_endpoint = current_plan[:, -2]
        shift = torch.norm(shifted_endpoint - self._last_chunk_endpoint, p=2, dim=-1)

        # Update cache with the new furthest point
        self._last_chunk_endpoint = current_plan[:, -1]

        return shift.mean().item()

    def get_action_jerk(self, actions_chunk):
        """
        Metric 6: Predicted Action Jerk.
        3rd-order finite difference (jerk) of the action chunk along time.
        High jerk = jittery/non-smooth predictions, typical of OOD inputs.
        """
        # actions_chunk: (B, chunk_size, action_dim)
        if actions_chunk.dim() < 3 or actions_chunk.shape[1] < 4:
            return 0.0

        velocity = torch.diff(actions_chunk, dim=1)
        acceleration = torch.diff(velocity, dim=1)
        jerk = torch.diff(acceleration, dim=1)
        return torch.norm(jerk, dim=-1).mean().item()

    # ------------------------------------------------------------------
    # Main entry point (called from select_action)
    # ------------------------------------------------------------------

    def process(
        self,
        batch: dict[str, torch.Tensor],
        intended_action: torch.Tensor,
        new_actions_chunk: torch.Tensor | None,
    ) -> torch.Tensor:
        """Route postprocessing after the policy has produced its intended action.

        Args:
            batch: Input observation batch.
            intended_action: Action selected by the policy's internal logic.
            new_actions_chunk: Newly inferred action chunk for this step, if any.
        """
        if new_actions_chunk is not None:
            actual_qpos = batch.get(OBS_STATE)
            target_qpos = new_actions_chunk[:, 0] if new_actions_chunk.dim() == 3 else new_actions_chunk
            self._recent_steps.append(self._process_step)
            self._recent_actions.append(intended_action.detach().clone())
            self._latest_temporal_disagreement = self.get_temporal_disagreement(new_actions_chunk)
            self._recent_disagreements.append(self._latest_temporal_disagreement)
            if self._enable_logging:
                self.compute_and_log(
                    actions_chunk=new_actions_chunk,
                    target_qpos=target_qpos,
                    actual_qpos=actual_qpos,
                    temporal_disagreement=self._latest_temporal_disagreement,
                )
            min_required = self._eval_delay + self._valley_lookback + 1
            if len(self._recent_disagreements) >= min_required:
                self._update_checkpoint_queue()

        is_failure = self._detect_failure(batch, intended_action)
        self._process_step += 1

        if is_failure and self._enable_failure_handling:
            return self._get_recovery_action(batch, intended_action)

        return intended_action

    def _update_checkpoint_queue(self) -> None:
        disagreements = list(self._recent_disagreements)
        total = len(disagreements)

        smoothed_window = gaussian_filter1d(np.array(disagreements), sigma=self._smoothing_sigma)

        eval_idx = total - 1 - self._eval_delay

        eval_val = smoothed_window[eval_idx]
        past_vals = smoothed_window[eval_idx - self._valley_lookback : eval_idx]
        future_vals = smoothed_window[eval_idx + 1 : eval_idx + 1 + self._valley_lookahead]

        if len(past_vals) == 0 or len(future_vals) == 0:
            return

        current_prominence = self._valley_prominence
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

        checkpoint_step = self._recent_steps[eval_idx]

        if checkpoint_step in self._checkpoint_step_set:
            return

        checkpoint_action = self._recent_actions[eval_idx]

        if len(self._checkpoint_action_queue) == self._checkpoint_action_queue.maxlen:
            oldest_step, _ = self._checkpoint_action_queue[0]
            self._checkpoint_step_set.discard(oldest_step)

        self._checkpoint_action_queue.append((checkpoint_step, checkpoint_action.detach().clone()))
        self._checkpoint_step_set.add(checkpoint_step)

        logger.debug(f"Registered new safe Checkpoint at step {checkpoint_step}")

    def _detect_failure(
        self,
        batch: dict[str, torch.Tensor],
        intended_action: torch.Tensor,
    ) -> bool:
        return self._latest_temporal_disagreement > self._cp_threshold

    def _get_recovery_action(
        self,
        batch: dict[str, torch.Tensor],
        intended_action: torch.Tensor,
    ) -> torch.Tensor:
        if self._checkpoint_action_queue:
            _, checkpoint_action = self._checkpoint_action_queue[0]
            return checkpoint_action.to(intended_action.device)
        return intended_action

    def compute_and_log(self, actions_chunk, target_qpos=None, actual_qpos=None, temporal_disagreement=None):
        """
        Compute all 6 metrics and write a single JSON line.

        Args:
            actions_chunk: (B, chunk_size, action_dim) raw model output BEFORE temporal ensembling.
            target_qpos: The action that will be sent to the robot (after ensembling).
            actual_qpos: Current robot joint state from observation.
        """
        metrics = {
            "episode": self._episode,
            "step": self._step,
            "timestamp": time.time(),
            "temporal_disagreement": self.get_temporal_disagreement(actions_chunk)
            if temporal_disagreement is None
            else temporal_disagreement,
            "following_error": self.get_following_error(target_qpos, actual_qpos),
            "attention_entropy": self.get_attention_entropy(),
            "mahalanobis_distance": self.get_mahalanobis_distance(),
            "endpoint_shift": self.get_endpoint_shift(actions_chunk),
            "action_jerk": self.get_action_jerk(actions_chunk),
        }

        # Save GAP-pooled backbone features for offline Mahalanobis distance
        if self._cam_features_this_step and self.output_dir is not None:
            pooled = []
            for feat in self._cam_features_this_step:
                p = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
                p = p.mean(dim=0)  # average over batch → (C,)
                pooled.append(p)
            # Concatenate cameras → (N_cameras * C,)
            feat_vector = torch.cat(pooled, dim=0)
            self._feature_buffer.append(
                {
                    "episode": self._episode,
                    "step": self._step,
                    "feature": feat_vector.cpu(),
                }
            )

        # Write to file
        if self._file_handle is not None:
            self._file_handle.write(json.dumps(metrics) + "\n")
            self._file_handle.flush()

        # Clear per-step caches
        self._cam_features_this_step.clear()
        self._last_attn_weights = None
        self._step += 1

        return metrics

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _flush_features(self):
        """Save accumulated backbone features to disk."""
        if not self._feature_buffer or self.output_dir is None:
            return
        fpath = self.output_dir / "backbone_features.pt"
        # Load existing data if present, then append
        existing = []
        if fpath.exists():
            try:
                existing = torch.load(fpath, weights_only=True)
            except Exception:
                existing = []
        combined = existing + self._feature_buffer
        torch.save(combined, fpath)
        logger.info(f"Saved {len(self._feature_buffer)} feature vectors to {fpath} (total: {len(combined)})")
        self._feature_buffer.clear()

    def reset(self):
        """Call at episode boundaries. Increments episode counter, keeps global step."""
        self._flush_features()
        self._episode += 1
        self._process_step = 0
        self._cam_features_this_step.clear()
        self._last_attn_weights = None
        self._last_chunk_endpoint = None
        self._latest_temporal_disagreement = 0.0
        self._recent_disagreements.clear()
        self._recent_actions.clear()
        self._recent_steps.clear()
        self._checkpoint_action_queue.clear()
        self._checkpoint_step_set.clear()

    def close(self):
        """Clean up hooks, flush features, and close file handle."""
        self._flush_features()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
