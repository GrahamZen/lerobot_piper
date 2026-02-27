import logging
from pathlib import Path

import torch

from lerobot.policies.failure_handling.config import FailureConfig
from lerobot.policies.failure_handling.metrics import FailureMetrics
from lerobot.policies.vlm_service import VLMService
from lerobot.utils.constants import OBS_STATE

logger = logging.getLogger(__name__)


class FailurePostprocessor:
    """
    Monitors ACT inference for signs of failure by dispatching observations
    to a FailureMetrics module and triggering VLMService recoveries.
    """

    def __init__(
        self,
        policy,
        output_dir=None,
        failure_handling_json_path: str | Path | None = None,
        enable_logging: bool | None = None,
    ):
        """
        Args:
            policy: ACTPolicy instance (fully constructed).
            output_dir: Directory to write failure_metrics.jsonl.
            failure_handling_json_path: Path to failure_handling.json.
            enable_logging: Optional override for logging feature.
        """
        self.policy = policy
        self.output_dir = Path(output_dir) if output_dir else None

        self.config = FailureConfig.from_json(failure_handling_json_path)
        if enable_logging is not None:
            self.config.enable_logging = bool(enable_logging)

        self.metrics = FailureMetrics(
            config=self.config,
            output_dir=self.output_dir,
        )
        self.metrics.bind_policy(self.policy)

        self.vlm_service = VLMService()
        self._hooks = []

        if self.config.enable_logging:
            self._register_hooks()
            if self.output_dir is not None:
                logger.info(
                    "FailurePostprocessor initialized (metrics/features buffered in memory until finalize)"
                )

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_hooks(self):
        """Register forward hooks to capture internal tensors into self.metrics."""
        model = self.policy.model

        # 1. Cross-attention weights from the last decoder layer
        last_decoder_layer = model.decoder.layers[-1]

        def cross_attn_hook(module, input, output):  # noqa: A002
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self.metrics.last_attn_weights = output[1].detach()

        h = last_decoder_layer.multihead_attn.register_forward_hook(cross_attn_hook)
        self._hooks.append(h)

        # 2. Backbone features (fires once per camera image)
        if self.policy.config.image_features:

            def backbone_hook(module, input, output):  # noqa: A002
                feat = output["feature_map"].detach()
                self.metrics.cam_features_this_step.append(feat)

            h = model.backbone.register_forward_hook(backbone_hook)
            self._hooks.append(h)

    # ------------------------------------------------------------------
    # Main entry point (called from select_action)
    # ------------------------------------------------------------------

    def process(
        self,
        batch: dict[str, torch.Tensor],
        intended_action: torch.Tensor,
        new_actions_chunk: torch.Tensor | None,
    ) -> torch.Tensor:
        """Route postprocessing after the policy has produced its intended action."""
        if new_actions_chunk is not None:
            actual_qpos = batch.get(OBS_STATE)
            target_qpos = new_actions_chunk[:, 0] if new_actions_chunk.dim() == 3 else new_actions_chunk

            self.metrics.latest_temporal_disagreement = self.metrics.get_temporal_disagreement(
                new_actions_chunk
            )

            if self.config.enable_failure_handling:
                self.metrics.append_state(intended_action)

            if self.config.enable_logging:
                self.metrics.compute_and_log(
                    actions_chunk=new_actions_chunk,
                    target_qpos=target_qpos,
                    actual_qpos=actual_qpos,
                    temporal_disagreement=self.metrics.latest_temporal_disagreement,
                )
                if self.config.flush_metrics_every_step:
                    self.metrics.flush_metrics()
                    self.metrics.flush_features()

        is_failure = False
        if self.config.enable_failure_handling:
            is_failure = self.metrics.detect_failure()

        self.metrics.process_step += 1

        if is_failure:
            return self._get_recovery_action(batch, intended_action)

        return intended_action

    def _get_recovery_action(
        self,
        batch: dict[str, torch.Tensor],
        intended_action: torch.Tensor,
    ) -> torch.Tensor:
        if not self.metrics.checkpoint_action_queue:
            return intended_action

        checkpoint_indices = list(range(len(self.metrics.checkpoint_action_queue)))
        selected_index = self.vlm_service.select_checkpoint_index(
            batch=batch,
            checkpoint_indices=checkpoint_indices,
            episode=self.metrics.episode,
            step=self.metrics.process_step,
        )

        if selected_index is None or not (0 <= selected_index < len(self.metrics.checkpoint_action_queue)):
            selected_index = 0

        _, checkpoint_action = self.metrics.checkpoint_action_queue[selected_index]
        return checkpoint_action.to(intended_action.device)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def finalize(self):
        self.metrics.finalize()

    def reset(self):
        """Call at episode boundaries. Increments episode counter, keeps global step."""
        self.metrics.reset()

    def close(self):
        """Clean up hooks, flush features, and close file handle."""
        self.finalize()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
