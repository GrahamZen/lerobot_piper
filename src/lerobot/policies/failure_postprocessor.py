"""FailurePostprocessor — thin orchestrator for failure detection and recovery.

Wires together:
  - ``FailureDetector``          — fixed TD-based failure detection
  - ``BaseCheckpointStrategy``   — extendable checkpoint / recovery logic
  - ``LoggingPlugin`` instances  — extendable extra metric logging
  - ``MetricsRecorder``          — JSONL buffering and flushing
"""

import dataclasses
import logging
import pprint
from pathlib import Path
from typing import Any

from torch import Tensor

from lerobot.policies.failure_handling.config import FailureConfig
from lerobot.policies.failure_handling.detector import FailureDetector
from lerobot.policies.failure_handling.perturbation import ActionPerturbation
from lerobot.policies.failure_handling.plugins import build_plugins
from lerobot.policies.failure_handling.recorder import MetricsRecorder
from lerobot.policies.failure_handling.strategies import build_strategy
from lerobot.utils.utils import log_say

logger = logging.getLogger(__name__)


class FailurePostprocessor:
    """Monitor ACT inference, log metrics, and trigger recovery when a failure is detected.

    Public attributes used by visualisation code and lerobot_record.py:
        config:                 ``FailureConfig`` loaded from JSON.
        detector:               ``FailureDetector`` with ``td_raw`` / ``td_smoothed``.
        strategy:               Active ``BaseCheckpointStrategy`` instance.
        recorder:               ``MetricsRecorder`` with ``last_row``.
        recovery_pending_wait:  True during the post-recovery wait period.
        playback_mode:          Set to True in visualisation to replay recorded metrics.
    """

    def __init__(
        self,
        policy,
        output_dir=None,
        failure_handling_json_path: str | Path | None = None,
        enable_logging: bool | None = None,
    ) -> None:
        self._policy = policy
        self.output_dir = Path(output_dir) if output_dir else None

        self.config = FailureConfig.from_json(failure_handling_json_path)
        if enable_logging is not None:
            self.config.enable_logging = bool(enable_logging)

        model_dir = Path(failure_handling_json_path).parent if failure_handling_json_path else None

        # Core components
        self.detector = FailureDetector(self.config.detector)
        self.strategy = build_strategy(self.config.strategy, policy, model_dir)
        self.plugins = build_plugins(self.config.plugins, policy)

        # Collect all forward hooks (strategy + plugins) for unified cleanup
        self._hooks: list = list(self.strategy.forward_hooks)
        for plugin in self.plugins:
            self._hooks.extend(plugin.register_hooks(policy))

        self.recorder = MetricsRecorder(
            output_dir=self.output_dir if self.config.enable_logging else None,
            flush_every_step=self.config.flush_metrics_every_step,
        )

        self.perturbation = ActionPerturbation(self.config.perturbation)

        self.recovery_pending_wait: bool = False
        self.playback_mode: bool = False
        self._step: int = 0  # Episode-local step counter
        self._pending_abs_state: Any = None  # Set by eval loop before each select_action()

        print(f"FailurePostprocessor config: {failure_handling_json_path or '(defaults)'}")
        pprint.pprint(dataclasses.asdict(self.config))

    def set_pending_abs_state(self, abs_state: Any) -> None:
        """Store the current per-env absolute sim state snapshot.

        Called from the eval loop *before* ``policy.select_action()`` so that
        ``_process_live`` can forward it to the strategy for checkpointing.
        ``abs_state`` is typically a list of dicts (one per env) with
        ``qpos``/``qvel`` arrays returned by ``LiberoEnv.get_abs_state()``.
        """
        self._pending_abs_state = abs_state

    # ------------------------------------------------------------------
    # Main entry point (called from ACTPolicy.select_action)
    # ------------------------------------------------------------------

    def process(
        self,
        batch: dict[str, Tensor],
        intended_action: Tensor,
        new_actions_chunk: Tensor | None,
        metrics_override: dict[str, Any] | None = None,
    ) -> Tensor:
        """Post-process the intended action; may substitute a recovery action.

        Args:
            batch:             Current observation batch.
            intended_action:   Action the policy selected.
            new_actions_chunk: Full predicted action chunk (used to compute TD).
                               Pass None if not available.
            metrics_override:  Pre-recorded metric dict for playback mode.

        Returns:
            ``intended_action`` normally, or a recovery action on detected failure.
        """
        if self.playback_mode and metrics_override is not None:
            return self._process_playback(intended_action, metrics_override)

        return self._process_live(batch, intended_action, new_actions_chunk)

    def _process_playback(self, intended_action: Tensor, metrics_override: dict[str, Any]) -> Tensor:
        """Playback path: replay recorded TD, compute live strategy metrics for visualisation."""
        td = float(metrics_override.get("td_raw", 0.0))
        self.detector.update(td)

        # Compute live strategy metrics (e.g. ResNet sim from backbone)
        live = self.strategy.compute_step_metrics()
        live.update({k: v for p in self.plugins for k, v in p.compute().items()})

        self.strategy.reset_step_state()
        for p in self.plugins:
            p.reset_step_state()

        self.recorder.log({**metrics_override, **live})
        self._step += 1
        return intended_action

    def _process_live(
        self,
        batch: dict[str, Tensor],
        intended_action: Tensor,
        new_actions_chunk: Tensor | None,
    ) -> Tensor:
        """Live inference path: compute metrics, update checkpoint, detect failure."""
        if self._step == 0:
            self.strategy.on_episode_start(intended_action)

        # Compute temporal disagreement
        if new_actions_chunk is not None:
            raw_td = self.detector.compute_raw_td(new_actions_chunk, self._policy)
        else:
            raw_td = 0.0
        smoothed_td = self.detector.update(raw_td)

        # Collect all metrics from strategy and plugins
        strategy_metrics = self.strategy.compute_step_metrics()
        plugin_metrics: dict[str, Any] = {}
        for p in self.plugins:
            plugin_metrics.update(p.compute())

        all_metrics: dict[str, Any] = {
            "td_raw": raw_td,
            "td_smoothed": smoothed_td,
            **strategy_metrics,
            **plugin_metrics,
        }

        # Update checkpoint strategy (pass abs state so slot stores it for recovery)
        self.strategy.update(self._step, intended_action, all_metrics, abs_state=self._pending_abs_state)
        all_metrics["best_slot_timestep"] = self.strategy.best_slot_timestep
        self.strategy.reset_step_state()
        for p in self.plugins:
            p.reset_step_state()

        # Log metrics
        if self.config.enable_logging:
            self.recorder.log(all_metrics)

        self._step += 1

        # Failure detection → stop perturbation, optionally trigger recovery
        if self.detector.is_failing():
            self.perturbation.on_failure_detected()
            if self.config.enable_failure_handling and self.strategy.can_recover():
                log_say("Failure detected")
                return self._do_recovery(batch, intended_action)

        return self.perturbation.apply(intended_action)

    def _do_recovery(self, batch: dict[str, Tensor], intended_action: Tensor) -> Tensor:
        """Retrieve the recovery action and reset tracking state."""
        recovery = self.strategy.select_recovery_action(intended_action.device)
        if recovery is None:
            logger.warning("FailurePostprocessor: strategy returned no recovery action")
            return intended_action

        self.recovery_pending_wait = True
        log_say("Attempting recovery")

        # Reset detection state; preserve strategy slot data so repeated failures
        # within the same episode return to the same good checkpoint.
        self.detector.reset()
        self.strategy.reset(keep_last_selected=True)
        self._clear_policy_runtime()
        self._step = 0

        return recovery

    def _clear_policy_runtime(self) -> None:
        """Clear policy-side caches invalidated by a recovery jump."""
        ensembler = getattr(self._policy, "temporal_ensembler", None)
        if ensembler is not None and hasattr(ensembler, "reset"):
            ensembler.reset()

        action_queue = getattr(self._policy, "_action_queue", None)
        if action_queue is not None and hasattr(action_queue, "clear"):
            action_queue.clear()

    # ------------------------------------------------------------------
    # Convenience methods for lerobot_record.py
    # ------------------------------------------------------------------

    def log_recovery_frame(self) -> bool:
        """Log a recovery-wait frame (no new inference this step)."""
        return self.recorder.log_recovery_frame()

    def flush(self) -> None:
        """Flush JSONL buffer and strategy-specific files."""
        self.recorder.flush()
        if self.output_dir is not None:
            self.strategy.flush(self.output_dir)

    def discard_current_episode(self) -> int:
        """Drop in-memory metric rows for the current episode."""
        return self.recorder.discard_current_episode()

    # ------------------------------------------------------------------
    # Episode / session lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Call at episode boundaries: flush, increment episode, reset step state."""
        self.flush()
        self.recorder.reset_episode()
        self.detector.reset()
        self.strategy.reset()
        self.perturbation.reset()
        self._step = 0

    def finalize(self) -> None:
        """Flush all pending data (call before process exits)."""
        self.flush()

    def close(self) -> None:
        """Finalize and remove all registered forward hooks."""
        self.finalize()
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
