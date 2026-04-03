"""Action perturbation module for failure handling.

Perturbation is active from episode start until the first failure is detected,
then disabled for the remainder of the episode.

Built-in perturbation functions (selectable by name in the JSON config):
    - ``gripper_gaussian_noise``: adds Gaussian noise to the last action dimension.

Custom functions can be registered at runtime via ``register_perturbation()``.
"""

import logging
from collections.abc import Callable

import torch
from torch import Tensor

from lerobot.policies.failure_handling.config import PerturbationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in perturbation functions
# Signature: fn(action: Tensor, std: float) -> Tensor
# ---------------------------------------------------------------------------


def _gripper_gaussian_noise(action: Tensor, std: float = 0.02) -> Tensor:
    """Add Gaussian noise to the two gripper dimensions (indices 6 and 13) of a 14-dim action."""
    out = action.clone()
    for idx in (6, 13):
        out[..., idx] = out[..., idx] + torch.randn_like(out[..., idx]) * std
    return out


_REGISTRY: dict[str, Callable] = {
    "gripper_gaussian_noise": _gripper_gaussian_noise,
}


def register_perturbation(name: str, fn: Callable) -> None:
    """Register a custom perturbation function.

    Args:
        name: Key used in the JSON config ``perturbation_fn`` field.
        fn:   Callable with signature ``(action: Tensor, std: float) -> Tensor``.
    """
    _REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Perturbation state machine
# ---------------------------------------------------------------------------


class ActionPerturbation:
    """Apply a registered perturbation function until the first failure is detected.

    State transitions per episode:
        episode start  → ``_active = True``
        failure detected → ``_active = False``  (stays False until next reset)
        episode reset  → ``_active = True``
    """

    def __init__(self, config: PerturbationConfig) -> None:
        self.enabled = config.enabled
        self.std = config.std
        self._active: bool = True

        if config.fn not in _REGISTRY:
            raise ValueError(
                f"Unknown perturbation function '{config.fn}'. Available: {sorted(_REGISTRY.keys())}"
            )
        self._fn = _REGISTRY[config.fn]
        self._fn_name = config.fn

    # ------------------------------------------------------------------

    def apply(self, action: Tensor) -> Tensor:
        """Return perturbed action if enabled and active, otherwise pass through."""
        if self.enabled and self._active:
            return self._fn(action, std=self.std)
        return action

    def on_failure_detected(self) -> None:
        """Disable perturbation for the rest of this episode."""
        if self._active:
            logger.debug("ActionPerturbation: stopping perturbation after failure detection")
            self._active = False

    def reset(self) -> None:
        """Re-enable perturbation for the next episode."""
        self._active = True
