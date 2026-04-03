"""Extendable logging-metric plugins.

Plugins add extra columns to ``failure_metrics.jsonl`` but do NOT influence
checkpoint selection or recovery decisions (that is the strategy's job).

Extend the system by:
1. Implementing ``LoggingPlugin`` (Protocol)
2. Decorating with ``@register_plugin``
3. Adding ``{"name": "<your_name>", "enabled": true}`` to ``plugins`` in failure_handling.json
"""

import logging
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LoggingPlugin(Protocol):
    """Interface for extra metric logging plugins."""

    name: str

    def register_hooks(self, policy) -> list:
        """Return a list of ``RemovableHook`` objects registered on *policy*.

        Called once during ``FailurePostprocessor.__init__``.
        The postprocessor stores and removes all hooks on ``close()``.
        """
        ...

    def compute(self) -> dict[str, Any]:
        """Compute this step's metrics.

        Returns a dict of ``{key: scalar_or_tensor}`` pairs that will be
        merged into ``all_metrics`` and logged to JSONL.
        """
        ...

    def reset_step_state(self) -> None:
        """Clear any per-step internal buffers after ``compute()``."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLUGIN_REGISTRY: dict[str, type] = {}


def register_plugin(cls):
    """Class decorator that adds a plugin to the global registry."""
    PLUGIN_REGISTRY[cls.name] = cls
    return cls


def build_plugins(plugin_cfgs: list[dict], policy) -> list[LoggingPlugin]:
    """Instantiate and initialise enabled plugins from *plugin_cfgs*.

    Unknown plugin names are warned about and skipped.
    """
    instances: list[LoggingPlugin] = []
    for cfg in plugin_cfgs:
        name = cfg.get("name", "")
        enabled = cfg.get("enabled", True)
        if not enabled:
            continue
        if name not in PLUGIN_REGISTRY:
            logger.warning("Unknown logging plugin '%s' — skipped", name)
            continue
        try:
            plugin = PLUGIN_REGISTRY[name](policy, cfg)
            instances.append(plugin)
            logger.info("Registered logging plugin: %s", name)
        except Exception as exc:
            logger.warning("Failed to initialise plugin '%s': %s — skipped", name, exc)

    return instances


# ---------------------------------------------------------------------------
# Built-in plugin: attention entropy
# ---------------------------------------------------------------------------


@register_plugin
class AttentionEntropyPlugin:
    """Logs the Shannon entropy of the decoder cross-attention weights.

    High entropy → diffuse/uncertain attention.
    Low entropy → focused/confident attention.

    Metric key logged: ``"attention_entropy"``
    """

    name = "attention_entropy"

    def __init__(self, policy, cfg: dict) -> None:
        self._last_attn_weights: Tensor | None = None
        self._hooks: list = []
        if policy is not None:
            self._register_hook(policy)

    def _register_hook(self, policy) -> None:
        model = getattr(policy, "model", None)
        decoder = getattr(model, "decoder", None) if model else None
        layers = getattr(decoder, "layers", None) if decoder else None
        if not layers:
            logger.warning(
                "AttentionEntropyPlugin: could not find model.decoder.layers — hook not registered"
            )
            return

        last_layer = layers[-1]
        mha = getattr(last_layer, "multihead_attn", None)
        if mha is None:
            logger.warning("AttentionEntropyPlugin: last decoder layer has no multihead_attn")
            return

        def _hook(module, input, output):  # noqa: A002
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                self._last_attn_weights = output[1].detach()

        handle = mha.register_forward_hook(_hook)
        self._hooks.append(handle)

    def register_hooks(self, policy) -> list:
        return self._hooks

    def compute(self) -> dict[str, Any]:
        if self._last_attn_weights is None:
            return {"attention_entropy": 0.0}
        p = self._last_attn_weights + 1e-9
        entropy = -torch.sum(p * torch.log(p), dim=-1).mean()
        return {"attention_entropy": float(entropy.item())}

    def reset_step_state(self) -> None:
        # Intentionally keep last_attn_weights across steps for continuous display;
        # only clear the per-step buffer if needed in a subclass.
        pass
