"""Checkpoint strategy abstraction and implementations.

Extend the system by:
1. Subclassing ``BaseCheckpointStrategy``
2. Decorating with ``@register_strategy``
3. Setting ``"recovery_strategy": "<your_name>"`` in failure_handling.json

Feature type is selected via ``checkpoint_feature_type`` in failure_handling.json.
Available types: "encoder_out" | "backbone"

npz key convention (checkpoint_features.npz):
  {feature_type}_{template_mode}  — template for list-based strategies (e.g. "encoder_out_mean")
  {feature_type}_flat             — L2-norm flat vectors for tensor strategies (e.g. "encoder_out_flat")
  {feature_type}_ep_matrix        — raw per-episode data (e.g. "encoder_out_ep_matrix")

Class hierarchy::

    BaseCheckpointStrategy (abstract)
    ├── _ListSlotStrategy              # list-based per-template slots, feature-type dispatch via config
    │   └── CheckpointStrategy         # name="checkpoint"
    └── _TensorSlotStrategy            # vectorised tensor-based flat slots, feature-type dispatch via config
        └── FlatCheckpointStrategy     # name="checkpoint_flat"
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.failure_handling.config import StrategyConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseCheckpointStrategy(ABC):
    """Interface for checkpoint selection and recovery action retrieval.

    A strategy may register PyTorch forward hooks to capture whatever
    intermediate tensors it needs (e.g. backbone features).

    At each inference step the postprocessor calls:
      1. ``compute_step_metrics()``  — compute & cache this step's metric values
      2. ``update(step, action, all_metrics)`` — maybe save a checkpoint
      3. ``reset_step_state()`` — clear per-step buffers

    On failure the postprocessor calls:
      4. ``can_recover()``
      5. ``select_recovery_action(device)``
      6. ``reset()``  — episode boundary / post-recovery reset
    """

    def __init__(self, policy, config: StrategyConfig, model_dir: Path | None) -> None:
        self._policy = policy
        self._config = config
        self._model_dir = model_dir

    @property
    @abstractmethod
    def forward_hooks(self) -> list:
        """List of registered PyTorch ``RemovableHook`` objects."""

    @abstractmethod
    def on_episode_start(self, initial_action: Tensor) -> None:
        """Register a fallback action to return if no slot has been updated yet."""

    @abstractmethod
    def compute_step_metrics(self) -> dict[str, Any]:
        """Compute strategy-specific metrics for this step.

        The returned dict is merged into ``all_metrics`` before ``update()``
        and before the recorder logs the row.  Keys must be JSON-serialisable
        scalars (or Tensors that the recorder will handle).
        """

    @abstractmethod
    def update(self, step: int, action: Tensor, all_metrics: dict[str, Any]) -> None:
        """Decide whether to save a checkpoint based on *all_metrics*."""

    @abstractmethod
    def reset_step_state(self) -> None:
        """Clear any per-step buffers (e.g. backbone feature cache)."""

    @abstractmethod
    def can_recover(self) -> bool:
        """True when a recovery action is available."""

    @abstractmethod
    def select_recovery_action(self, device) -> Tensor | None:
        """Return the best recovery action, or None if none is available."""

    @abstractmethod
    def flush(self, output_dir: Path) -> None:
        """Persist strategy-specific data to *output_dir* (e.g. similarity tensors)."""

    @abstractmethod
    def reset(self, *, keep_last_selected: bool = False) -> None:
        """Reset internal state.

        Args:
            keep_last_selected: When True (called after recovery), preserve
                checkpoint slot data so repeated failures within the same
                episode return to the same good checkpoint.  When False
                (episode boundary), perform a full reset.
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[BaseCheckpointStrategy]] = {}


def register_strategy(cls: type[BaseCheckpointStrategy]) -> type[BaseCheckpointStrategy]:
    """Class decorator that adds a strategy to the global registry."""
    STRATEGY_REGISTRY[cls.name] = cls
    return cls


def build_strategy(
    strategy_cfg: StrategyConfig,
    policy,
    model_dir: Path | None,
) -> BaseCheckpointStrategy:
    """Instantiate and return the strategy identified by *strategy_cfg.name*."""
    key = strategy_cfg.name.strip().lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown recovery strategy '{strategy_cfg.name}'. Available: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[key](policy, strategy_cfg, model_dir)


# ---------------------------------------------------------------------------
# Feature pooling helpers
# ---------------------------------------------------------------------------

_SlotEntry = dict  # {timestep: int, max_sim: float, action: Tensor | None}


def _pool_encoder_out_feat(feat_buffer: list) -> "Tensor | None":
    """Mean over batch from encoder_out buffer.

    Args:
        feat_buffer: List containing one (B, D) tensor — the mean-pooled
            (over non-latent tokens) encoder output captured by the hook.

    Returns:
        (D,) feature vector, or None if buffer is empty.
    """
    if not feat_buffer:
        return None
    return feat_buffer[0].mean(dim=0)  # (D,) — mean over batch


def _pool_backbone_feats(feat_buffer: list) -> "Tensor | None":
    """Global average pool each camera's feature map and mean across cameras.

    Args:
        feat_buffer: List of (B, C, H, W) tensors, one per camera per forward pass.

    Returns:
        (C,) mean feature vector, or None if buffer is empty.
    """
    if not feat_buffer:
        return None
    cam_vecs = [F.adaptive_avg_pool2d(f, (1, 1)).view(f.shape[0], -1).mean(dim=0) for f in feat_buffer]
    return torch.stack(cam_vecs, dim=0).mean(dim=0)  # (C,)


# ---------------------------------------------------------------------------
# Hook registration helpers
# ---------------------------------------------------------------------------


def _attach_encoder_out_hook(strategy: "BaseCheckpointStrategy", policy, label: str) -> None:
    """Register a forward hook on ``model.encoder``.

    Captures the encoder output (seq_len, B, D) and mean-pools all observation
    tokens → (B, D) stored in ``strategy._feat_buffer``.

    ACT prepends a latent token at index 0 (zeros at inference time); ACT-FM
    has no latent token.  We detect this via ``config.use_vae`` and skip
    index 0 only when a latent is present.
    """
    model = getattr(policy, "model", None)
    encoder = getattr(model, "encoder", None) if model else None
    if encoder is None:
        logger.warning("%s: policy has no model.encoder — hook not registered", label)
        return

    has_latent = bool(getattr(getattr(policy, "config", None), "use_vae", False))

    def _hook(_module, _input, output):
        # output: (seq_len, B, D)
        # Skip index 0 only for ACT (latent token); ACT-FM starts at 0.
        obs_tokens = output[1:] if has_latent else output
        strategy._feat_buffer.append(obs_tokens.mean(dim=0).detach())  # (B, D)

    strategy._hooks.append(encoder.register_forward_hook(_hook))


def _attach_backbone_hook(strategy: "BaseCheckpointStrategy", policy, label: str) -> None:
    """Register a forward hook on ``model.backbone`` → fills ``strategy._feat_buffer``."""
    model = getattr(policy, "model", None)
    backbone = getattr(model, "backbone", None) if model else None
    if backbone is None:
        logger.warning("%s: policy has no model.backbone — hook not registered", label)
        return

    def _hook(_module, _input, output):
        feat = output.get("feature_map") if isinstance(output, dict) else None
        if feat is not None:
            strategy._feat_buffer.append(feat.detach())

    strategy._hooks.append(backbone.register_forward_hook(_hook))


# ---------------------------------------------------------------------------
# Feature-type dispatch tables
# ---------------------------------------------------------------------------

_POOL_FN: dict = {
    "encoder_out": _pool_encoder_out_feat,
    "backbone": _pool_backbone_feats,
}

_HOOK_FN: dict = {
    "encoder_out": _attach_encoder_out_hook,
    "backbone": _attach_backbone_hook,
}


# ---------------------------------------------------------------------------
# Intermediate base: list-based slots
# ---------------------------------------------------------------------------


class _ListSlotStrategy(BaseCheckpointStrategy, ABC):
    """Shared implementation for strategies that use list-based per-template checkpoint slots.

    Each slot independently tracks ``(timestep, max_sim, was_peak, action)`` for one
    template row.  Recovery returns the action from the most-recently-peaked slot.

    Feature type and template selection are driven entirely by config fields:
      - ``checkpoint_feature_type``: "encoder_out" | "backbone"
      - ``checkpoint_template_mode``: "mean" | "kde"
      - npz key loaded: ``{feature_type}_{template_mode}`` (e.g. "encoder_out_mean")

    Concrete subclasses only need to declare:
      - ``name``             — registry key
      - ``_flush_filename``  — .pt filename for sim records, or ``None`` to skip
    """

    _flush_filename: "str | None" = None

    feat_template: Tensor | None = None
    last_similarity: "Tensor | float" = 0.0

    def __init__(self, policy, config: StrategyConfig, model_dir: Path | None) -> None:
        super().__init__(policy, config, model_dir)
        self._feat_buffer: list[Tensor] = []
        self._slots: list[_SlotEntry] = []
        self._fallback_action: Tensor | None = None
        self._sim_records: list[dict] = []
        self._hooks: list = []
        self._episode: int = 0

        self._latest_peak_slot_idx: int = -1

        self._load_template(model_dir)
        if self.feat_template is not None and policy is not None:
            self._register_hooks(policy)

    @property
    def _feature_type(self) -> str:
        return self._config.feature_type

    @property
    def _sim_metric_key(self) -> str:
        return f"{self._feature_type}_similarity"

    def _load_template(self, model_dir: Path | None) -> None:
        if model_dir is None:
            return
        npz_path = model_dir / "checkpoint_features.npz"
        if not npz_path.exists():
            logger.info("No checkpoint features at %s — %s strategy inactive", npz_path, self.name)
            return
        ft = self._feature_type
        mode = self._config.template_mode
        array_key = f"{ft}_{mode}"
        try:
            import numpy as np

            data = np.load(npz_path, allow_pickle=True)
            if array_key not in data.files:
                logger.warning(
                    "Key '%s' not found in %s — %s strategy inactive", array_key, npz_path, self.name
                )
                return
            self.feat_template = torch.from_numpy(data[array_key])
            if self.feat_template.dim() == 1:
                self.feat_template = self.feat_template.unsqueeze(0)
            logger.info(
                "Loaded %s template from %s using '%s' (n=%d slots, dim=%d)",
                self.name,
                npz_path,
                array_key,
                self.feat_template.shape[0],
                self.feat_template.shape[1],
            )
        except Exception as exc:
            logger.warning("Failed to load %s template from %s: %s", self.name, npz_path, exc)

    def _register_hooks(self, policy) -> None:
        ft = self._feature_type
        hook_fn = _HOOK_FN.get(ft)
        if hook_fn is None:
            raise ValueError(f"Unknown checkpoint_feature_type '{ft}'. Available: {sorted(_HOOK_FN)}")
        hook_fn(self, policy, self.name)

    def compute_step_metrics(self) -> dict[str, Any]:
        metric_key = self._sim_metric_key
        if self.feat_template is None or not self._feat_buffer:
            return {metric_key: self.last_similarity}
        ft = self._feature_type
        pool_fn = _POOL_FN.get(ft)
        if pool_fn is None:
            raise ValueError(f"Unknown checkpoint_feature_type '{ft}'. Available: {sorted(_POOL_FN)}")
        feat_vec = pool_fn(self._feat_buffer)
        if feat_vec is None:
            return {metric_key: self.last_similarity}
        template = self.feat_template.to(feat_vec.device)  # (n, D)
        sim: Tensor = F.cosine_similarity(
            feat_vec.unsqueeze(0).expand(template.shape[0], -1), template, dim=1
        )
        self.last_similarity = sim
        return {metric_key: sim}

    def _init_slots(self, n: int) -> None:
        self._slots = [
            {"timestep": -1, "max_sim": float("-inf"), "action": None, "was_peak": False} for _ in range(n)
        ]

    # --- BaseCheckpointStrategy interface ---

    @property
    def forward_hooks(self) -> list:
        return self._hooks

    def on_episode_start(self, initial_action: Tensor) -> None:
        self._fallback_action = initial_action.detach().cpu().clone()

    def update(self, step: int, action: Tensor, all_metrics: dict[str, Any]) -> None:
        if self.feat_template is None:
            return
        raw_sim = all_metrics.get(self._sim_metric_key, 0.0)
        if not torch.is_tensor(raw_sim):
            return

        sim: Tensor = raw_sim  # (n,)
        if not self._slots:
            self._init_slots(sim.shape[0])

        action_clone = action.detach().cpu().clone()
        for i, s in enumerate(sim.tolist()):
            if s > self._slots[i]["max_sim"]:
                self._slots[i]["max_sim"] = s
                self._slots[i]["timestep"] = step
                self._slots[i]["action"] = action_clone

        self._sim_records.append({"episode": self._episode, "step": step, "sim": sim.cpu()})

        # Check if the slot with the highest current-step similarity has peaked
        threshold = self._config.peak_timestep_threshold
        best_i = int(sim.argmax().item())
        best_s = self._slots[best_i]
        if best_s["timestep"] >= 0 and step - best_s["timestep"] > threshold and not best_s["was_peak"]:
            # Reset the previously peaked slot back to initial state
            if self._latest_peak_slot_idx >= 0 and self._latest_peak_slot_idx != best_i:
                old = self._slots[self._latest_peak_slot_idx]
                old["max_sim"] = float("-inf")
                old["was_peak"] = False
            self._slots[best_i]["was_peak"] = True
            self._latest_peak_slot_idx = best_i

    def reset_step_state(self) -> None:
        self._feat_buffer.clear()

    @property
    def best_slot_timestep(self) -> int:
        """Timestep of the most recently peaked (was_peak) slot, or -1 if none has peaked."""
        if self._latest_peak_slot_idx >= 0 and self._slots:
            return self._slots[self._latest_peak_slot_idx]["timestep"]
        return -1

    def can_recover(self) -> bool:
        slot_ready = bool(self._slots) and any(s["timestep"] >= 0 for s in self._slots)
        return self.feat_template is not None and (slot_ready or self._fallback_action is not None)

    def select_recovery_action(self, device) -> Tensor | None:
        if self._latest_peak_slot_idx >= 0 and self._slots:
            slot = self._slots[self._latest_peak_slot_idx]
            if slot["action"] is not None:
                print(
                    f"[Recovery/{self.name}] checkpoint step={slot['timestep']} "
                    f"sim={slot['max_sim']:.4f} (latest peak slot {self._latest_peak_slot_idx})"
                )
                return slot["action"].to(device)
        if self._fallback_action is not None:
            logger.warning("%s: no peaked slot found — using fallback action", self.name)
            return self._fallback_action.to(device)
        return None

    def flush(self, output_dir: Path) -> None:
        if not self._sim_records or output_dir is None or self._flush_filename is None:
            return
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fpath = output_dir / self._flush_filename
        existing: list = []
        if fpath.exists():
            try:
                existing = torch.load(fpath, weights_only=True)
            except Exception:
                existing = []
        combined = existing + self._sim_records
        torch.save(combined, fpath)
        logger.info(
            "Saved %d %s records to %s (total: %d)",
            len(self._sim_records),
            self.name,
            fpath,
            len(combined),
        )
        self._sim_records.clear()

    def reset(self, *, keep_last_selected: bool = False) -> None:
        self._feat_buffer.clear()
        self.last_similarity = 0.0
        self._episode += 1
        if not keep_last_selected:
            self._slots = []
            self._latest_peak_slot_idx = -1


# ---------------------------------------------------------------------------
# Intermediate base: tensor-based flat slots
# ---------------------------------------------------------------------------


class _TensorSlotStrategy(BaseCheckpointStrategy, ABC):
    """Shared implementation for strategies using vectorised tensor-based checkpoint slots.

    Template vectors are pre-L2-normalised; similarity is computed via a single
    matmul (equivalent to cosine similarity for unit-norm vectors).

    Tracks ``was_peak`` per slot: once a slot's similarity starts declining after
    its maximum, it is considered "peaked" and eligible for recovery selection.

    Feature type is driven by ``checkpoint_feature_type`` config field.
    npz flat key loaded: ``{feature_type}_flat`` (e.g. "encoder_out_flat").

    Concrete subclasses only need to declare:
      - ``name`` — registry key
    """

    feat_template: Tensor | None = None
    last_similarity: "Tensor | float" = 0.0

    def __init__(self, policy, config: StrategyConfig, model_dir: Path | None) -> None:
        super().__init__(policy, config, model_dir)
        self._feat_buffer: list[Tensor] = []
        self._fallback_action: Tensor | None = None
        self._hooks: list = []
        self._episode: int = 0

        self._slot_max_sim: Tensor | None = None
        self._slot_timestep: Tensor | None = None
        self._slot_was_peak: Tensor | None = None
        self._slot_actions: list = []
        self._latest_peak_slot_idx: int = -1

        self._load_template(model_dir)
        if self.feat_template is not None and policy is not None:
            self._register_hooks(policy)

    @property
    def _feature_type(self) -> str:
        return self._config.feature_type

    def _load_template(self, model_dir: Path | None) -> None:
        if model_dir is None:
            return
        npz_path = model_dir / "checkpoint_features.npz"
        if not npz_path.exists():
            logger.info("No checkpoint features at %s — %s strategy inactive", npz_path, self.name)
            return
        array_key = f"{self._feature_type}_flat"
        try:
            import numpy as np

            data = np.load(npz_path, allow_pickle=True)
            if array_key not in data.files:
                logger.warning(
                    "Key '%s' not found in %s — %s strategy inactive", array_key, npz_path, self.name
                )
                return
            self.feat_template = F.normalize(
                torch.from_numpy(data[array_key]), dim=1
            )  # (N, d), L2-normalised
            self._init_tensor_slots(self.feat_template.shape[0])
            logger.info(
                "Loaded %s flat template from %s using '%s' (N=%d vectors, dim=%d)",
                self.name,
                npz_path,
                array_key,
                self.feat_template.shape[0],
                self.feat_template.shape[1],
            )
        except Exception as exc:
            logger.warning("Failed to load %s flat template from %s: %s", self.name, npz_path, exc)

    def _register_hooks(self, policy) -> None:
        ft = self._feature_type
        hook_fn = _HOOK_FN.get(ft)
        if hook_fn is None:
            raise ValueError(f"Unknown checkpoint_feature_type '{ft}'. Available: {sorted(_HOOK_FN)}")
        hook_fn(self, policy, self.name)

    def compute_step_metrics(self) -> dict[str, Any]:
        ft = self._feature_type
        if self.feat_template is None or not self._feat_buffer:
            self.last_similarity = 0.0
            return {f"{ft}_flat_sim_max": 0.0}
        pool_fn = _POOL_FN.get(ft)
        if pool_fn is None:
            raise ValueError(f"Unknown checkpoint_feature_type '{ft}'. Available: {sorted(_POOL_FN)}")
        feat_vec = pool_fn(self._feat_buffer)
        if feat_vec is None:
            self.last_similarity = 0.0
            return {f"{self._feature_type}_flat_sim_max": 0.0}
        feat_norm = F.normalize(feat_vec.unsqueeze(0), dim=1)  # (1, d)
        sim: Tensor = (feat_norm @ self.feat_template.to(feat_norm.device).T).squeeze(
            0
        )  # (N,) — cosine similarity
        self.last_similarity = sim
        return {
            f"{self._feature_type}_flat_sim": sim,
            f"{self._feature_type}_flat_sim_max": float(sim.max().item()),
        }

    def _init_tensor_slots(self, n: int) -> None:
        self._slot_max_sim = torch.full((n,), float("-inf"))
        self._slot_timestep = torch.full((n,), -1, dtype=torch.long)
        self._slot_was_peak = torch.zeros(n, dtype=torch.bool)
        self._slot_actions = [None] * n

    # --- BaseCheckpointStrategy interface ---

    @property
    def forward_hooks(self) -> list:
        return self._hooks

    def on_episode_start(self, initial_action: Tensor) -> None:
        self._fallback_action = initial_action.detach().cpu().clone()

    def update(self, step: int, action: Tensor, all_metrics: dict[str, Any]) -> None:
        if self.feat_template is None or not torch.is_tensor(self.last_similarity):
            return

        sim: Tensor = self.last_similarity.cpu()  # (N,)
        better: Tensor = sim > self._slot_max_sim

        indices = better.nonzero(as_tuple=True)[0]
        if len(indices) == 0:
            return

        action_cpu = action.detach().cpu().clone()
        self._slot_max_sim[indices] = sim[indices]
        for idx in indices.tolist():
            self._slot_timestep[idx] = step
            self._slot_actions[idx] = action_cpu

        # Check if the slot with the highest current-step similarity has peaked
        threshold = self._config.peak_timestep_threshold
        best_idx = int(sim.argmax().item())
        if (
            self._slot_timestep[best_idx] >= 0
            and step - int(self._slot_timestep[best_idx].item()) > threshold
            and not self._slot_was_peak[best_idx].item()
        ):
            # Reset the previously peaked slot back to initial state
            if self._latest_peak_slot_idx >= 0 and self._latest_peak_slot_idx != best_idx:
                self._slot_max_sim[self._latest_peak_slot_idx] = float("-inf")
                self._slot_was_peak[self._latest_peak_slot_idx] = False
            self._slot_was_peak[best_idx] = True
            self._latest_peak_slot_idx = best_idx

    def reset_step_state(self) -> None:
        self._feat_buffer.clear()

    @property
    def best_slot_timestep(self) -> int:
        """Timestep of the most recently peaked (was_peak) slot, or -1 if none has peaked."""
        if self._latest_peak_slot_idx >= 0 and self._slot_timestep is not None:
            return int(self._slot_timestep[self._latest_peak_slot_idx].item())
        return -1

    def can_recover(self) -> bool:
        if self.feat_template is None:
            return False
        return bool((self._slot_timestep >= 0).any().item()) or self._fallback_action is not None

    def select_recovery_action(self, device) -> Tensor | None:
        if self._latest_peak_slot_idx >= 0 and self._slot_actions[self._latest_peak_slot_idx] is not None:
            idx = self._latest_peak_slot_idx
            print(
                f"[Recovery/{self.name}] checkpoint step={int(self._slot_timestep[idx].item())} "
                f"sim={float(self._slot_max_sim[idx].item()):.4f} (latest peak slot {idx})"
            )
            return self._slot_actions[idx].to(device)
        if self._fallback_action is not None:
            logger.warning("%s: no peaked slot found — using fallback", self.name)
            return self._fallback_action.to(device)
        return None

    def flush(self, output_dir: Path) -> None:
        pass

    def reset(self, *, keep_last_selected: bool = False) -> None:
        self._feat_buffer.clear()
        self.last_similarity = 0.0
        self._episode += 1
        if not keep_last_selected and self._slot_max_sim is not None:
            self._slot_max_sim.fill_(float("-inf"))
            self._slot_timestep.fill_(-1)
            self._slot_was_peak.fill_(False)
            self._slot_actions = [None] * len(self._slot_actions)
            self._latest_peak_slot_idx = -1


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


@register_strategy
class CheckpointStrategy(_ListSlotStrategy):
    """List-based checkpoint strategy.

    Feature type and template key are driven by config:
      - ``checkpoint_feature_type``: "encoder_out" | "backbone"
      - ``checkpoint_template_mode``: "mean" | "kde"
      - npz key: ``{feature_type}_{template_mode}`` (e.g. "encoder_out_mean")
    """

    name = "checkpoint"
    _flush_filename = "similarity_records.pt"


@register_strategy
class FlatCheckpointStrategy(_TensorSlotStrategy):
    """Flat (tensor-based) checkpoint strategy using all individual feature vectors.

    Feature type driven by ``checkpoint_feature_type`` config field.
    npz key: ``{feature_type}_flat`` (e.g. "encoder_out_flat").
    """

    name = "checkpoint_flat"
