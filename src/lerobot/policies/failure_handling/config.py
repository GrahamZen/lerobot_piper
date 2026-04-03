import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-configs — one per component
# ---------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """Parameters for the TD-based failure detector."""

    failure_threshold: float = 0.21  # Smoothed TD > this → failure detected
    td_smoothing_sigma: float = 4.0  # Causal Gaussian std dev for TD smoothing
    td_window_size: int = 31  # Sliding window size for TD history deque
    td_rho: float = 1.0  # Geometric decay for TD weighting (1.0 = uniform)

    def __post_init__(self):
        self.td_window_size = max(1, self.td_window_size)
        if self.failure_threshold < 0:
            self.failure_threshold = 0.3


@dataclass
class StrategyConfig:
    """Parameters for the checkpoint recovery strategy."""

    name: str = "checkpoint"  # Key in STRATEGY_REGISTRY
    feature_type: str = "encoder_out"  # Feature source: "encoder_out" | "backbone"
    template_mode: str = "mean"  # "mean" or "kde" — which aggregated vector to use
    peak_timestep_threshold: int = 30  # Steps after peak before slot is considered "peaked"


@dataclass
class PerturbationConfig:
    """Parameters for the action perturbation module."""

    enabled: bool = False  # If True, perturb actions until first failure
    fn: str = "gripper_gaussian_noise"  # Key in perturbation._REGISTRY
    std: float = 0.02  # Noise std passed to the perturbation function


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class FailureConfig:
    """Top-level configuration for the failure detection and recovery system.

    Nested structure mirrors the JSON layout:
        {
          "enable_failure_handling": ...,
          "enable_logging": ...,
          "flush_metrics_every_step": ...,
          "detector":     { ... },
          "strategy":     { ... },
          "perturbation": { ... },
          "plugins":      [ ... ]
        }

    Load from a JSON file with ``from_json()``.
    Missing sub-sections fall back to sub-config defaults.
    """

    # --- Runtime toggles ---
    enable_failure_handling: bool = False  # If True, trigger recovery on detected failure
    enable_logging: bool = True  # If True, write failure_metrics.jsonl
    flush_metrics_every_step: bool = False  # If True, flush JSONL every step

    # --- Component configs ---
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)

    # --- Extra logging plugins ---
    plugins: list[dict] = field(default_factory=list)
    # e.g. [{"name": "attention_entropy", "enabled": true}]

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: "str | Path | None") -> "FailureConfig":
        """Load config from a JSON file.

        Missing keys fall back to dataclass defaults.
        Extra/unknown keys are silently ignored.
        Returns a default ``FailureConfig`` if *path* is None or does not exist.
        """
        if path is None:
            return cls()

        cfg_path = Path(path)
        if not cfg_path.exists():
            logger.warning("failure_handling.json not found at %s — using defaults", cfg_path)
            return cls()

        try:
            with cfg_path.open() as f:
                raw = json.load(f)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — using defaults", cfg_path, exc)
            return cls()

        try:
            detector = _build_sub(DetectorConfig, raw.get("detector", {}))
            strategy = _build_sub(StrategyConfig, raw.get("strategy", {}))
            perturbation = _build_sub(PerturbationConfig, raw.get("perturbation", {}))

            top_known = {"enable_failure_handling", "enable_logging", "flush_metrics_every_step", "plugins"}
            top_kwargs = {k: v for k, v in raw.items() if k in top_known}

            cfg = cls(detector=detector, strategy=strategy, perturbation=perturbation, **top_kwargs)
            logger.info("Loaded failure handling config from %s", cfg_path)
            return cfg
        except Exception as exc:
            logger.warning("Failed to build FailureConfig from %s: %s — using defaults", cfg_path, exc)
            return cls()


def _build_sub(cls, d: dict):
    """Instantiate a sub-config dataclass, ignoring unknown keys."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in d.items() if k in known})
