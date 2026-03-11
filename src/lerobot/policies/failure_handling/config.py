import logging
from dataclasses import dataclass, field
from pathlib import Path

import draccus

logger = logging.getLogger(__name__)


@dataclass
class TemporalDisagreementConfig:
    enabled: bool = True
    failure_threshold: float = 0.3
    cp_threshold: float = -1.0  # Placeholder, fallback to failure_threshold if < 0
    window_size: int = 31
    eval_delay: int = 15
    smoothing_sigma: float = 2.0
    valley_lookback: int = 8
    valley_lookahead: int = 8
    valley_prominence: float = 0.0
    rho: float = 1.0  # Geometric decay factor for weighted temporal disagreement

    def __post_init__(self):
        if self.cp_threshold < 0:
            self.cp_threshold = self.failure_threshold

        self.valley_lookback = max(1, self.valley_lookback)
        self.valley_lookahead = max(1, self.valley_lookahead)
        self.eval_delay = max(1, self.eval_delay)
        self.eval_delay = max(self.eval_delay, self.valley_lookahead)

        min_required_window = self.eval_delay + self.valley_lookback + 1
        self.window_size = max(self.window_size, min_required_window)


@dataclass
class BaseMetricConfig:
    enabled: bool = True
    threshold: float = 1.0


@dataclass
class MahalanobisConfig(BaseMetricConfig):
    offline_mahalanobis_path: Path | None = None


@dataclass
class ActionEntropyConfig(BaseMetricConfig):
    """Config for KDE-based action entropy computed from overlapping chunk predictions.

    Attributes:
        enabled: Whether to compute and log action entropy each step.
        min_bandwidth: Lower bound for Silverman's rule bandwidth, prevents
            degenerate kernels when consecutive predictions are nearly identical.
        min_density: Lower bound for KDE density before taking log, avoids -inf.
    """

    enabled: bool = False
    min_bandwidth: float = 1e-5
    min_density: float = 1e-35


@dataclass
class MetricsConfig:
    temporal_disagreement: TemporalDisagreementConfig = field(default_factory=TemporalDisagreementConfig)
    following_error: BaseMetricConfig = field(
        default_factory=lambda: BaseMetricConfig(enabled=True, threshold=0.05)
    )
    attention_entropy: BaseMetricConfig = field(default_factory=BaseMetricConfig)
    mahalanobis_distance: MahalanobisConfig = field(default_factory=MahalanobisConfig)
    endpoint_shift: BaseMetricConfig = field(default_factory=BaseMetricConfig)
    action_jerk: BaseMetricConfig = field(default_factory=BaseMetricConfig)
    action_entropy: ActionEntropyConfig = field(default_factory=ActionEntropyConfig)


@dataclass
class FailureConfig:
    demo_video_path: str | Path | None = None
    enable_logging: bool = True
    enable_failure_handling: bool = False
    flush_metrics_every_step: bool = False
    checkpoint_queue_size: int = 5

    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self):
        self.checkpoint_queue_size = max(1, self.checkpoint_queue_size)

    @classmethod
    def from_json(cls, path: str | Path | None) -> "FailureConfig":
        """Loads config using draccus from JSON if it exists, otherwise returns defaults."""
        if path is None:
            return cls()

        cfg_path = Path(path)
        if not cfg_path.exists():
            logger.warning(f"failure_handling json not found at {cfg_path}, using defaults")
            return cls()

        try:
            cfg = draccus.parse(cls, config_path=cfg_path, args=[])
            logger.info(f"Loaded failure handling config from {cfg_path}")
            return cfg
        except Exception as exc:
            logger.warning(f"Failed to read failure handling config at {cfg_path}: {exc}. Using defaults.")
            return cls()
