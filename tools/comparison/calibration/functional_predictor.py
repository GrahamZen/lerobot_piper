"""Functional conformal prediction band for time-series anomaly detection.

Ported from FAIL-Detect (RSS 2025):
    "Can We Detect Failures Without Failure Data? Uncertainty-Aware Runtime
    Failure Detection for Imitation Learning Policies"
    https://github.com/BerkeleyAutomation/FAIL-Detect

Original source: UQ_test/timeseries_cp/methods/functional_predictor.py

Usage
-----
Given per-episode score sequences from **successful** trajectories only:

    from tools.comparison.calibration.functional_predictor import (
        FunctionalPredictor, ModulationType, RegressionType
    )

    predictor = FunctionalPredictor(
        modulation_type=ModulationType.Tfunc,
        regression_type=RegressionType.Mean,
    )
    # train_data : (n_train, T)  — successful episode score sequences
    # cal_data   : (n_cal,   T)  — successful episode score sequences (held-out)
    lower_bound = predictor.get_one_sided_prediction_band(
        train_data, cal_data, alpha=0.025, lower_bound=True
    )  # shape (T,) — time-varying failure threshold

At test time, a trajectory is flagged as failure the first timestep its
score falls at or below this lower bound.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

# ---------------------------------------------------------------------------
# Enums (mirrors timeseries_cp)
# ---------------------------------------------------------------------------


class RegressionType(Enum):
    Mean = 1


class ModulationType(Enum):
    Const = 1
    Stdev = 2
    Tfunc = 3  # recommended — best empirical performance in FAIL-Detect paper


# ---------------------------------------------------------------------------
# Regression helper
# ---------------------------------------------------------------------------


def _regress(training_data: np.ndarray, regression_type: RegressionType) -> np.ndarray:
    """Point-predict a single trajectory from training data.

    Args:
        training_data: ``(n_train, T)`` array of score sequences.
        regression_type: Only ``Mean`` is implemented.

    Returns:
        ``(1, T)`` array.
    """
    if regression_type == RegressionType.Mean:
        return np.mean(training_data, axis=0, keepdims=True)
    raise NotImplementedError(f"Unknown regression type: {regression_type}")


# ---------------------------------------------------------------------------
# FunctionalPredictor
# ---------------------------------------------------------------------------


class FunctionalPredictor:
    """Distribution-free functional conformal prediction band.

    Based on "The Importance of Being a Band: Finite-Sample Exact
    Distribution-Free Prediction Sets for Functional Data"
    (https://arxiv.org/abs/2102.06746), applied to robot failure detection
    in the FAIL-Detect paper.

    Parameters
    ----------
    modulation_type:
        Controls the shape of the prediction band width over time.
        ``Tfunc`` (default) uses the data-adaptive functional modulation
        from the paper and gives the best empirical results.
    regression_type:
        Point estimate used as the band centre.  Only ``Mean`` is supported.
    """

    def __init__(
        self,
        modulation_type: ModulationType = ModulationType.Tfunc,
        regression_type: RegressionType = RegressionType.Mean,
    ) -> None:
        self.modulation_type = modulation_type
        self.regression_type = regression_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_one_sided_prediction_band(
        self,
        training_data: np.ndarray,
        calibration_data: np.ndarray,
        alpha: float,
        lower_bound: bool = True,
    ) -> np.ndarray:
        """Compute a one-sided time-varying prediction band from successful episodes.

        Args:
            training_data: ``(n_train, T)`` — per-step scores of successful
                trajectories used to fit the band centre and modulation.
            calibration_data: ``(n_cal, T)`` — held-out successful trajectories
                used to set the conformal quantile.
            alpha: Significance level.  The band is valid with probability
                ``1 - alpha`` under exchangeability.  Typical values: 0.01–0.05.
            lower_bound: If ``True`` returns the lower bounding trajectory
                (scores **below** it signal failure).  If ``False`` returns the
                upper bounding trajectory.

        Returns:
            ``(T,)`` array — the time-varying threshold trajectory.
        """
        seq_len = training_data.shape[-1]
        assert calibration_data.shape[-1] == seq_len, "Train / cal sequence lengths must match."
        assert 0.0 < alpha < 1.0

        point_pred = _regress(training_data, self.regression_type)  # (1, T)
        modulation = self._get_modulation(training_data, point_pred, alpha)  # (1, T)

        if not lower_bound:
            cal_scores = [
                float(np.max((cal_traj - point_pred) / modulation)) for cal_traj in calibration_data
            ]
        else:
            cal_scores = [
                float(np.max((point_pred - cal_traj) / modulation)) for cal_traj in calibration_data
            ]

        band_width = float(np.quantile(cal_scores, 1.0 - alpha))

        result = point_pred - band_width * modulation if lower_bound else point_pred + band_width * modulation

        return result.flatten()  # (T,)

    def get_prediction_band(
        self,
        training_data: np.ndarray,
        calibration_data: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a symmetric two-sided prediction band.

        Returns:
            ``(upper, lower)`` — each ``(T,)`` array.
        """
        seq_len = training_data.shape[-1]
        assert calibration_data.shape[-1] == seq_len
        assert 0.0 < alpha < 1.0

        point_pred = _regress(training_data, self.regression_type)
        modulation = self._get_modulation(training_data, point_pred, alpha)

        cal_scores = [
            float(np.max(np.abs(cal_traj - point_pred) / modulation)) for cal_traj in calibration_data
        ]
        n_cal = len(cal_scores)
        band_width = float(np.sort(cal_scores)[int(np.ceil((n_cal + 1) * (1 - alpha))) - 1])

        upper = (point_pred + band_width * modulation).flatten()
        lower = (point_pred - band_width * modulation).flatten()
        return upper, lower

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_modulation(
        self,
        training_data: np.ndarray,
        point_pred: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Compute per-timestep modulation ``σ(t)``; shape ``(1, T)``."""
        eps = 1e-8
        seq_len = training_data.shape[-1]

        if self.modulation_type == ModulationType.Const:
            return np.ones((1, seq_len)) / seq_len

        if self.modulation_type == ModulationType.Stdev:
            return np.std(training_data, axis=0, ddof=1, keepdims=True) + eps

        if self.modulation_type == ModulationType.Tfunc:
            n_train = training_data.shape[0]
            abs_dev = np.abs(training_data - point_pred)  # (n_train, T)

            # If the (1-α) quantile index would exceed n_train, fall back to
            # max over all training trajectories (conservative but valid).
            idx = int(np.ceil((n_train + 1) * (1 - alpha))) - 1
            if idx >= n_train:
                return abs_dev.max(axis=0, keepdims=True) + eps

            # Gamma = the (1-α) quantile of the per-trajectory max deviations.
            per_traj_max = abs_dev.max(axis=1)  # (n_train,)
            gamma = float(np.sort(per_traj_max)[idx])

            # Keep only trajectories whose max deviation ≤ gamma.
            mask = per_traj_max <= gamma
            filtered = abs_dev[mask]  # (n_kept, T)
            if filtered.shape[0] == 0:
                return abs_dev.max(axis=0, keepdims=True) + eps

            return filtered.max(axis=0, keepdims=True) + eps

        raise NotImplementedError(f"Unknown modulation type: {self.modulation_type}")
