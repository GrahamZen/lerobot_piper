#!/usr/bin/env python
"""Plot temporal disagreement curves with CP thresholds.

Loads failure_metrics.jsonl, extracts temporal_disagreement values,
computes a smoothed version, calculates conformal prediction thresholds
for both, and displays the two plots.

Usage from a notebook:
    from tools.viz.plt_temporal_disagreement import plot_temporal_disagreement
    plot_temporal_disagreement("/path/to/failure_metrics.jsonl")
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# ---------------------------------------------------------------------------
# Default smoothing configuration (matches the notebook defaults)
# ---------------------------------------------------------------------------
DEFAULT_SMOOTHING_CONFIG: dict = {
    "smoothing_method": "gaussian",  # 'gaussian' | 'savgol'
    "sigma": 6.0,
    "window_length": 61,
    "polyorder": 3,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_failure_metrics(metrics_path: str | Path) -> tuple[np.ndarray, dict[int, dict]]:
    """Load *failure_metrics.jsonl* and return (sorted_steps, metrics_by_step)."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"failure_metrics.jsonl not found: {metrics_path}")

    failure_metrics: dict[int, dict] = {}
    with metrics_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                m = json.loads(line)
                if "step" in m:
                    failure_metrics[int(m["step"])] = m
            except (KeyError, ValueError, TypeError):
                continue

    if not failure_metrics:
        raise ValueError(f"No valid metrics found in {metrics_path}")

    steps = np.array(sorted(failure_metrics.keys()), dtype=np.int64)
    return steps, failure_metrics


def _smooth_series(
    values: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Apply Gaussian or Savitzky-Golay smoothing to *values*."""
    if len(values) < 5:
        return values.copy()

    method = cfg.get("smoothing_method", "gaussian")
    if method == "gaussian":
        sigma = cfg.get("sigma", 8.0)
        print(sigma)
        return gaussian_filter1d(values, sigma, mode="nearest")

    # savgol
    window = min(cfg.get("window_length", 61), len(values))
    if window % 2 == 0:
        window -= 1
    window = max(window, 5)
    polyorder = min(cfg.get("polyorder", 3), window - 1)
    return savgol_filter(values, window_length=window, polyorder=polyorder, mode="interp")


def _compute_cp_threshold(
    calibration_scores: np.ndarray,
    alpha: float = 0.001,
) -> float:
    """Conformal‐prediction threshold at miscoverage level *alpha*."""
    valid = calibration_scores[~np.isnan(calibration_scores)]
    if len(valid) == 0:
        return np.nan
    n = len(valid)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    return float(np.quantile(valid, q_level))


def _plot_single_curve(
    steps: np.ndarray,
    y: np.ndarray,
    name: str,
    *,
    title: str | None = None,
    style: str = "-",
    marker: str | None = None,
    step_range: tuple[int, int] | None = None,
    hline: float | None = None,
) -> None:
    """Plot a single curve with an optional horizontal threshold line."""
    plt.figure(figsize=(14, 4))

    if step_range is not None:
        lo, hi = step_range
        mask = (steps >= lo) & (steps <= hi)
        steps_sub = steps[mask]
        y_sub = y[mask]
    else:
        steps_sub = steps
        y_sub = y

    plt.plot(steps_sub, y_sub, linestyle=style, marker=marker, linewidth=1.5, markersize=3)
    plt.title(title if title is not None else name)
    plt.xlabel("step")
    plt.ylabel(name)
    plt.grid(True, alpha=0.3)

    if hline is not None:
        plt.axhline(y=hline, color="r", linestyle="--", label=f"CP threshold={hline:.6f}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_temporal_disagreement(
    metrics_path: str | Path,
    *,
    alpha: float = 0.001,
    step_range: tuple[int, int] | None = None,
) -> float:
    """Load metrics, compute CP threshold, and display the raw plot.

    Parameters
    ----------
    metrics_path : str | Path
        Path to the ``failure_metrics.jsonl`` file.
    alpha : float, optional
        Miscoverage level for the conformal‐prediction threshold (default 0.001).
    step_range : tuple[int, int] | None, optional
        If provided, restrict the x‐axis to ``(start_step, end_step)``.

    Returns
    -------
    float
        ``cp_threshold_raw``
    """
    # 1. Load data
    steps, failure_metrics = _load_failure_metrics(metrics_path)
    print(f"Loaded {len(steps)} points from: {metrics_path}")

    # 2. Build raw temporal disagreement array
    td_raw = np.array(
        [float(failure_metrics[s].get("temporal_disagreement", 0.0)) for s in steps],
        dtype=np.float64,
    )

    # 3. Compute CP threshold
    cp_raw = _compute_cp_threshold(td_raw, alpha)
    print(f"CP Threshold (raw):      {cp_raw}")

    # 4. Plot raw temporal disagreement with threshold
    _plot_single_curve(
        steps,
        td_raw,
        "Temporal Disagreement on CP Calibration Set (Non-Filtered)",
        step_range=step_range,
        hline=cp_raw,
    )

    return cp_raw


def _causal_gaussian_series(
    values: np.ndarray,
    *,
    sigma: float = 4.0,
) -> np.ndarray:
    """Causal (online) Gaussian smoothing.

    Simulates live processing: for each timestep *t* the smoothed value is
    computed using only ``values[t], values[t-1], …`` — no future lookahead.

    A one-sided Gaussian kernel (offsets 0, 1, 2, …, 4·σ) is applied over
    the available past samples and renormalized so the weights sum to 1.

    Parameters
    ----------
    values : np.ndarray
        1-D array of raw values.
    sigma : float, optional
        Standard deviation of the Gaussian kernel (default 4.0).

    Returns
    -------
    np.ndarray
        Causally smoothed array with the same length as *values*.
    """
    print(sigma)
    n = len(values)
    out = np.empty(n, dtype=np.float64)

    # Build one-sided Gaussian kernel: offset 0 = current sample
    radius = int(np.ceil(4 * sigma))
    offsets = np.arange(radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)

    for t in range(n):
        w = min(t + 1, len(kernel))  # available past samples
        k = kernel[:w]
        k_norm = k / k.sum()
        # values[t] * k[0]  +  values[t-1] * k[1]  +  …
        out[t] = np.dot(k_norm, values[t - w + 1 : t + 1][::-1])

    return out


def plot_temporal_disagreement_with_threshold(
    metrics_path: str | Path,
    cp_threshold: float,
    cp_threshold_smoothed: float,
    *,
    sigma: float = 4.0,
    step_range: tuple[int, int] | None = None,
) -> None:
    """Load metrics and display both plots using *provided* CP thresholds.

    The smoothed curve is computed with a **causal Gaussian filter**: at every
    timestep the kernel only looks at past observations, exactly as if the
    disagreement values were arriving one by one at runtime.

    Parameters
    ----------
    metrics_path : str | Path
        Path to the ``failure_metrics.jsonl`` file.
    cp_threshold : float
        Pre‐computed CP threshold for the raw temporal disagreement plot.
    cp_threshold_smoothed : float
        Pre‐computed CP threshold for the smoothed temporal disagreement plot.
    sigma : float, optional
        Standard deviation of the one-sided Gaussian kernel (default 4.0).
    step_range : tuple[int, int] | None, optional
        If provided, restrict the x‐axis to ``(start_step, end_step)``.
    """
    # 1. Load data
    steps, failure_metrics = _load_failure_metrics(metrics_path)
    print(f"Loaded {len(steps)} points from: {metrics_path}")

    # 2. Build raw temporal disagreement array
    td_raw = np.array(
        [float(failure_metrics[s].get("temporal_disagreement", 0.0)) for s in steps],
        dtype=np.float64,
    )

    # 3. Smooth with causal Gaussian (online – each value depends only on the past)
    td_smoothed = _causal_gaussian_series(td_raw, sigma=sigma)

    print(f"CP Threshold (raw):      {cp_threshold}")
    print(f"CP Threshold (smoothed): {cp_threshold_smoothed}")

    # 4. Plot raw temporal disagreement with threshold
    raw_exceeded = steps[td_raw > cp_threshold].tolist()
    print(f"Points exceeding raw threshold ({cp_threshold:.6f}): {raw_exceeded}")

    _plot_single_curve(
        steps,
        td_raw,
        "temporal_disagreement",
        title="Raw Temporal Disagreement",
        step_range=step_range,
        hline=cp_threshold,
    )

    # 5. Plot smoothed temporal disagreement with threshold
    smoothed_exceeded = steps[td_smoothed > cp_threshold_smoothed].tolist()
    print(f"Points exceeding smoothed threshold ({cp_threshold_smoothed:.6f}): {smoothed_exceeded}")

    _plot_single_curve(
        steps,
        td_smoothed,
        "temporal_disagreement",
        title=f"Gaussian Filtered Temporal Disagreement (sigma={sigma})",
        step_range=step_range,
        hline=cp_threshold_smoothed,
    )


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------


def _ema_series(values: np.ndarray, *, span: int = 15) -> np.ndarray:
    """Apply an Exponential Moving Average (EMA) to *values*.

    Parameters
    ----------
    values : np.ndarray
        1-D array of raw values.
    span : int, optional
        The span parameter controls how many past observations influence the
        current value.  ``alpha = 2 / (span + 1)``.  Default is 15.

    Returns
    -------
    np.ndarray
        EMA-smoothed array with the same length as *values*.
    """
    alpha = 2.0 / (span + 1)
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# ---------------------------------------------------------------------------
# Public API – EMA variant
# ---------------------------------------------------------------------------


def plot_temporal_disagreement_with_threshold_ema(
    metrics_path: str | Path,
    cp_threshold: float,
    *,
    ema_span: int = 15,
    step_range: tuple[int, int] | None = None,
) -> None:
    """Plot EMA-smoothed temporal disagreement with a pre-computed CP threshold.

    This is an alternative to :func:`plot_temporal_disagreement_with_threshold`
    that replaces the Gaussian / Savitzky-Golay smoothing with an Exponential
    Moving Average (EMA) and only produces the temporal-disagreement plot
    (no separate smoothed plot).

    Parameters
    ----------
    metrics_path : str | Path
        Path to the ``failure_metrics.jsonl`` file.
    cp_threshold : float
        Pre-computed CP threshold drawn as a horizontal line.
    ema_span : int, optional
        EMA span (default 15).  ``alpha = 2 / (span + 1)``.
    step_range : tuple[int, int] | None, optional
        If provided, restrict the x-axis to ``(start_step, end_step)``.
    """
    # 1. Load data
    steps, failure_metrics = _load_failure_metrics(metrics_path)
    print(f"Loaded {len(steps)} points from: {metrics_path}")

    # 2. Build raw temporal disagreement array
    td_raw = np.array(
        [float(failure_metrics[s].get("temporal_disagreement", 0.0)) for s in steps],
        dtype=np.float64,
    )

    # 3. Apply EMA
    td_ema = _ema_series(td_raw, span=ema_span)

    print(f"CP Threshold: {cp_threshold}")
    print(f"EMA span:     {ema_span}")

    # 4. Plot raw + EMA temporal disagreement with threshold
    if step_range is not None:
        lo, hi = step_range
        mask = (steps >= lo) & (steps <= hi)
        steps_sub = steps[mask]
        td_raw_sub = td_raw[mask]
        td_ema_sub = td_ema[mask]
    else:
        steps_sub = steps
        td_raw_sub = td_raw
        td_ema_sub = td_ema

    plt.figure(figsize=(14, 4))
    plt.plot(steps_sub, td_raw_sub, linewidth=1, alpha=0.4, color="C0", label="raw")
    plt.plot(steps_sub, td_ema_sub, linewidth=1.5, color="C1", label=f"EMA (span={ema_span})")
    if cp_threshold is not None:
        plt.axhline(y=cp_threshold, color="r", linestyle="--", label=f"CP threshold={cp_threshold:.6f}")
    plt.title("temporal_disagreement")
    plt.xlabel("step")
    plt.ylabel("temporal_disagreement")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
