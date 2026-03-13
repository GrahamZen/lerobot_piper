"""Computation helpers for failure detection metrics.

Plotting is optional and only enabled when explicitly requested.
"""

from __future__ import annotations

import importlib
import os
from typing import Literal

import numpy as np


def compute_cp_threshold(calibration_scores: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(calibration_scores)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(calibration_scores, q_level))
    return cp_threshold, q_level, n


def _resolve_hdbscan_class():
    try:
        hdbscan_module = importlib.import_module("hdbscan")
        return hdbscan_module.HDBSCAN
    except Exception:  # nosec B110
        pass

    try:
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        return sklearn_cluster.HDBSCAN
    except Exception as exc:
        raise ImportError(
            "HDBSCAN is required. Install `hdbscan` (preferred), or use a sklearn build with HDBSCAN."
        ) from exc


def _resolve_isolation_forest_class():
    try:
        sklearn_ensemble = importlib.import_module("sklearn.ensemble")
        return sklearn_ensemble.IsolationForest
    except Exception as exc:
        raise ImportError("scikit-learn is required for RoboBase-style outlier filtering.") from exc


def _resolve_matplotlib_pyplot():
    try:
        matplotlib_pyplot = importlib.import_module("matplotlib.pyplot")
        return matplotlib_pyplot
    except Exception as exc:
        raise ImportError("matplotlib is required for plotting entropy clustering figures.") from exc


def _safe_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    std = float(np.std(x))
    if std <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - float(np.mean(x))) / std


def _plot_entropy_and_clusters(
    entropy_for_curve: np.ndarray,
    features: np.ndarray,
    initial_labels: np.ndarray,
    refined_labels: np.ndarray,
    output_dir: str,
    rollout_id: int,
    entropy_curve_name: str,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    plt = _resolve_matplotlib_pyplot()
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6) if figsize == (8, 6) else figsize)
    plt.plot(np.arange(len(entropy_for_curve)), entropy_for_curve, marker="o", markersize=5)
    plt.title("1D Data Plot")
    plt.xlabel("Timestep")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, entropy_curve_name))
    plt.close()

    plt.figure(figsize=figsize)
    plt.scatter(features[:, 0], features[:, 1], c=initial_labels, cmap="viridis", marker="o")
    plt.title("HDBSCAN Initial Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label="Cluster Label")
    plt.savefig(os.path.join(output_dir, f"rollout{rollout_id}-hdbscan-raw.png"))
    plt.close()

    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        features[:, 0],
        features[:, 1],
        c=refined_labels,
        cmap="viridis",
        marker="o",
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("Refined Cluster Label", rotation=270, labelpad=15)
    plt.title("HDBSCAN + Custom Merge Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"rollout{rollout_id}-hdbscan-refine.png"))
    plt.close()


def _remove_outliers_isolation_forest_like_robobase(
    data: np.ndarray,
    contamination: float = 0.1,
) -> np.ndarray:
    isolation_forest = _resolve_isolation_forest_class()
    model = isolation_forest(contamination=contamination)
    predictions = model.fit_predict(data.reshape(-1, 1))
    repaired = data.copy()

    if repaired.size == 0:
        return repaired

    if predictions[0] == -1:
        next_idx = 1
        while next_idx < len(repaired) and predictions[next_idx] == -1:
            next_idx += 1
        if next_idx < len(repaired):
            repaired[0] = repaired[next_idx]

    if predictions[-1] == -1:
        prev_idx = len(repaired) - 2
        while prev_idx >= 0 and predictions[prev_idx] == -1:
            prev_idx -= 1
        if prev_idx >= 0:
            repaired[-1] = repaired[prev_idx]

    for i in range(1, len(repaired) - 1):
        if predictions[i] == -1:
            prev_idx = i - 1
            while prev_idx >= 0 and predictions[prev_idx] == -1:
                prev_idx -= 1

            next_idx = i + 1
            while next_idx < len(repaired) and predictions[next_idx] == -1:
                next_idx += 1

            if prev_idx >= 0 and next_idx < len(repaired):
                repaired[i] = (repaired[prev_idx] + repaired[next_idx]) / 2.0
            elif prev_idx >= 0:
                repaired[i] = repaired[prev_idx]
            elif next_idx < len(repaired):
                repaired[i] = repaired[next_idx]
    return repaired


def cluster_entropy_hdbscan_aloha(
    entropy: np.ndarray,
    min_cluster_size: int = 5,
    warmup_noise_frames: int = 50,
    plot: bool = False,
    plot_dir: str | None = None,
    rollout_id: int | None = None,
) -> np.ndarray:
    """Reproduce ALOHA `imitate_episodes.py` HDBSCAN labeling.

    Returns binary labels where:
    - 0: precision region
    - 1: non-precision/noise region
    """

    hdbscan_cls = _resolve_hdbscan_class()
    entropy = np.asarray(entropy, dtype=float).reshape(-1)
    n = len(entropy)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    entropy_norm = _safe_zscore(entropy)
    indices = _safe_zscore(np.arange(n, dtype=float))
    features = np.stack((indices, entropy_norm), axis=-1)

    clusterer = hdbscan_cls(min_cluster_size=min_cluster_size)
    initial_labels = np.asarray(clusterer.fit_predict(features), dtype=int)

    if n > warmup_noise_frames:
        initial_labels[:warmup_noise_frames] = -1
    else:
        initial_labels[:] = -1

    refined_labels = np.full_like(initial_labels, -1)
    unique_labels = np.unique(initial_labels[initial_labels >= 0])
    for label in unique_labels:
        cluster_points = features[initial_labels == label]
        if np.mean(cluster_points[:, 1]) < 0:
            refined_labels[initial_labels == label] = 0
        else:
            refined_labels[initial_labels == label] = -1

    if plot and plot_dir is not None and rollout_id is not None:
        _plot_entropy_and_clusters(
            entropy_for_curve=entropy,
            features=features,
            initial_labels=initial_labels,
            refined_labels=refined_labels,
            output_dir=plot_dir,
            rollout_id=rollout_id,
            entropy_curve_name=f"rollout{rollout_id}_entropy.png",
            figsize=(8, 6),
        )

    return np.abs(refined_labels).astype(np.int64)


def _split_large_clusters_like_robobase(
    labels: np.ndarray,
    max_size: int = 25,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    if labels.size == 0:
        return labels

    max_label = int(np.max(labels)) if np.any(labels >= 0) else -1
    new_label = max_label + 1

    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) > max_size:
            num_splits = len(cluster_indices) // max_size + int(len(cluster_indices) % max_size > 0)
            for i in range(num_splits):
                split_indices = cluster_indices[i * max_size : (i + 1) * max_size]
                labels[split_indices] = new_label
                new_label += 1
    return labels


def cluster_entropy_hdbscan_robobase(
    entropy: np.ndarray,
    min_cluster_size: int = 5,
    contamination: float = 0.1,
    max_cluster_size: int = 25,
    plot: bool = False,
    plot_dir: str | None = None,
    rollout_id: int | None = None,
) -> np.ndarray:
    """Reproduce RoboBase `utils.hdbscan_with_custom_merge` labeling."""

    hdbscan_cls = _resolve_hdbscan_class()
    entropy = np.asarray(entropy, dtype=float).reshape(-1)
    n = len(entropy)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)

    entropy_norm = _safe_zscore(entropy)
    entropy_norm = _remove_outliers_isolation_forest_like_robobase(
        entropy_norm,
        contamination=contamination,
    )
    entropy_norm = _safe_zscore(entropy_norm)
    indices = _safe_zscore(np.arange(n, dtype=float))
    features = np.stack((indices, entropy_norm), axis=-1)

    clusterer = hdbscan_cls(min_cluster_size=min_cluster_size)
    initial_labels = np.asarray(clusterer.fit_predict(features), dtype=int)
    initial_labels = _split_large_clusters_like_robobase(initial_labels, max_size=max_cluster_size)

    refined_labels = np.full_like(initial_labels, -1)
    unique_labels = np.unique(initial_labels[initial_labels >= 0])
    for label in unique_labels:
        cluster_points = features[initial_labels == label]
        if np.mean(cluster_points[:, 1] < 1):
            refined_labels[initial_labels == label] = 0
        else:
            refined_labels[initial_labels == label] = -1

    if plot and plot_dir is not None and rollout_id is not None:
        _plot_entropy_and_clusters(
            entropy_for_curve=entropy_norm,
            features=features,
            initial_labels=initial_labels,
            refined_labels=refined_labels,
            output_dir=plot_dir,
            rollout_id=rollout_id,
            entropy_curve_name=f"rollout{rollout_id}-entropy-curve.png",
            figsize=(10, 6),
        )

    return np.abs(refined_labels).astype(np.int64)


def compute_action_entropy_safe_threshold(
    episodes_entropies: list[np.ndarray],
    percentile: float = 99.0,
    pipeline: Literal["aloha", "robobase"] = "aloha",
    plot: bool = False,
    plot_dir: str | None = None,
) -> tuple[float, float, int, int]:
    """Compute safety threshold and drop threshold from entropy samples.

    `pipeline="aloha"` reproduces `aloha/act/imitate_episodes.py`.
    `pipeline="robobase"` reproduces `robobase/robobase/utils.py`.
    """

    if plot and plot_dir is None:
        plot_dir = os.path.join(os.getcwd(), "plot")

    all_precision_entropies = []
    all_non_precision_entropies = []
    total_frames = 0
    episode_index = 0

    for episode_entropy_raw in episodes_entropies:
        episode_entropy = np.asarray(episode_entropy_raw, dtype=float).reshape(-1)
        num_frames = len(episode_entropy)
        if num_frames == 0:
            episode_index += 1
            continue
        total_frames += num_frames

        if pipeline == "aloha":
            labels = cluster_entropy_hdbscan_aloha(
                episode_entropy, plot=plot, plot_dir=plot_dir, rollout_id=episode_index
            )
        elif pipeline == "robobase":
            labels = cluster_entropy_hdbscan_robobase(
                episode_entropy, plot=plot, plot_dir=plot_dir, rollout_id=episode_index
            )
        else:
            raise ValueError(f"Unsupported pipeline: {pipeline}")

        precision_mask = labels == 0
        all_precision_entropies.extend(episode_entropy[precision_mask])
        all_non_precision_entropies.extend(episode_entropy[~precision_mask])
        episode_index += 1

    all_precision_entropies = np.array(all_precision_entropies)
    all_non_precision_entropies = np.array(all_non_precision_entropies)

    if all_precision_entropies.size == 0:
        valid_non_empty = [
            np.asarray(ep, dtype=float).reshape(-1) for ep in episodes_entropies if len(ep) > 0
        ]
        if not valid_non_empty:
            return 0.0, 0.0, 0, 0
        all_entropies_flat = np.concatenate(valid_non_empty)
        return float(np.percentile(all_entropies_flat, 50)), 0.15, 0, total_frames
    safe_threshold = float(np.percentile(all_precision_entropies, percentile))
    if all_non_precision_entropies.size > 0:
        median_free_entropy = float(np.median(all_non_precision_entropies))
        typical_prominence = median_free_entropy - safe_threshold
        drop_threshold = max(0.08, typical_prominence * 0.5)
    else:
        drop_threshold = 0.15

    return safe_threshold, drop_threshold, int(all_precision_entropies.size), total_frames
