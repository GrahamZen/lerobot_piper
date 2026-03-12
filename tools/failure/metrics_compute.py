"""Pure computation helpers for failure detection metrics.

These functions have no I/O dependencies and can be reused independently.
"""

from __future__ import annotations

import importlib

import numpy as np


def compute_cp_threshold(calibration_scores: np.ndarray, alpha: float) -> tuple[float, float, int]:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n = len(calibration_scores)
    q_level = float(np.ceil((n + 1) * (1 - alpha)) / n)
    q_level = min(q_level, 1.0)
    cp_threshold = float(np.quantile(calibration_scores, q_level))
    return cp_threshold, q_level, n


def compute_action_entropy_safe_threshold(
    episodes_entropies: list[np.ndarray],
    percentile: float = 95.0,
    min_cluster_size: int = 50,
    min_samples: int = 10,
    contamination: float = 0.05,
    plot: bool = False,
) -> tuple[float, int, int]:
    try:
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        sklearn_preprocessing = importlib.import_module("sklearn.preprocessing")
        sklearn_ensemble = importlib.import_module("sklearn.ensemble")
    except Exception as exc:
        raise ImportError(
            "scikit-learn is required. Please install scikit-learn in the current environment."
        ) from exc

    hdbscan = sklearn_cluster.HDBSCAN
    standard_scaler = sklearn_preprocessing.StandardScaler
    isolation_forest = sklearn_ensemble.IsolationForest

    if not isinstance(episodes_entropies, list) or len(episodes_entropies) == 0:
        raise ValueError("episodes_entropies must be a non-empty list of 1D arrays")

    all_precision_entropies = []
    total_frames = 0

    # Used to collect plotting data: stores (entropy_array, p_mask_array) for each episode
    plot_data = []

    for episode_entropy_raw in episodes_entropies:
        if episode_entropy_raw.ndim != 1:
            raise ValueError("Each episode entropy must be a 1D array")

        t = len(episode_entropy_raw)
        if t == 0:
            continue

        total_frames += t

        # Copy to avoid modifying the original data
        episode_entropy = episode_entropy_raw.copy()

        # ==========================================
        # 0. Paper step 1: Isolation Forest outlier filtering with nearest-neighbor replacement
        # ==========================================
        iso_forest = isolation_forest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(episode_entropy.reshape(-1, 1))

        normal_indices = np.where(outlier_labels == 1)[0]
        outlier_indices = np.where(outlier_labels == -1)[0]

        if len(normal_indices) > 0 and len(outlier_indices) > 0:
            for idx in outlier_indices:
                nearest_normal_idx = normal_indices[np.argmin(np.abs(normal_indices - idx))]
                episode_entropy[idx] = episode_entropy[nearest_normal_idx]

        # ==========================================
        # 1. Paper step 2: Concatenate the time index t
        # ==========================================
        t_indices = np.arange(t).reshape(-1, 1)
        h_values = episode_entropy.reshape(-1, 1)
        features = np.hstack([t_indices, h_values])

        # ==========================================
        # 2. Paper step 3: Episode-level normalization
        # ==========================================
        scaler = standard_scaler()
        features_normalized = scaler.fit_transform(features)

        # ==========================================
        # 3. Paper step 4: Run HDBSCAN clustering
        # ==========================================
        clusterer = hdbscan(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(features_normalized)

        # ==========================================
        # 4. Paper step 5: Filter and extract the high-precision region (P set)
        # ==========================================
        valid_p_indices = []
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue  # Label -1 is noise and belongs to the Casualness set

            cluster_mask = labels == label
            mean_norm_entropy = features_normalized[cluster_mask, 1].mean()

            if mean_norm_entropy < 0:
                valid_p_indices.extend(np.where(cluster_mask)[0])

        all_precision_entropies.extend(episode_entropy[valid_p_indices])

        # [Added] Collect visualization data
        if plot:
            p_mask = np.zeros(t, dtype=bool)
            p_mask[valid_p_indices] = True
            plot_data.append((episode_entropy, p_mask))

    all_precision_entropies = np.array(all_precision_entropies)

    # Fallback mechanism
    if all_precision_entropies.size == 0:
        all_entropies_flat = np.concatenate(episodes_entropies)
        safe_threshold = float(np.percentile(all_entropies_flat, 50))
        p_size = 0
    else:
        # ==========================================
        # 5. Extract the hard threshold for real-world deployment
        # ==========================================
        safe_threshold = float(np.percentile(all_precision_entropies, percentile))
        p_size = int(all_precision_entropies.size)

    # ==========================================
    # 6. [Added] Plotting logic
    # ==========================================
    if plot and plot_data:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: 'matplotlib' is required for plotting. Skipping visualization.")
        else:
            plt.figure(figsize=(14, 6))
            current_t = 0

            for i, (ep_entropy, p_mask) in enumerate(plot_data):
                t_ep = len(ep_entropy)
                t_arr = np.arange(current_t, current_t + t_ep)

                # Plot the P set (blue points)
                plt.scatter(
                    t_arr[p_mask],
                    ep_entropy[p_mask],
                    c="#1f77b4",
                    s=12,
                    alpha=0.8,
                    label="P Set (Precision)" if i == 0 else "",
                )

                # Plot the C set (orange points)
                plt.scatter(
                    t_arr[~p_mask],
                    ep_entropy[~p_mask],
                    c="#ff7f0e",
                    s=12,
                    alpha=0.4,
                    label="C Set (Casualness)" if i == 0 else "",
                )

                # Draw separators between episodes
                if i > 0:
                    plt.axvline(x=current_t, color="gray", linestyle=":", alpha=0.5)

                current_t += t_ep

            # Draw the computed safe-threshold line
            plt.axhline(
                y=safe_threshold,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Safe Threshold ({percentile}th percentile): {safe_threshold:.4f}",
            )

            plt.title("Action Entropy Clustering: P Set vs C Set Across Episodes")
            plt.xlabel("Global Time Steps")
            plt.ylabel("Action Entropy")
            plt.legend(loc="upper right")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    return safe_threshold, p_size, total_frames
