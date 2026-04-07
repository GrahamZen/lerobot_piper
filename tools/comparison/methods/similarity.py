"""Mahalanobis-distance OOD detector on policy encoder embeddings.

Follows the FIPER "embedding similarity" method (2025).  During calibration
the detector is fitted on all successful-episode embeddings; at inference it
computes the Mahalanobis distance of each step's embedding to the fitted
Gaussian distribution.

PCA is applied first (when sklearn is available and emb_dim > n_pca_components)
to avoid ill-conditioned covariance matrices.

Unlike RND, no training is required — the model is fitted entirely from
calibration embeddings at the start of ``compute_comparison.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

if TYPE_CHECKING:
    from sklearn.decomposition import PCA

from tools.comparison.methods.base import BaseDetector
from tools.comparison.methods.embedding_extractor import EmbeddingExtractor


class SimilarityDetector(BaseDetector):
    """Mahalanobis distance to in-distribution embeddings.

    Parameters
    ----------
    extractor:
        Shared :class:`EmbeddingExtractor` attached to the same policy.
    n_pca_components:
        Target dimensionality after PCA.  Skipped if the embedding is already
        smaller.  Set to ``None`` to disable PCA entirely.
    """

    name = "similarity"

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        n_pca_components: int = 32,
    ) -> None:
        self.extractor = extractor
        self.n_pca = n_pca_components
        self._pca: PCA | None = None
        self._mean: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._fitted = False
        self._history: list[float] = []

    # ------------------------------------------------------------------
    # Fitting (call once before any update)
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit the Gaussian distribution from calibration embeddings.

        Args:
            embeddings: ``(N, D)`` array of in-distribution embeddings.
        """
        embs = embeddings.astype(np.float64)

        if self.n_pca is not None and embs.shape[1] > self.n_pca:
            try:
                from sklearn.decomposition import PCA

                self._pca = PCA(n_components=self.n_pca)
                embs = self._pca.fit_transform(embs)
            except ImportError:
                print("[SimilarityDetector] sklearn not found — skipping PCA.")

        self._mean = embs.mean(axis=0)
        # Tikhonov regularisation avoids singular covariance
        cov = np.cov(embs.T) + 1e-6 * np.eye(embs.shape[1])
        self._inv_cov = np.linalg.inv(cov)
        self._fitted = True
        print(
            f"[SimilarityDetector] Fitted on {len(embeddings)} embeddings "
            f"(raw_dim={embeddings.shape[1]}, fit_dim={embs.shape[1]})"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibration_score(self) -> float:
        """Episode maximum Mahalanobis distance."""
        return max(self._history) if self._history else 0.0

    def update(self, step: int, chunks: Tensor | None) -> float:
        if not self._fitted:
            return 0.0

        emb = self.extractor.last
        if emb is None:
            self._history.append(0.0)
            return 0.0

        e = emb.astype(np.float64)
        if self._pca is not None:
            e = self._pca.transform(e.reshape(1, -1))[0]

        diff = e - self._mean
        dist = float(np.sqrt(max(float(diff @ self._inv_cov @ diff), 0.0)))
        self._history.append(dist)
        return dist
