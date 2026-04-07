"""PCA + K-means OOD detector on policy encoder embeddings.

Follows FAIL-Detect (RSS 2025) ``PCAKMeansNet`` exactly:
    1. PCA reduces observation embeddings to ``emb_dim`` components.
    2. K-means (64 centroids by default) is fit on the PCA-reduced embeddings.
    3. OOD score = minimum Euclidean distance to any centroid.

Unlike :class:`~tools.comparison.methods.similarity.SimilarityDetector`
(which uses Mahalanobis distance to a single Gaussian), this is non-parametric
and handles multi-modal trajectory distributions.

Fitting is done once from calibration embeddings (no gradient training needed).
The fitted model can be saved/loaded as a ``.pt`` file for reuse.

Usage in ``compute_comparison.py``::

    from tools.comparison.methods.pca_kmeans import PCAKMeansDetector

    detector = PCAKMeansDetector(extractor=extractor, emb_dim=55, n_clusters=64)
    # fit on calibration embeddings (N, D):
    detector.fit(calibration_embeddings)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from tools.comparison.methods.base import BaseDetector
from tools.comparison.methods.embedding_extractor import EmbeddingExtractor

# ---------------------------------------------------------------------------
# PCAKMeansNet — mirrors FAIL-Detect's net_PCA.py
# ---------------------------------------------------------------------------


class PCAKMeansNet(nn.Module):
    """PCA + K-means distance network.

    Stores PCA components and K-means centroids as non-trainable buffers so
    the model can be saved/loaded with ``torch.save`` / ``torch.load`` and
    run entirely on GPU if needed.

    Args:
        emb_dim   : Target PCA dimensionality.
        n_clusters: Number of K-means centroids (default 64).
    """

    def __init__(self, emb_dim: int, n_clusters: int = 64) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.n_clusters = n_clusters
        # Buffers are registered after fitting; declare as None placeholders.
        self._fitted = False

    def fit(self, x: np.ndarray) -> None:
        """Fit PCA and K-means on calibration embeddings.

        Args:
            x: ``(N, input_dim)`` float32/float64 array of encoder embeddings
               from **successful** episodes.
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        pca = PCA(n_components=self.emb_dim, svd_solver="full")
        x_enc = pca.fit_transform(x).astype(np.float32)
        print(f"[PCAKMeansNet] PCA: {x.shape[1]} → {self.emb_dim} dims")

        km = KMeans(n_clusters=self.n_clusters, n_init="auto")
        km.fit(x_enc)
        print(f"[PCAKMeansNet] K-means: {self.n_clusters} centroids fitted on {len(x)} samples")

        # Store as buffers (torch tensors, device-portable)
        self.register_buffer(
            "pca_components",
            torch.tensor(pca.components_, dtype=torch.float32),  # (emb_dim, input_dim)
        )
        self.register_buffer(
            "centroids",
            torch.tensor(km.cluster_centers_, dtype=torch.float32),  # (n_clusters, emb_dim)
        )
        self._fitted = True

    def forward(self, x: Tensor) -> Tensor:
        """Compute min-distance-to-centroid scores.

        Args:
            x: ``(B, input_dim)`` observation embeddings.

        Returns:
            ``(B,)`` minimum Euclidean distance to nearest centroid.
        """
        assert self._fitted, "Call fit() before forward()."
        x_enc = nn.functional.linear(x.float(), self.pca_components)  # (B, emb_dim)
        distances = torch.cdist(x_enc, self.centroids)  # (B, n_clusters)
        return distances.min(dim=1).values  # (B,)


# ---------------------------------------------------------------------------
# PCAKMeansDetector
# ---------------------------------------------------------------------------


class PCAKMeansDetector(BaseDetector):
    """OOD detector based on PCA + K-means min-distance on encoder embeddings.

    Mirrors FAIL-Detect's ``PCAKMeansNet`` and ``PCA_kmeans_UQ`` exactly.

    Parameters
    ----------
    extractor:
        Shared :class:`EmbeddingExtractor` attached to the policy.
    emb_dim:
        PCA target dimensionality.  FAIL-Detect uses task-specific values
        (e.g. 55 for square, 120 for transport).  A reasonable default for
        ACT is ``min(32, obs_dim)``.
    n_clusters:
        Number of K-means centroids (default 64, same as FAIL-Detect).
    device:
        Torch device for the PCAKMeansNet.
    checkpoint_path:
        Optional path to a pre-fitted ``.pt`` file.  If provided, ``fit()``
        is not required.
    """

    name = "pca_kmeans"

    def __init__(
        self,
        extractor: EmbeddingExtractor,
        emb_dim: int = 32,
        n_clusters: int = 64,
        device: torch.device | str = "cpu",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.extractor = extractor
        self.device = torch.device(device)
        self._history: list[float] = []

        self.model = PCAKMeansNet(emb_dim=emb_dim, n_clusters=n_clusters)

        if checkpoint_path is not None:
            self._load(Path(checkpoint_path))

    # ------------------------------------------------------------------
    # Fitting / persistence
    # ------------------------------------------------------------------

    def fit(self, embeddings: np.ndarray) -> None:
        """Fit PCA + K-means from calibration embeddings.

        Args:
            embeddings: ``(N, D)`` array of in-distribution embeddings.
        """
        self.model.fit(embeddings)
        self.model.to(self.device)
        self.model.eval()

    def save(self, path: Path) -> None:
        """Save the fitted model to a ``.pt`` file."""
        assert self.model._fitted, "Model must be fitted before saving."
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "emb_dim": self.model.emb_dim,
                "n_clusters": self.model.n_clusters,
                "state_dict": self.model.state_dict(),
            },
            path,
        )
        print(f"[PCAKMeansDetector] Saved → {path}")

    def _load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)  # nosec B614
        self.model = PCAKMeansNet(
            emb_dim=ckpt["emb_dim"],
            n_clusters=ckpt["n_clusters"],
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model._fitted = True
        self.model.to(self.device).eval()
        print(
            f"[PCAKMeansDetector] Loaded from {path} "
            f"(emb_dim={ckpt['emb_dim']}, n_clusters={ckpt['n_clusters']})"
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
        """Episode maximum score (used for CP band calibration)."""
        return max(self._history) if self._history else 0.0

    def update(self, step: int, chunks: Tensor | None) -> float:
        emb = self.extractor.last
        if emb is None or not self.model._fitted:
            self._history.append(0.0)
            return 0.0

        x = torch.tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            score = float(self.model(x).item())
        self._history.append(score)
        return score
