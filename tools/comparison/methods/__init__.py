from tools.comparison.methods.embedding_extractor import EmbeddingExtractor
from tools.comparison.methods.pca_kmeans import PCAKMeansDetector
from tools.comparison.methods.rnd import RNDDetector
from tools.comparison.methods.similarity import SimilarityDetector
from tools.comparison.methods.stac import STACDetector

DETECTOR_REGISTRY: dict[str, type] = {
    "rnd": RNDDetector,
    "similarity": SimilarityDetector,
    "pca_kmeans": PCAKMeansDetector,
    "stac": STACDetector,
}

__all__ = [
    "RNDDetector",
    "SimilarityDetector",
    "PCAKMeansDetector",
    "EmbeddingExtractor",
    "STACDetector",
    "DETECTOR_REGISTRY",
]
