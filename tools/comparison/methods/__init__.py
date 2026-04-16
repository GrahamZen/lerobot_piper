from tools.comparison.methods.action_entropy import ActionEntropyDetector
from tools.comparison.methods.embedding_extractor import EmbeddingExtractor
from tools.comparison.methods.logpzo import LogpZODetector
from tools.comparison.methods.pca_kmeans import PCAKMeansDetector
from tools.comparison.methods.rnd import RNDDetector
from tools.comparison.methods.similarity import SimilarityDetector
from tools.comparison.methods.stac import STACDetector

DETECTOR_REGISTRY: dict[str, type] = {
    "action_entropy": ActionEntropyDetector,
    "rnd": RNDDetector,
    "similarity": SimilarityDetector,
    "pca_kmeans": PCAKMeansDetector,
    "stac": STACDetector,
    "logpzo": LogpZODetector,
}

__all__ = [
    "ActionEntropyDetector",
    "RNDDetector",
    "SimilarityDetector",
    "PCAKMeansDetector",
    "EmbeddingExtractor",
    "STACDetector",
    "LogpZODetector",
    "DETECTOR_REGISTRY",
]
