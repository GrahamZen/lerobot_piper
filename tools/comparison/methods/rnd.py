"""Random Network Distillation (RND) OOD detector on policy encoder embeddings.

Reference:
    Burda et al., "Exploration by Random Network Distillation" (2018).
    Applied here for OOD detection following FIPER (2025).

The detector requires a pre-trained RND checkpoint produced by
``tools/comparison/train_rnd.py``.

OOD score at each step: ``||predictor(emb) - target(emb)||²``
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from tools.comparison.methods.base import BaseDetector
from tools.comparison.methods.embedding_extractor import EmbeddingExtractor


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class RNDDetector(BaseDetector):
    """OOD detector using Random Network Distillation on encoder embeddings.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` file produced by ``train_rnd.py``.
    extractor:
        Shared :class:`EmbeddingExtractor` attached to the same policy.
    device:
        Torch device for the RND networks.
    """

    name = "rnd"

    def __init__(
        self,
        checkpoint_path: str | Path,
        extractor: EmbeddingExtractor,
        device: torch.device | str = "cpu",
    ) -> None:
        self.extractor = extractor
        self.device = torch.device(device)
        self._history: list[float] = []

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        in_dim: int = ckpt["in_dim"]
        hidden_dim: int = ckpt.get("hidden_dim", 1024)
        out_dim: int = ckpt.get("out_dim", 512)

        self.target = _MLP(in_dim, hidden_dim, out_dim).to(self.device)
        self.predictor = _MLP(in_dim, hidden_dim, out_dim).to(self.device)
        self.target.load_state_dict(ckpt["target"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.target.eval()
        self.predictor.eval()

        print(f"[RNDDetector] Loaded from {checkpoint_path}  (in_dim={in_dim})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._history.clear()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibration_score(self) -> float:
        """Episode maximum RND error (used for threshold calibration)."""
        return max(self._history) if self._history else 0.0

    def update(self, step: int, chunks: Tensor | None) -> float:
        emb = self.extractor.last
        if emb is None:
            self._history.append(0.0)
            return 0.0

        x = torch.tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            t = self.target(x)
            p = self.predictor(x)
        score = float((p - t).pow(2).mean())
        self._history.append(score)
        return score
