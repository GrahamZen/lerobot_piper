"""logpZO failure detector — from FAIL-Detect (RSS 2025).

Reference:
    "FAIL-Detect: Failure Detection for Robot Manipulation via
    Uncertainty Quantification" (RSS 2025).

Algorithm
---------
A Continuous Flow Matching (CFM) model is trained on encoder embeddings
from successful trajectories only.  At inference time the model predicts
the velocity field that pushes the current observation toward N(0,I).
The score is the squared L2 norm of the "pushed" observation:

    score_t = ||obs_emb + flow(obs_emb, t=0)||²

High score → OOD observation → likely failure.

The flow model is a ConditionalUnet1D operating on the embedding reshaped
into a 1-D sequence of action-dim-width tokens (the ``adjust_xshape``
trick from FAIL-Detect).

Usage
-----
Requires a pre-trained checkpoint produced by
``compute_comparison.py`` via ``_train_logpzo_checkpoint()``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from tools.comparison.methods.base import BaseDetector
from tools.comparison.methods.embedding_extractor import EmbeddingExtractor
from tools.comparison.methods.unet.conditional_unet1d import ConditionalUnet1D


def _adjust_xshape(x: Tensor, in_dim: int) -> Tensor:
    """Reshape flat embedding (B, D) → (B, seq_len, in_dim).

    Pads D to the nearest multiple of in_dim, then pads seq_len to the
    nearest multiple of 4 (required by the UNet's downsampling path).
    Mirrors ``data_loader.adjust_xshape`` from FAIL-Detect exactly.
    """
    total_dim = x.shape[1]
    remain = total_dim % in_dim
    if remain > 0:
        pad = in_dim - remain
        x = torch.cat([x, x.new_zeros(x.shape[0], pad)], dim=1)
        total_dim += pad
    seq_len = total_dim // in_dim
    if seq_len % 4 != 0:
        extra = (4 - seq_len % 4) * in_dim
        x = torch.cat([x, x.new_zeros(x.shape[0], extra)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)


def build_flow_model(in_dim: int) -> ConditionalUnet1D:
    """Build the ConditionalUnet1D flow model (same config as FAIL-Detect)."""
    return ConditionalUnet1D(
        input_dim=in_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False,
    )


class LogpZODetector(BaseDetector):
    """OOD detector using a flow-matching density estimator on encoder embeddings.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` file produced by ``_train_logpzo_checkpoint``.
    extractor:
        Shared :class:`EmbeddingExtractor` attached to the policy.
    device:
        Torch device for the flow model.
    in_dim:
        Width of each token when the embedding is reshaped into a 1-D
        sequence.  Should match ``action_dim`` (default 14 for dual-arm).

    Notes
    -----
    The checkpoint stores ``emb_mean`` and ``emb_std`` computed from the
    calibration embeddings.  At inference embeddings are z-score normalised
    before being fed to the flow model, which is essential because ACT encoder
    features are not naturally near N(0,I) — without normalisation the flow
    model cannot learn a useful density and TPR collapses to 0.
    """

    name = "logpzo"

    def __init__(
        self,
        checkpoint_path: str | Path,
        extractor: EmbeddingExtractor,
        device: torch.device | str = "cpu",
        in_dim: int = 14,
    ) -> None:
        self.extractor = extractor
        self.device = torch.device(device)
        self.in_dim = in_dim
        self._history: list[float] = []

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.net = build_flow_model(in_dim).to(self.device)
        self.net.load_state_dict(ckpt["model"])
        self.net.eval()

        # z-score normalisation stats (computed during training)
        emb_mean = ckpt.get("emb_mean")
        emb_std = ckpt.get("emb_std")
        if emb_mean is not None and emb_std is not None:
            self._emb_mean = torch.tensor(emb_mean, dtype=torch.float32, device=self.device)
            self._emb_std = torch.tensor(emb_std, dtype=torch.float32, device=self.device).clamp(min=1e-6)
        else:
            self._emb_mean = None
            self._emb_std = None
        print(
            f"[LogpZODetector] Loaded from {checkpoint_path}  (in_dim={in_dim}, normalised={emb_mean is not None})"
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
        """Episode maximum logpZO score (used for threshold calibration)."""
        return max(self._history) if self._history else 0.0

    def update(self, step: int, chunks: Tensor | None) -> float:
        emb = self.extractor.last
        if emb is None:
            self._history.append(0.0)
            return 0.0

        x = torch.tensor(emb, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, D)

        # z-score normalise using calibration statistics
        if self._emb_mean is not None and self._emb_std is not None:
            x = (x - self._emb_mean) / self._emb_std

        x_seq = _adjust_xshape(x, self.in_dim)  # (1, seq_len, in_dim)

        with torch.no_grad():
            t_zero = torch.zeros(1, device=self.device)
            pred_v = self.net(x_seq, t_zero)  # (1, seq_len, in_dim)
            pushed = x_seq + pred_v  # push toward N(0,I)
            score = float(pushed.reshape(1, -1).pow(2).sum())  # ||pushed||²

        self._history.append(score)
        return score
