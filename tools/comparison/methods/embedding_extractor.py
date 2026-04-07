"""Hook-based extractor for ACT encoder embeddings.

Registers a forward hook on ``policy.model.encoder``.  The encoder output
has shape ``(seq_len, B, D)`` where index 0 is the latent token (zeros at
inference).  We skip it and mean-pool the rest → ``(D,)`` numpy array.
"""

from __future__ import annotations

import numpy as np


class EmbeddingExtractor:
    """Captures encoder embeddings via a forward hook.

    Usage::

        extractor = EmbeddingExtractor()
        extractor.attach(policy)
        policy.predict_action_chunk(batch)  # hook fires automatically
        emb = extractor.last  # np.ndarray (D,)
    """

    def __init__(self) -> None:
        self._embedding: np.ndarray | None = None
        self._batch_embedding: np.ndarray | None = None
        self._state: np.ndarray | None = None
        self._handle = None

    def attach(self, policy) -> None:
        """Register a forward hook on ``policy.model.encoder``."""
        model = getattr(policy, "model", None)
        encoder = getattr(model, "encoder", None) if model else None
        if encoder is None:
            raise RuntimeError("policy has no model.encoder — cannot attach EmbeddingExtractor.")

        def _hook(_module, _input, output):
            # output: (seq_len, B, D) — index 0 is the latent token
            emb = output[1:].mean(dim=0)  # (B, D)
            self._batch_embedding = emb.detach().cpu().float().numpy()  # (B, D)
            self._embedding = self._batch_embedding[0]  # (D,) — backward compat

        self._handle = encoder.register_forward_hook(_hook)

    def detach(self) -> None:
        """Remove the hook (call when done to avoid memory leaks)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def last(self) -> np.ndarray | None:
        """The embedding captured during the most recent forward pass (single frame, shape ``(D,)``)."""
        return self._embedding

    @property
    def last_batch(self) -> np.ndarray | None:
        """All embeddings from the most recent batched forward pass, shape ``(B, D)``."""
        return self._batch_embedding

    def set_embedding(self, emb: np.ndarray) -> None:
        """Manually inject an embedding (used for replaying calibration data)."""
        self._embedding = emb

    @property
    def last_state(self) -> np.ndarray | None:
        """Proprioceptive state set via :meth:`set_state`."""
        return self._state

    def set_state(self, state: np.ndarray) -> None:
        """Store proprioceptive state for the current frame."""
        self._state = state
