import logging

import torch

logger = logging.getLogger(__name__)


class VLMService:
    def __init__(self):
        pass

    def select_checkpoint_index(
        self,
        batch: dict[str, torch.Tensor],
        checkpoint_indices: list[int],
        episode: int,
        step: int,
    ) -> int | None:
        return None
