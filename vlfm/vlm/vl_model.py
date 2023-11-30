# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

import numpy as np
import torch


class BaseVL:
    """image and text embedding seperate."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def get_image_embedding(self, image: np.ndarray, head: str = "") -> torch.tensor:
        raise NotImplementedError

    def get_text_embedding(self, txt: str, head: str = "") -> torch.tensor:
        raise NotImplementedError

    def get_similarity(
        self, image_embedding: torch.tensor, txt_embedding: torch.tensor
    ) -> float:
        raise NotImplementedError

    def get_similarity_batch(
        self, image_embeddings: torch.tensor, txt_embedding: torch.tensor
    ) -> np.ndarray:
        raise NotImplementedError
