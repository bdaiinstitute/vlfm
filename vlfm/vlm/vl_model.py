# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

import numpy as np


class BaseVL:
    """BLIP 2 image and text embedding seperate."""

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_text_embedding(self, txt: str) -> np.ndarray:
        raise NotImplementedError

    def get_similarity(
        self, image_embedding: np.ndarray, txt_embedding: np.ndarray
    ) -> float:
        raise NotImplementedError

    def get_similarity_batch(
        self, image_embeddings: np.ndarray, txt_embedding: np.ndarray
    ) -> float:
        raise NotImplementedError
