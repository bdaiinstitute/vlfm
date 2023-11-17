# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

from .vl_model import BaseVL


# TODO: only working as a model used directly not as a server at the moment because
# of passing the data
class CLIP(BaseVL):
    """CLIP image and text embedding seperate."""

    def __init__(
        self,
        name: str = "clip_feature_extractor",
        model_type: str = "ViT-B-16",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.device = device

    def get_image_embedding(self, image: np.ndarray) -> torch.tensor:  # np.ndarray:
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        sample = {"image": img, "text_input": None}

        image_features = self.model.extract_features(sample)

        return image_features.squeeze()  # .cpu().numpy()

    def get_text_embedding(self, txt: str) -> torch.tensor:  # np.ndarray:
        sample = {"image": None, "text_input": txt}

        text_features = self.model.extract_features(sample)

        return text_features.squeeze()  # .cpu().numpy()

    def get_similarity(
        self, image_embedding: torch.tensor, txt_embedding: torch.tensor
    ) -> float:
        """
        Compute the cosine similarity between the image and the prompt (both already embedded).

        Args:
            image_embedding (numpy.ndarray): The input image embedded by BLIP-2.
            txt_embedding (numpy.ndarray): The text to compare the image to embedded by BLIP-2..

        Returns:
            float: The cosine similarity between the image and the prompt.
        """

        # cosine = (image_embedding @ txt_embedding[0, :].T).max()
        cosine = (image_embedding @ txt_embedding.t()).item()

        return cosine

    def get_similarity_batch(
        self, image_embeddings: torch.tensor, txt_embedding: torch.tensor
    ) -> np.ndarray:
        # cosine = (image_embeddings @ txt_embedding[0, :].T).max(axis=1)

        cosine = image_embeddings @ txt_embedding.t()

        return cosine.cpu().numpy()
