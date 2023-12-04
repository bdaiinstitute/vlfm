# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import cv2
import numpy as np
import torch

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry
except ModuleNotFoundError:
    print("Could not import SAM. This is OK if you are not using it.")

from .vl_model import BaseVL


# TODO: only working as a model used directly not as a server at the moment because
# of passing the data
class SAM_crop(BaseVL):
    """Find masks uing SAM and then crop around them."""

    def __init__(
        self,
        feature_model: str = "BLIP2_unimodal",
        device: Optional[Any] = None,
        use_adapter: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        if feature_model == "BLIP2_unimodal":
            from .blip2_unimodal import BLIP2unimodal

            self.vl_model: BaseVL = BLIP2unimodal(
                device=device, use_adapter=use_adapter
            )
            self.feats_sz = [32, 256]
        elif feature_model == "CLIP":
            from .clip import CLIP

            self.vl_model = CLIP(device=device, use_adapter=use_adapter)
            self.feats_sz = [512]
        else:
            assert (
                False
            ), f"Invalid model {feature_model} (options are BLIP2_unimodal or CLIP)"

        self.device = device

        sam = sam_model_registry["vit_h"](checkpoint="data/sam_vit_h_4b8939.pth").to(
            device
        )
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def get_image_embedding(
        self, image: np.ndarray, head: str = ""
    ) -> torch.tensor:  # np.ndarray:
        final_im = torch.zeros(
            [image.shape[0], image.shape[1]] + self.feats_sz, device=self.device
        )

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = self.mask_generator.generate(img)

        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]

            segmentation = np.where(mask)
            if (
                len(segmentation) != 0
                and len(segmentation[1]) != 0
                and len(segmentation[0]) != 0
            ):
                x_min = int(np.min(segmentation[1]))
                x_max = int(np.max(segmentation[1]))
                y_min = int(np.min(segmentation[0]))
                y_max = int(np.max(segmentation[0]))
            else:
                continue

            if (y_min >= y_max) or (x_min >= x_max):
                continue

            mask_img = image[y_min:y_max, x_min:x_max, :]
            image_embedding_mask = self.vl_model.get_image_embedding(mask_img, head)

            final_im[mask] = image_embedding_mask.reshape([1] + self.feats_sz).repeat(
                [np.sum(mask[:])] + [1] * len(self.feats_sz)
            )

        if len(self.feats_sz) == 1:
            return final_im.permute(2, 0, 1)
        if len(self.feats_sz) == 1:
            return final_im.permute(2, 3, 0, 1).reshape(
                -1, image.shape[0], image.shape[1]
            )

    def get_text_embedding(
        self, txt: str, head: str = ""
    ) -> torch.tensor:  # np.ndarray:
        return self.vl_model.get_text_embedding(txt, head)

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

        return self.vl_model.get_similarity(image_embedding, txt_embedding)

    def get_similarity_batch(
        self, image_embeddings: torch.tensor, txt_embedding: torch.tensor
    ) -> np.ndarray:
        return self.vl_model.get_similarity_batch(image_embeddings, txt_embedding)
