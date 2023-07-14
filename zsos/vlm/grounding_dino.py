import os
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

from groundingdino.util.inference import load_model, predict
from zsos.vlm.detections import ObjectDetections

GROUNDING_DINO_CONFIG = os.environ["GROUNDING_DINO_CONFIG"]
GROUNDING_DINO_WEIGHTS = os.environ["GROUNDING_DINO_WEIGHTS"]

if "CLASSES_PATH" in os.environ:
    with open(os.environ["CLASSES_PATH"]) as f:
        CLASSES = " . ".join(f.read().splitlines()) + " ."
else:
    print("[grounding_dino.py] WARNING: $CLASSES_PATH not set, using default classes")
    CLASSES = "chair . person . dog ."  # default classes


class GroundingDINO:
    def __init__(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        weights_path: str = GROUNDING_DINO_WEIGHTS,
        classes: str = CLASSES,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: torch.device = torch.device("cuda"),
    ):
        self.model = load_model(
            model_config_path=config_path, model_checkpoint_path=weights_path
        ).to(device)
        self.classes = classes
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def predict(
        self,
        image_tensor: torch.Tensor,
        image_numpy: Optional[np.ndarray] = None,
        visualize: bool = False,
    ) -> ObjectDetections:
        image_transformed = F.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image_tensor (torch.Tensor): The input image in the form of a tensor.
            image_numpy (Optional[np.ndarray]): Optionally provide the numpy version to
                use for the visualization.
            visualize (bool, optional): A flag indicating whether to visualize the
                output data. Defaults to False.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        if image_numpy is None:
            image_numpy = np.clip(
                image_tensor.permute(1, 2, 0).cpu().numpy() * 255, 0, 255
            )
            image_numpy = cv2.convertScaleAbs(image_numpy)
        det = ObjectDetections(
            boxes, logits, phrases, image_source=image_numpy, visualize=visualize
        )
        return det


if __name__ == "__main__":
    import argparse

    from server_wrapper import ServerMixin, host_model, str_to_image

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9040)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.predict(image, payload["phrase"]).to_json()}

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
