import os
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

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
        det = ObjectDetections(image_numpy, boxes, logits, phrases, visualize=visualize)
        return det


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GroundingDINO Demo")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    model = GroundingDINO()

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    detections = model.predict(image_tensor, visualize=True)

    # Do something with the detections or annotated image
    print("detections.boxes")
    print(detections.boxes)
    print("detections.logits")
    print(detections.logits)
    print("detections.phrases")
    print(detections.phrases)

    # Save the annotated image
    Image.fromarray(detections.annotated_frame).save("annotated_image.jpg")
    print("Saved annotated image to annotated_image.jpg")
