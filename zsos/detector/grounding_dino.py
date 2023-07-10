import os
from typing import List

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.ops import box_convert

from groundingdino.util.inference import annotate, load_model, predict

GROUNDING_DINO_CONFIG = os.environ["GROUNDING_DINO_CONFIG"]
GROUNDING_DINO_WEIGHTS = os.environ["GROUNDING_DINO_WEIGHTS"]

if "CLASSES_PATH" in os.environ:
    with open(os.environ["CLASSES_PATH"]) as f:
        CLASSES = " . ".join(f.read().splitlines()) + " ."
else:
    print("[grounding_dino.py] WARNING: $CLASSES_PATH not set, using default classes")
    CLASSES = "chair . person . dog ."  # default classes


class ObjectDetections:
    def __init__(
        self,
        image_source: np.ndarray,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        phrases: List[str],
        visualize: bool = False,
        fmt: str = "cxcywh",
    ):
        self.image_source = image_source
        self.boxes = box_convert(boxes=boxes, in_fmt=fmt, out_fmt="xyxy")
        self.logits = logits
        self.phrases = phrases
        if visualize:
            self.annotated_frame = annotate(
                image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
            )
        else:
            self.annotated_frame = None


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
        self, image_tensor: torch.Tensor, visualize: bool = False
    ) -> ObjectDetections:
        """
        :param image_tensor: an RGB tensor of shape (3, H, W) with values in [0, 1]
        :param visualize: whether to return an annotated image within the
            ObjectDetections object
        :return: ObjectDetections
        """
        image_transformed = F.normalize(
            image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        image_numpy = np.asarray(
            image_tensor.mul(255).permute(1, 2, 0).byte().cpu(), dtype=np.uint8
        )[:, :, ::-1]
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
