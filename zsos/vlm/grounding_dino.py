import os

import numpy as np
import torch
import torchvision.transforms.functional as F

from groundingdino.util.inference import load_model, predict
from zsos.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

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

    def predict(self, image: np.ndarray, visualize: bool = False) -> ObjectDetections:
        """
        This function makes predictions on an input image tensor or numpy array using a
        pretrained model.

        Arguments:
            image (np.ndarray): An image in the form of a numpy array.
            visualize (bool, optional): A flag indicating whether to visualize the
                output data. Defaults to False.

        Returns:
            ObjectDetections: An instance of the ObjectDetections class containing the
                object detections.
        """
        # Convert image to tensor and normalize from 0-255 to 0-1
        image_tensor = F.to_tensor(image)
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
        detections = ObjectDetections(
            boxes, logits, phrases, image_source=image, visualize=visualize
        )

        classes = self.classes.split(" . ")
        keep = torch.tensor(
            [p in classes for p in detections.phrases], dtype=torch.bool
        )

        detections.boxes = detections.boxes[keep]
        detections.logits = detections.logits[keep]
        detections.phrases = [p for i, p in enumerate(detections.phrases) if keep[i]]

        return detections


class GroundingDINOClient:
    def __init__(
        self, url: str = "http://localhost:9040/gdino", classes: str = CLASSES
    ):
        self.url = url
        self.classes = classes

    def predict(
        self, image_numpy: np.ndarray, visualize: bool = False
    ) -> ObjectDetections:
        response = send_request(self.url, image=image_numpy)
        detections = ObjectDetections.from_json(
            response, image_source=image_numpy, visualize=visualize
        )

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9040)
    args = parser.parse_args()

    print("Loading model...")

    class GroundingDINOServer(ServerMixin, GroundingDINO):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return self.predict(image).to_json()

    gdino = GroundingDINOServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(gdino, name="gdino", port=args.port)
