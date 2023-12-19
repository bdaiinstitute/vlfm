# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_FIBER import GLIPDemo

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

DEFAULT_CONFIG = "FIBER/fine_grained/configs/refcocog.yaml"
DEFAULT_WEIGHTS = "FIBER/fine_grained/models/fiber_refcocog.pth"


class FIBER:
    def __init__(self, config_file: str = DEFAULT_CONFIG, weights: str = DEFAULT_WEIGHTS):
        cfg.merge_from_file(config_file)
        cfg.num_gpus = 1
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.TEST.IMS_PER_BATCH = 1
        cfg.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = -1
        cfg.TEST.EVAL_TASK = "grounding"
        cfg.MODEL.ATSS.PRE_NMS_TOP_N = 3000
        cfg.MODEL.ATSS.DETECTIONS_PER_IMG = 100
        cfg.MODEL.ATSS.INFERENCE_TH = 0.0
        cfg.MODEL.WEIGHT = weights

        cfg.freeze()

        self.fiber = GLIPDemo(cfg, confidence_threshold=0.2)

    def detect(self, image: np.ndarray, phrase: str, visualize: bool = False) -> ObjectDetections:
        """
        Given an image and a phrase, this function detects the presence of the most
        suitable object described by the phrase in the image. The output object's
        bounding boxes are normalized between 0 and 1.

        The coordinates are provided in "xyxy" format (top-left = x0, y0 and
        bottom-right = x1, y1).

        Arguments:
            image (np.ndarray): The input image in which to detect objects.
            phrase (str): The phrase describing the object(s) to detect.
            visualize (bool, optional): If True, visualizes the detections on the image.
                Defaults to False.

        Returns:
            ObjectDetections: A data structure containing the detection results,
                including the source image, the normalized bounding boxes of detected
                objects, the prediction scores (logits), the phrase used for the
                detection, and flag indicating if visualization is enabled.
        """
        result = self.fiber.inference(image, phrase)
        # Normalize result.bbox to be between 0 and 1
        normalized_bbox = result.bbox / torch.tensor([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        dets = ObjectDetections(
            image_source=image,
            boxes=normalized_bbox,
            logits=result.extra_fields["scores"],
            phrases=[phrase],
            fmt="xyxy",
        )

        return dets


class FIBERClient:
    def __init__(self, url: str = "http://localhost:9080/fiber"):
        self.url = url

    def detect(self, image: np.ndarray, phrase: str, visualize: bool = False) -> ObjectDetections:
        response = send_request(self.url, image=image, phrase=phrase)["response"]
        detections = ObjectDetections.from_json(response, image_source=image)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9080)
    args = parser.parse_args()

    print("Loading model...")

    class FIBERServer(ServerMixin, FIBER):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            return {"response": self.detect(image, payload["phrase"]).to_json()}

    fiber = FIBERServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(fiber, name="fiber", port=args.port)
