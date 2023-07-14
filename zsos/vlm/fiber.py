import numpy as np
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_FIBER import GLIPDemo
from PIL import Image

from zsos.vlm.detections import ObjectDetections


class Fiber:
    def __init__(self, config_file: str, weights: str):
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

        self.fiber = GLIPDemo(cfg)

    def detect(
        self, image: np.ndarray, phrase: str, visualize: bool = False
    ) -> ObjectDetections:
        result = self.fiber.inference(image, phrase)
        # Normalize result.bbox to be between 0 and 1
        normalized_bbox = result.bbox / torch.tensor(
            [image.shape[1], image.shape[0], image.shape[1], image.shape[0]]
        )

        dets = ObjectDetections(
            image_source=image,
            boxes=normalized_bbox,
            logits=result.extra_fields["scores"],
            phrases=[phrase],
            visualize=visualize,
            fmt="xyxy",
        )

        return dets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing FIBER phrase grounding model")
    parser.add_argument("config_file", metavar="FILE", help="path to config file")
    parser.add_argument("weights_file", metavar="FILE", help="path to weights file")
    parser.add_argument("image_path", metavar="FILE", help="path to image file")
    parser.add_argument("phrase", metavar="FILE", help="phrase to ground")
    args = parser.parse_args()

    fiber = Fiber(args.config_file, args.weights_file)
    image = np.array(Image.open(args.image_path))
    dets = fiber.detect(image, args.phrase, visualize=True)

    # Save the image from dets
    Image.fromarray(dets.annotated_frame).save("test.png")
