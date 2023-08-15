import os
from typing import List

import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry

from .server_wrapper import (
    ServerMixin,
    bool_arr_to_str,
    host_model,
    send_request,
    str_to_bool_arr,
    str_to_image,
)


class MobileSAM:
    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_t",
        device: str = None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.device = device

        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.to(device=device)
        mobile_sam.eval()
        self.predictor = SamPredictor(mobile_sam)

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Segments the object in the given bounding box from the image.

        Args:
            image (numpy.ndarray): The input image as a numpy array.
            bbox (List[int]): The bounding box as a numpy array in the
                format [x1, y1, x2, y2].

        Returns:
            np.ndarray: The segmented object as a numpy array (boolean mask). The mask
                is the same size as the bbox, cropped out of the image.

        """
        self.predictor.set_image(image)
        masks, _, _ = self.predictor.predict(box=np.array(bbox), multimask_output=False)
        cropped_mask = masks[0][bbox[1] : bbox[3], bbox[0] : bbox[2]]

        return cropped_mask


class MobileSAMClient:
    def __init__(self, url: str = "http://localhost:8767/mobile_sam"):
        self.url = url

    def segment_bbox(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        response = send_request(self.url, image=image, bbox=bbox)
        cropped_mask_str = response["cropped_mask"]

        shape = (bbox[3] - bbox[1], bbox[2] - bbox[0])
        cropped_mask = str_to_bool_arr(cropped_mask_str, shape=shape)

        return cropped_mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    print("Loading model...")

    class MobileSAMServer(ServerMixin, MobileSAM):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            cropped_mask = self.segment_bbox(image, payload["bbox"])
            cropped_mask_str = bool_arr_to_str(cropped_mask)
            return {"cropped_mask": cropped_mask_str}

    mobile_sam = MobileSAMServer(sam_checkpoint=os.environ["MOBILE_SAM_CHECKPOINT"])
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(mobile_sam, name="mobile_sam", port=args.port)