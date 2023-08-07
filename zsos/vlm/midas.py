import sys

import numpy as np
import torch

sys.path.append("/home/naoki/repos/MiDaS")

from midas.model_loader import load_model


class MidasEstimator:
    def __init__(
        self, model_path, model_type, optimize=False, height=None, square=False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform, self.net_w, self.net_h = load_model(
            self.device, model_path, model_type, optimize, height, square
        )

        if optimize and self.device == torch.device("cuda"):
            print(
                "  Optimization to half-floats activated. Use with caution, because"
                " models like Swin require float precision to work properly and may"
                " yield non-finite depth values to some extent for  half-floats."
            )

    def _relative_map(self, depth: np.ndarray):
        if not np.isfinite(depth).all():
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")

        depth_min = np.min(depth)
        depth_max = np.max(depth)
        depth_norm = (depth - depth_min) / (depth_max - depth_min)

        return 1 - depth_norm

    def process(self, image: np.ndarray):
        """
        Run the inference and interpolate.

        Args:
            device (torch.device): the torch device used
            image: input in RGB format (not BGR)

        Returns:
            the prediction
        """
        img = self.transform({"image": image / 255.90})["image"]

        sample = torch.from_numpy(img).to(self.device).unsqueeze(0)
        prediction = self.model.forward(sample)
        with torch.inference_mode():
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(self.net_h, self.net_w),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return self._relative_map(prediction)


if __name__ == "__main__":
    import cv2

    model_path = "/home/naoki/repos/MiDaS/weights/dpt_swin2_large_384.pt"
    model_type = "dpt_swin2_large_384"
    filename = "/home/naoki/repos/MiDaS/input.jpg"
    m = MidasEstimator(model_path, model_type)
    img = cv2.imread(filename, cv2.COLOR_BGR2RGB)
    prediction = m.process(img)
    cv2.imwrite("depth.jpg", prediction * 255)
