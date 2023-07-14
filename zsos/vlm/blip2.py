import base64

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from lavis.models import load_model_and_preprocess
from PIL import Image


class BLIP2:
    def __init__(
        self,
        name: str = "blip2_t5",
        model_type: str = "pretrain_flant5xxl",
        device: str = None,
    ):
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        self.device = device

    def ask(self, image, prompt=None):
        pil_img = Image.fromarray(image)
        processed_image = (
            self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)
        )

        import time

        st = time.time()
        if prompt is None:
            out = self.model.generate({"image": processed_image})
        else:
            out = self.model.generate({"image": processed_image, "prompt": prompt})
        print(f"Time taken: {time.time() - st:.2f}s")

        return out


app = Flask(__name__)
# blip = BLIP2(name="blip2_t5", model_type="pretrain_flant5xxl")
blip = BLIP2(name="blip2_opt", model_type="pretrain_opt2.7b")


@app.route("/blip2", methods=["POST"])
def blip2():
    # Get the JSON payload from the request
    payload = request.json

    # Extract the image and prompt from the payload
    image = payload.get("image")
    prompt = payload.get("prompt")

    # Convert the image to numpy format
    image = str_to_image(image)

    # Perform the task using the "blip" object
    result = blip.ask(image, prompt)

    # Return the response as JSON
    return jsonify({"result": result})


def image_to_str(img_np, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str):
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8070)
    args = parser.parse_args()

    print("Starting server...")
    app.run(host="localhost", port=args.port)
