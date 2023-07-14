import base64
from typing import Any

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


class ServerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000):
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)

    @app.route(f"/{name}", methods=["POST"])
    def process_request():
        payload = request.json
        return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)


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


def send_request(url, **kwargs):
    # Create a payload dict which is a clone of kwargs but all np.array values are
    # converted to strings
    payload = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
        else:
            payload[k] = v

    # Set the headers
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload)

    if resp.status_code == 200:
        result = resp.json()
    else:
        raise Exception("Request failed")

    return result
