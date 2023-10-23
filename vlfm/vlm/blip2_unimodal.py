# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from vlfm.vlm.server_wrapper import (
    ServerMixin,
    host_model,
    np_to_str,
    send_request,
    str_to_image,
    str_to_np,
)

try:
    from lavis.models import load_model_and_preprocess
except ModuleNotFoundError:
    print("Could not import lavis. This is OK if you are only using the client.")

from .vl_model import BaseVL


# TODO: only working as a model used directly not as a server at the moment because
# of passing the data
class BLIP2unimodal(BaseVL):
    """BLIP 2 image and text embedding seperate."""

    def __init__(
        self,
        name: str = "blip2_feature_extractor",
        model_type: str = "pretrain",
        device: Optional[Any] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.vis_processors, self.text_processors = (
            load_model_and_preprocess(
                name=name,
                model_type=model_type,
                is_eval=True,
                device=device,
            )
        )
        self.device = device

    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(image)
        img = self.vis_processors["eval"](pil_img).unsqueeze(0).to(self.device)

        sample = {"image": img, "text_input": ""}

        image_features = self.model.extract_features(sample, mode="image")

        return image_features.image_embeds_proj.squeeze().cpu().numpy()

    def get_text_embedding(self, txt: str) -> np.ndarray:
        sample = {"image": np.array([0]), "text_input": txt}

        text_features = self.model.extract_features(sample, mode="text")

        return text_features.text_embeds_proj.squeeze().cpu().numpy()

    def get_similarity(
        self, image_embedding: np.ndarray, txt_embedding: np.ndarray
    ) -> float:
        """
        Compute the cosine similarity between the image and the prompt (both already embedded).

        Args:
            image_embedding (numpy.ndarray): The input image embedded by BLIP-2.
            txt_embedding (numpy.ndarray): The text to compare the image to embedded by BLIP-2..

        Returns:
            float: The cosine similarity between the image and the prompt.
        """

        cosine = (image_embedding @ txt_embedding[0, :].T).max()

        return cosine

    def get_similarity_batch(
        self, image_embeddings: np.ndarray, txt_embedding: np.ndarray
    ) -> np.ndarray:
        cosine = (image_embeddings @ txt_embedding[0, :].T).max(axis=1)

        return cosine


class BLIP2unimodalClient(BaseVL):
    def __init__(self, port: int = 12182):
        self.url = f"http://localhost:{port}/VLmodel"

    def get_image_embedding(self, image: np.ndarray) -> np.ndarray:
        print(f" BLIP2unimodalClient.get_image_embedding: {image.shape}")
        response = send_request(self.url, funct="image_embed", image=image)
        return np.array(response["response"])

    def get_text_embedding(self, txt: str) -> np.ndarray:
        print(f" BLIP2unimodalClient.get_text_embedding: {txt}")
        response = send_request(self.url, funct="text_embed", txt=txt)
        return np.array(response["response"])

    def get_similarity(
        self, image_embedding: np.ndarray, txt_embedding: np.ndarray
    ) -> float:
        print(
            f" BLIP2unimodalClient.get_similarity: {image_embedding.shape},"
            f" {txt_embedding.shape}"
        )
        response = send_request(
            self.url,
            funct="similarity",
            image_embed=np_to_str(image_embedding),
            txt_embed=np_to_str(txt_embedding),
            embed_dtype=image_embedding.dtype,
            embed_image_shape=image_embedding.shape,
            embed_txt_shape=txt_embedding.shape,
        )
        return float(response["response"])

    def get_similarity_batch(
        self, image_embeddings: np.ndarray, txt_embedding: np.ndarray
    ) -> float:
        print(
            f" BLIP2unimodalClient.get_similarity_batch: {image_embeddings.shape},"
            f" {txt_embedding.shape}"
        )
        response = send_request(
            self.url,
            funct="similarity_batch",
            image_embed=np_to_str(image_embeddings),
            txt_embed=np_to_str(txt_embedding),
            embed_dtype=image_embeddings.dtype,
            embed_image_shape=image_embeddings.shape,
            embed_txt_shape=txt_embedding.shape,
        )
        return str_to_np(
            response["response"], arr_type=response["dtype"], shape=response["shape"]
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    args = parser.parse_args()

    print("Loading model...")

    class BLIP2unimodalServer(ServerMixin, BLIP2unimodal):
        def process_payload(self, payload: dict) -> dict:
            funct = payload["funct"]
            if funct == "image_embed":
                image = str_to_image(payload["image"])
                return {"response": self.get_image_embedding(image)}
            elif funct == "text_embed":
                return {"response": self.get_text_embedding(payload["txt"])}
            elif funct == "similarity" or funct == "similarity_batch":
                image_embed = str_to_np(
                    payload["image_embed"],
                    arr_type=payload["embed_dtype"],
                    shape=payload["embed_image_shape"],
                )
                text_embed = str_to_np(
                    payload["txt_embed"],
                    arr_type=payload["embed_dtype"],
                    shape=payload["embed_text_shape"],
                )
                if funct == "similarity":
                    return {"response": self.get_similarity(image_embed, text_embed)}
                elif funct == "similarity_batch":
                    res = self.get_similarity_batch(image_embed, text_embed)
                    return {
                        "response": np_to_str(res),
                        "shape": res.shape,
                        "dtype": res.dtype,
                    }
            raise Exception("Invalid function string for BLIP2unimodal")

    blip = BLIP2unimodalServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(blip, name="VLmodel", port=args.port)
