# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

try:
    import clip
    from lseg_feature_extraction.additional_utils.models import LSeg_MultiEvalModule
    from lseg_feature_extraction.modules.lseg_module import LSegModule
except ModuleNotFoundError:
    print("Could not import stuff for LSeg, which is fine if it's not being used")
import numpy as np
import torch
import torchvision.transforms as transforms
from encoding.models.sseg import BaseNet

from vlfm.adapter.adapter import Adapter

from .vl_model import BaseVL


# TODO: only working as a model used directly not as a server at the moment because
# of passing the data
class LSeg(BaseVL):
    """LSeg image and text embedding seperate."""

    def __init__(
        self,
        device: Optional[Any] = None,
        use_adapter: bool = False,
    ) -> None:
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.device = device

        self.use_adapter = use_adapter

        if self.use_adapter:
            self.img_head = Adapter(512, orig_weight=0.2, blip=False).to(device)
            self.embed_head = Adapter(512, orig_weight=0.2, blip=False).to(device)

            self.text_embed_head = Adapter(512, orig_weight=0.8, blip=False).to(device)
            self.text_img_head = Adapter(512, orig_weight=0.8, blip=False).to(device)

            self.img_head.load_state_dict(torch.load("data/img_head.pth"))
            self.embed_head.load_state_dict(torch.load("data/embed_head.pth"))
            self.text_img_head.load_state_dict(torch.load("data/text_img_head.pth"))
            self.text_embed_head.load_state_dict(torch.load("data/text_embed_head.pth"))

            self.img_head.eval()
            self.embed_head.eval()
            self.text_img_head.eval()
            self.text_embed_head.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                transforms.Resize((256, 256)),
            ]
        )

        module = LSegModule.load_from_checkpoint(
            checkpoint_path="data/demo_e200.ckpt",
            data_path="datasets/",
            dataset="ade20k",
            backbone="clip_vitl16_384",
            aux=False,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=0,
            batch_size=1,
            max_epochs=0,
            ignore_index=255,
            dropout=0.0,
            scale_inv=False,
            augment=False,
            no_batchnorm=False,
            widehead=True,
            widehead_hr=False,
            map_locatin="cpu",
            arch_option=0,
            block_depth=0,
            activation="lrelu",
        )

        # model
        if isinstance(module.net, BaseNet):
            model = module.net
        else:
            model = module

        model = model.eval()
        model = model.cpu()
        # scales = (
        #     [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
        #     if args.dataset == "citys"
        #     else [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        # )
        scales = [1]

        model.mean = [0.5, 0.5, 0.5]
        model.std = [0.5, 0.5, 0.5]

        model.crop_size = 2 * 256
        model.base_size = 2 * 256

        self.evaluator = LSeg_MultiEvalModule(model, scales=scales, flip=True).cuda()
        self.evaluator.eval()

        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")

    def get_image_embedding(self, image: np.ndarray, head: str = "") -> torch.tensor:
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.evaluator.parallel_forward(image, "")
            image_features = outputs[0][0].half()

        if self.use_adapter:
            if head == "img":
                return self.img_head(image_features).squeeze()
            elif head == "embed":
                # return self.embed_head(image_features).squeeze()
                return image_features.squeeze()  # should do on path not single images!
            else:
                assert False, (
                    "If using adapter need to specify head when extracting img features"
                    " (img or embed)"
                )

        else:
            return image_features.squeeze()  # .cpu().numpy()

    def get_text_embedding(
        self, txt: str, head: str = ""
    ) -> torch.tensor:  # np.ndarray:
        with torch.no_grad():
            text_tokens = clip.tokenize(txt, context_length=77, truncate=True).to(
                self.device
            )  # tokenize
            text_features = self.clip_model.encode_text(
                text_tokens
            ).float()  # embed with text encoder

            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_features.to(self.device)

        if self.use_adapter:
            if head == "img":
                return self.text_img_head(text_features).squeeze()
            elif head == "embed":
                return self.text_embed_head(text_features).squeeze()
            else:
                assert False, (
                    "If using adapter need to specify head when extracting text"
                    " features (img or embed)"
                )

        else:
            return text_features.squeeze()  # .cpu().numpy()

    def get_similarity(
        self, image_embedding: torch.tensor, txt_embedding: torch.tensor
    ) -> float:
        """
        Compute the cosine similarity between the image and the prompt (both already embedded).

        Args:
            image_embedding (numpy.ndarray): The input image embedded by BLIP-2.
            txt_embedding (numpy.ndarray): The text to compare the image to embedded by BLIP-2..

        Returns:
            float: The cosine similarity between the image and the prompt.
        """

        # cosine = (image_embedding @ txt_embedding[0, :].T).max()
        cosine = (image_embedding @ txt_embedding.t()).item()

        return cosine

    def get_similarity_batch(
        self, image_embeddings: torch.tensor, txt_embedding: torch.tensor
    ) -> np.ndarray:
        # cosine = (image_embeddings @ txt_embedding[0, :].T).max(axis=1)

        cosine = image_embeddings @ txt_embedding.t()

        return cosine.cpu().numpy()
