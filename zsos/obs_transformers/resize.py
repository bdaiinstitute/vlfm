import copy
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from gym import spaces
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import (
    ObsTransformConfig,
)
from habitat_baselines.utils.common import (
    get_image_height_width,
    overwrite_gym_box_shape,
)
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor


@baseline_registry.register_obs_transformer()
class Resize(ObservationTransformer):
    def __init__(
        self,
        size: Tuple[int, int],
        channels_last: bool = True,
        trans_keys: Tuple[str, ...] = ("rgb", "depth", "semantic"),
        semantic_key: str = "semantic",
    ):
        """Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension
        """
        super(Resize, self).__init__()
        self._size: Tuple[int, int] = size
        self.channels_last: bool = channels_last
        self.trans_keys: Tuple[str, ...] = trans_keys
        self.semantic_key = semantic_key

    def transform_observation_space(self, observation_space: spaces.Dict):
        observation_space = copy.deepcopy(observation_space)
        for key in observation_space.spaces:
            if key in self.trans_keys:
                # In the observation space dict, the channels are always last
                h, w = get_image_height_width(
                    observation_space.spaces[key], channels_last=True
                )
                if self._size == (h, w):
                    continue
                logger.info(
                    "Resizing observation of %s: from %s to %s"
                    % (key, (h, w), self._size)
                )
                observation_space.spaces[key] = overwrite_gym_box_shape(
                    observation_space.spaces[key], self._size
                )
        return observation_space

    def _transform_obs(
        self, obs: torch.Tensor, interpolation_mode: str
    ) -> torch.Tensor:
        return image_resize(
            obs,
            self._size,
            channels_last=self.channels_last,
            interpolation_mode=interpolation_mode,
        )

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for sensor in self.trans_keys:
            if sensor in observations:
                interpolation_mode = "area"
                if self.semantic_key in sensor:
                    interpolation_mode = "nearest"
                observations[sensor] = self._transform_obs(
                    observations[sensor], interpolation_mode
                )
        return observations

    @classmethod
    def from_config(cls, config: "DictConfig"):
        return cls(
            tuple(config.size),
            config.channels_last,
            config.trans_keys,
            config.semantic_key,
        )


def image_resize(
    img: Tensor,
    size: Tuple[int, int],
    channels_last: bool = False,
    interpolation_mode="area",
) -> torch.Tensor:
    """Resizes an img.

    Args:
        img: the array object that needs to be resized (HWC) or (NHWC)
        size: the size that you want
        channels: a boolean that channel is the last dimension
    Returns:
        The resized array as a torch tensor.
    """
    img = torch.as_tensor(img)
    no_batch_dim = len(img.shape) == 3
    if len(img.shape) < 3 or len(img.shape) > 5:
        raise NotImplementedError()
    if no_batch_dim:
        img = img.unsqueeze(0)  # Adds a batch dimension
    if channels_last:
        if len(img.shape) == 4:
            # NHWC -> NCHW
            img = img.permute(0, 3, 1, 2)
        else:
            # NDHWC -> NDCHW
            img = img.permute(0, 1, 4, 2, 3)

    img = torch.nn.functional.interpolate(
        img.float(), size=size, mode=interpolation_mode
    ).to(dtype=img.dtype)
    if channels_last:
        if len(img.shape) == 4:
            # NCHW -> NHWC
            img = img.permute(0, 2, 3, 1)
        else:
            # NDCHW -> NDHWC
            img = img.permute(0, 1, 3, 4, 2)
    if no_batch_dim:
        img = img.squeeze(dim=0)  # Removes the batch dimension
    return img


@dataclass
class ResizeConfig(ObsTransformConfig):
    type: str = Resize.__name__
    size: Tuple[int, int] = (224, 224)
    channels_last: bool = True
    trans_keys: Tuple[str, ...] = (
        "rgb",
        "depth",
        "semantic",
    )
    semantic_key: str = "semantic"


cs = ConfigStore.instance()

cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.resize",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="resize",
    node=ResizeConfig,
)
