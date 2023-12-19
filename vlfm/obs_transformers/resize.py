# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

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

from vlfm.obs_transformers.utils import image_resize


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

    def transform_observation_space(self, observation_space: spaces.Dict) -> spaces.Dict:
        observation_space = copy.deepcopy(observation_space)
        for key in observation_space.spaces:
            if key in self.trans_keys:
                # In the observation space dict, the channels are always last
                h, w = get_image_height_width(observation_space.spaces[key], channels_last=True)
                if self._size == (h, w):
                    continue
                logger.info("Resizing observation of %s: from %s to %s" % (key, (h, w), self._size))
                observation_space.spaces[key] = overwrite_gym_box_shape(observation_space.spaces[key], self._size)
        return observation_space

    def _transform_obs(self, obs: torch.Tensor, interpolation_mode: str) -> torch.Tensor:
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
                observations[sensor] = self._transform_obs(observations[sensor], interpolation_mode)
        return observations

    @classmethod
    def from_config(cls, config: "DictConfig") -> "Resize":
        return cls(
            (int(config.size[0]), int(config.size[1])),
            config.channels_last,
            config.trans_keys,
            config.semantic_key,
        )


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
