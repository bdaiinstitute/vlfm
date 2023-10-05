from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy, ZSOSConfig
from vlfm.policy.itm_policy import ITMPolicyV2

INITIAL_ARM_YAWS = np.deg2rad([-90, -60, -30, 0, 30, 60, 90, 0]).tolist()


class RealityMixin:
    """
    This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly in the real world (vs. Habitat), and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run on the
    Spot robot in the real world.
    """

    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = False
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    _initial_yaws: List = INITIAL_ARM_YAWS.copy()

    def __init__(self: BaseObjectNavPolicy, *args: Any, **kwargs: Any) -> None:
        super().__init__(sync_explored_areas=True, *args, **kwargs)
        self._depth_model = torch.hub.load(
            "isl-org/ZoeDepth", "ZoeD_NK", config_mode="eval", pretrained=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self._object_map.use_dbscan = False

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused, **kwargs_unused):
        policy_config: ZSOSConfig = config.policy
        kwargs = {k: policy_config[k] for k in ZSOSConfig.kwaarg_names}  # type: ignore

        return cls(**kwargs)

    def act(
        self: Union["RealityMixin", BaseObjectNavPolicy],
        observations: Dict[str, Any],
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic=False,
    ) -> Dict[str, Any]:
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = (
                observations["objectgoal"] + " . " + self._non_coco_caption
            )
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        action: Tensor = parent_cls.act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )[0]

        # The output of the policy is a (1, 2) tensor of floats, where the first element
        # is the linear velocity and the second element is the angular velocity. We
        # convert this numpy array to a dictionary with keys "angular" and "linear" so
        # that it can be passed to the Spot robot.
        if self._done_initializing:
            action_dict = {
                "angular": action[0][0].item(),
                "linear": action[0][1].item(),
                "arm_yaw": -1,
                "info": self._policy_info,
            }
        else:
            action_dict = {
                "angular": 0,
                "linear": 0,
                "arm_yaw": action[0][0].item(),
                "info": self._policy_info,
            }

        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        self._done_initializing = len(self._initial_yaws) == 0

        return action_dict

    def get_action(
        self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True
    ) -> Dict[str, Any]:
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: BaseObjectNavPolicy) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._initial_yaws = INITIAL_ARM_YAWS.copy()
        self._done_initializing = False

    def _initialize(self) -> Tensor:
        yaw = self._initial_yaws.pop(0)
        return torch.tensor([[yaw]], dtype=torch.float32)

    def _cache_observations(
        self: Union["RealityMixin", BaseObjectNavPolicy], observations: Dict[str, Any]
    ):
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (Dict[str, Any]): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return

        self._obstacle_map: ObstacleMap
        for obs_map_data in observations["obstacle_map_depths"][:-1]:
            depth, tf, min_depth, max_depth, fx, fy, topdown_fov = obs_map_data
            self._obstacle_map.update_map(
                depth,
                tf,
                min_depth,
                max_depth,
                fx,
                fy,
                topdown_fov,
                explore=False,
            )

        _, tf, min_depth, max_depth, fx, fy, topdown_fov = observations[
            "obstacle_map_depths"
        ][-1]
        self._obstacle_map.update_map(
            None,
            tf,
            min_depth,
            max_depth,
            fx,
            fy,
            topdown_fov,
            explore=True,
            update_obstacles=False,
        )

        self._obstacle_map.update_agent_traj(
            observations["robot_xy"], observations["robot_heading"]
        )
        frontiers = self._obstacle_map.frontiers

        height, width = observations["nav_depth"].shape
        nav_depth = torch.from_numpy(observations["nav_depth"])
        nav_depth = nav_depth.reshape(1, height, width, 1).to("cuda")

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": nav_depth,  # for pointnav
            "robot_xy": observations["robot_xy"],  # (2,) np.ndarray
            "robot_heading": observations["robot_heading"],  # float in radians
            "object_map_rgbd": observations["object_map_rgbd"],
            "value_map_rgbd": observations["value_map_rgbd"],
        }

    def _infer_depth(
        self, rgb: np.ndarray, min_depth: float, max_depth: float
    ) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        img_pil = Image.fromarray(rgb)
        with torch.inference_mode():
            depth = self._depth_model.infer_pil(img_pil)
        depth = (np.clip(depth, min_depth, max_depth)) / (max_depth - min_depth)
        return depth


@dataclass
class RealityConfig(DictConfig):
    policy: ZSOSConfig = ZSOSConfig()


class RealityITMPolicyV2(RealityMixin, ITMPolicyV2):
    pass
