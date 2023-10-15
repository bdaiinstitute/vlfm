from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from frontier_exploration.base_explorer import BaseExplorer
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.config.default_structured_configs import (
    PolicyConfig,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch import Tensor

from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections

from ..mapping.obstacle_map import ObstacleMap
from .base_objectnav_policy import BaseObjectNavPolicy, ZSOSConfig
from .itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3

HM3D_ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
MP3D_ID_TO_NAME = [
    "chair",
    "table|dining table|coffee table|side table|desk",  # "table",
    "framed photograph",  # "picture",
    "cabinet",
    "pillow",  # "cushion",
    "couch",  # "sofa",
    "bed",
    "nightstand",  # "chest of drawers",
    "potted plant",  # "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv",  # "tv monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym equipment",
    "seating",
    "clothes",
]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    """This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _compute_frontiers: bool = False

    def __init__(
        self,
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        dataset_type: str = "hm3d",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))
        self._dataset_type = dataset_type

    @classmethod
    def from_config(
        cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any
    ) -> "HabitatMixin":
        policy_config: ZSOSPolicyConfig = config.habitat_baselines.rl.policy
        kwargs = {
            k: policy_config[k] for k in ZSOSPolicyConfig.kwaarg_names  # type: ignore
        }

        # In habitat, we need the height of the camera to generate the camera transform
        sim_sensors_cfg = config.habitat.simulator.agents.main_agent.sim_sensors
        kwargs["camera_height"] = sim_sensors_cfg.rgb_sensor.position[1]

        # Synchronize the mapping min/max depth values with the habitat config
        kwargs["min_depth"] = sim_sensors_cfg.depth_sensor.min_depth
        kwargs["max_depth"] = sim_sensors_cfg.depth_sensor.max_depth
        kwargs["camera_fov"] = sim_sensors_cfg.depth_sensor.hfov
        kwargs["image_width"] = sim_sensors_cfg.depth_sensor.width

        # Only bother visualizing if we're actually going to save the video
        kwargs["visualize"] = len(config.habitat_baselines.eval.video_option) > 0

        if "hm3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "hm3d"
        elif "mp3d" in config.habitat.dataset.data_path:
            kwargs["dataset_type"] = "mp3d"
        else:
            raise ValueError("Dataset type could not be inferred from habitat config")

        return cls(**kwargs)

    def act(
        self: Union["HabitatMixin", BaseObjectNavPolicy],
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        if self._dataset_type == "hm3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = HM3D_ID_TO_NAME[object_id]
        elif self._dataset_type == "mp3d":
            obs_dict[ObjectGoalSensor.cls_uuid] = MP3D_ID_TO_NAME[object_id]
            self._non_coco_caption = (
                " . ".join(MP3D_ID_TO_NAME).replace("|", " . ") + " ."
            )
        else:
            raise ValueError(f"Dataset type {self._dataset_type} not recognized")
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        try:
            action, rnn_hidden_states = parent_cls.act(
                obs_dict, rnn_hidden_states, prev_actions, masks, deterministic
            )
        except StopIteration:
            action = self._stop_action
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self._done_initializing = not self._num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT

    def _reset(self) -> None:
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        parent_cls._reset()
        self._start_yaw = None

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(detections)

        if not self._visualize:  # type: ignore
            return info

        if self._start_yaw is None:
            self._start_yaw = self._observations_cache["habitat_start_yaw"]
        info["start_yaw"] = self._start_yaw
        return info

    def _cache_observations(
        self: Union["HabitatMixin", BaseObjectNavPolicy], observations: TensorDict
    ) -> None:
        """Caches the rgb, depth, and camera transform from the observations.

        Args:
           observations (TensorDict): The observations from the current timestep.
        """
        if len(self._observations_cache) > 0:
            return
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        self._obstacle_map: ObstacleMap
        if self._compute_frontiers:
            self._obstacle_map.update_map(
                depth,
                tf_camera_to_episodic,
                self._min_depth,
                self._max_depth,
                self._fx,
                self._fy,
                self._camera_fov,
            )
            frontiers = self._obstacle_map.frontiers
            self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        else:
            if "frontier_sensor" in observations:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
            else:
                frontiers = np.array([])

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": observations["depth"],  # for pointnav
            "robot_xy": robot_xy,
            "robot_heading": camera_yaw,
            "object_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._fx,
                    self._fy,
                )
            ],
            "value_map_rgbd": [
                (
                    rgb,
                    depth,
                    tf_camera_to_episodic,
                    self._min_depth,
                    self._max_depth,
                    self._camera_fov,
                )
            ],
            "habitat_start_yaw": observations["heading"][0].item(),
        }


@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,  # can be anything because it is not used
        *args: Any,
        **kwargs: Any,
    ) -> PolicyActionData:
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV2(HabitatMixin, ITMPolicyV2):
    pass


@baseline_registry.register_policy
class HabitatITMPolicyV3(HabitatMixin, ITMPolicyV3):
    pass


@dataclass
class ZSOSPolicyConfig(ZSOSConfig, PolicyConfig):
    pass


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="zsos_policy", node=ZSOSPolicyConfig)
