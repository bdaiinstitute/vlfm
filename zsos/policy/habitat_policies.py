from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from habitat.tasks.nav.nav import HeadingSensor
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

from frontier_exploration.base_explorer import BaseExplorer
from zsos.utils.geometry_utils import xyz_yaw_to_tf_matrix
from zsos.vlm.grounding_dino import ObjectDetections

from .base_objectnav_policy import BaseObjectNavPolicy
from .itm_policy import ITMPolicy

ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]


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

    _id_to_padding: Dict[str, float] = {
        "bed": 0.3,
        "couch": 0.15,
    }
    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method

    def __init__(self, camera_height: float, *args: Any, **kwargs: Any) -> None:
        self._camera_height = camera_height
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config: DictConfig, *args_unused, **kwargs_unused):
        policy_config: ZSOSPolicyConfig = config.habitat_baselines.rl.policy
        kwargs = {
            k: policy_config[k] for k in ZSOSPolicyConfig.arg_names()  # type: ignore
        }

        # In habitat, we need the height of the camera to generate the camera transform
        agent_config = config.habitat.simulator.agents.main_agent
        kwargs["camera_height"] = agent_config.sim_sensors.rgb_sensor.position[1]

        return cls(**kwargs)

    def act(
        self: BaseObjectNavPolicy,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic=False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        obs_dict[ObjectGoalSensor.cls_uuid]: str = ID_TO_NAME[object_id]
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        action, rnn_hidden_states = parent_cls.act(
            obs_dict, rnn_hidden_states, prev_actions, masks, deterministic
        )
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

    def _get_policy_info(
        self, observations: TensorDict, detections: ObjectDetections
    ) -> Dict[str, Any]:
        """Get policy info for logging"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        info = parent_cls._get_policy_info(observations, detections)
        if self._start_yaw is None:
            self._start_yaw = observations[HeadingSensor.cls_uuid][0].item()
        info["start_yaw"] = self._start_yaw
        return info

    def _get_object_camera_info(
        self, observations: TensorDict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations (TensorDict): The observations from the current timestep.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The rgb image, depth image, and
                camera transform. The depth image is normalized to be between 0 and 1.
                The camera transform is the transform from the camera to the episodic
                frame, a 4x4 transformation matrix.
        """
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height])
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
        return rgb, depth, tf_camera_to_episodic


@baseline_registry.register_policy
class OracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        explorer_key = [k for k in observations.keys() if k.endswith("_explorer")][0]
        pointnav_action = observations[explorer_key]
        return pointnav_action


@baseline_registry.register_policy
class SuperOracleFBEPolicy(HabitatMixin, BaseObjectNavPolicy):
    def act(
        self, observations: TensorDict, rnn_hidden_states: Any, *args, **kwargs
    ) -> PolicyActionData:
        return PolicyActionData(
            actions=observations[BaseExplorer.cls_uuid],
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self._policy_info],
        )


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass


@dataclass
class ZSOSPolicyConfig(PolicyConfig):
    name: str = "HabitatITMPolicy"
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (244, 224)
    det_conf_threshold: float = 0.6
    pointnav_stop_radius: float = 0.85
    object_map_min_depth: float = 0.5
    object_map_max_depth: float = 5.0
    object_map_hfov: float = 79.0
    value_map_max_depth: float = 5.0
    value_map_hfov: float = 79.0
    object_map_proximity_threshold: float = 1.5
    visualize: bool = True

    @classmethod
    def arg_names(cls) -> List[str]:
        # All the above except "name". Also excludes all attributes from parent classes.
        return [
            "pointnav_policy_path",
            "depth_image_shape",
            "det_conf_threshold",
            "pointnav_stop_radius",
            "object_map_min_depth",
            "object_map_max_depth",
            "object_map_hfov",
            "object_map_proximity_threshold",
            "value_map_max_depth",
            "value_map_hfov",
            "visualize",
        ]


cs = ConfigStore.instance()
cs.store(group="habitat_baselines/rl/policy", name="zsos_policy", node=ZSOSPolicyConfig)
