# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from depth_camera_filtering import filter_depth
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import BaseObjectNavPolicy
from vlfm.policy.itm_policy import ITMPolicy, ITMPolicyV2, ITMPolicyV3
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix
from vlfm.vlm.grounding_dino import ObjectDetections


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class SemExpMixin:
    """This Python mixin only contains code relevant for running a BaseObjectNavPolicy
    explicitly within Habitat (vs. the real world, etc.) and will endow any parent class
    (that is a subclass of BaseObjectNavPolicy) with the necessary methods to run in
    Habitat.
    """

    _stop_action: Tensor = TorchActionIDs.STOP
    _start_yaw: Union[float, None] = None  # must be set by _reset() method
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}

    def __init__(
        self: Union["SemExpMixin", BaseObjectNavPolicy],
        camera_height: float,
        min_depth: float,
        max_depth: float,
        camera_fov: float,
        image_width: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore
        assert self._compute_frontiers, "Must set self._compute_frontiers = True"
        self._camera_height = camera_height
        self._min_depth = min_depth
        self._max_depth = max_depth
        camera_fov_rad = np.deg2rad(camera_fov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = image_width / (2 * np.tan(camera_fov_rad / 2))

        self._compute_frontiers: bool = super()._compute_frontiers  # type: ignore

    def act(
        self: Union["SemExpMixin", BaseObjectNavPolicy],
        observations: Dict[str, Union[Tensor, str]],
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = True,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Converts object ID to string name, returns action as PolicyActionData"""
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        try:
            action, rnn_hidden_states = parent_cls.act(
                observations, None, None, masks, deterministic
            )
        except StopIteration:
            action = self._stop_action
        return action, self._policy_info

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
        self: Union["SemExpMixin", BaseObjectNavPolicy], observations: Dict[str, Any]
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


class SemExpITMPolicy(SemExpMixin, ITMPolicy):
    pass


class SemExpITMPolicyV2(SemExpMixin, ITMPolicyV2):
    pass


class SemExpITMPolicyV3(SemExpMixin, ITMPolicyV3):
    pass
