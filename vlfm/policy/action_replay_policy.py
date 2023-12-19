# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, List

import cv2
import numpy as np
import torch
from depth_camera_filtering import filter_depth
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from omegaconf import DictConfig

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_policy import BasePolicy
from vlfm.utils.geometry_utils import xyz_yaw_to_tf_matrix


@baseline_registry.register_policy
class ActionReplayPolicy(BasePolicy):
    def __init__(
        self,
        forward_step_size: float,
        turn_angle: float,
        min_obstacle_height: float,
        max_obstacle_height: float,
        obstacle_map_area_threshold: float,
        agent_radius: float,
        hole_area_thresh: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        assert "VLFM_RECORD_ACTIONS_DIR" in os.environ, "Must set VLFM_RECORD_ACTIONS_DIR"
        self._dir = os.environ["VLFM_RECORD_ACTIONS_DIR"]
        filepath = os.path.join(self._dir, "actions.txt")
        with open(filepath, "r") as f:
            self._actions = [int(i) for i in f.readlines()]
        turn_repeat = int(30 / turn_angle)
        step_repeat = int(0.25 / forward_step_size)
        for turn_type in [2, 3]:
            self._actions = repeat_elements(self._actions, turn_type, turn_repeat)
        self._actions = repeat_elements(self._actions, 1, step_repeat)
        img_dir = os.path.join(self._dir, "imgs")
        # Create the directory if it doesn't exist
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        self.curr_idx = 0

        self._camera_height = 0.88
        self._obstacle_map = ObstacleMap(
            min_height=min_obstacle_height,
            max_height=max_obstacle_height,
            area_thresh=obstacle_map_area_threshold,
            agent_radius=agent_radius,
            hole_area_thresh=hole_area_thresh,
            pixels_per_meter=50,
            size=2500,
        )
        self._obstacle_map.radius_padding_color = (0, 0, 0)
        self._camera_fov_rad = np.deg2rad(79)

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        **kwargs: Any,
    ) -> "ActionReplayPolicy":
        policy_cfg = config.habitat_baselines.rl.policy

        return cls(
            forward_step_size=config.habitat.simulator.forward_step_size,
            turn_angle=config.habitat.simulator.turn_angle,
            min_obstacle_height=policy_cfg.min_obstacle_height,
            max_obstacle_height=policy_cfg.max_obstacle_height,
            obstacle_map_area_threshold=policy_cfg.obstacle_map_area_threshold,
            agent_radius=policy_cfg.agent_radius,
            hole_area_thresh=policy_cfg.hole_area_thresh,
        )

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        # Save the rgb and depth images
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        rgb_path = os.path.join(
            self._dir,
            "imgs",
            f"{self.curr_idx:05d}_rgb.png",
        )
        depth_path = os.path.join(
            self._dir,
            "imgs",
            f"{self.curr_idx:05d}_depth.png",
        )

        # Save the images
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(rgb_path, bgr)
        depth_int = (depth * 255).astype("uint8")
        cv2.imwrite(depth_path, depth_int)

        # Log the position and yaw
        x, y = observations["gps"][0].cpu().numpy()
        csv_data = [
            str(x),
            str(-y),
            str(observations["compass"][0].cpu().item()),
            str(observations["heading"][0].item()),
        ]
        csv_line = ",".join(csv_data)
        filepath = os.path.join(
            self._dir,
            "position_yaws.csv",
        )

        # If the file doesn't exist, create it and write the header
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write("x,y,compass,heading\n")
        with open(filepath, "a") as f:
            f.write(f"{csv_line}\n")

        # obstacle map updating
        image_width = depth.shape[1]
        fx = fy = image_width / (2 * np.tan(self._camera_fov_rad / 2))

        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self._camera_height])
        robot_xy = camera_position[:2]
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        self._obstacle_map.update_map(
            depth,
            tf_camera_to_episodic,
            0.5,  # self._min_depth,
            5.0,  # self._max_depth,
            fx,
            fy,
            self._camera_fov_rad,
        )
        self._obstacle_map.update_agent_traj(robot_xy, camera_yaw)
        frontier_map_path = os.path.join(
            self._dir,
            "imgs",
            f"{self.curr_idx:05d}_frontier_map.png",
        )
        cv2.imwrite(frontier_map_path, self._obstacle_map.visualize())

        action = torch.tensor([self._actions[self.curr_idx]], dtype=torch.long)
        action_data = PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[{}],
        )
        self.curr_idx += 1

        return action_data


def repeat_elements(lst: List[int], element: int, repeat_count: int) -> List[int]:
    new_list = []
    for i in lst:
        if i == element:
            new_list.extend([i] * repeat_count)
        else:
            new_list.append(i)
    return new_list
