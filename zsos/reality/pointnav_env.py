import time
from typing import Dict, Tuple, Union

import cv2
import numpy as np
import torch
from depth_camera_filtering import filter_depth

from zsos.mapping.object_map import convert_to_global_frame
from zsos.policy.utils.pointnav_policy import rho_theta
from zsos.reality.robots.base_robot import BaseRobot
from zsos.reality.robots.camera_ids import SpotCamIds


class PointNavEnv:
    """Gym environment for doing the PointNav task."""

    max_depth: float = 3.5
    success_radius: float = 0.425
    goal: np.ndarray = np.array([0.0, 0.0])
    max_lin_dist: float = 0.25
    max_ang_dist: float = np.deg2rad(30)
    time_step: float = 0.5
    depth_shape: Tuple[int, int] = (212, 240)  # height, width
    info: Dict = {}

    def __init__(self, robot: BaseRobot):
        self.robot = robot

    def reset(self, goal: np.ndarray, relative=True) -> Dict[str, np.ndarray]:
        if relative:
            # Transform (x,y) goal from robot frame to global frame
            pos, yaw = self.robot.xy_yaw
            pos_w_z = np.array([pos[0], pos[1], 0.0])  # inject dummy z value
            goal_w_z = np.array([goal[0], goal[1], 0.0])  # inject dummy z value
            goal = convert_to_global_frame(pos_w_z, yaw, goal_w_z)[:2]  # drop z
        self.goal = goal
        return self._get_obs()

    def step(
        self, action: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[Dict, float, bool, Dict]:
        self.info = {}
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        ang_vel, lin_vel = self._compute_velocities(action)
        self.robot.command_base_velocity(ang_vel, lin_vel)
        time.sleep(self.time_step)
        self.robot.command_base_velocity(0.0, 0.0)
        r_t = self._get_rho_theta()
        print("rho: ", r_t[0], "theta: ", np.rad2deg(r_t[1]))
        return self._get_obs(), 0.0, self.done, self.info

    @property
    def done(self) -> bool:
        rho = self._get_rho_theta()[0]
        return rho < self.success_radius

    def _compute_velocities(self, action: np.ndarray) -> Tuple[float, float]:
        ang_dist, lin_dist = np.clip(
            action[0],
            -1.0,
            1.0,
        )
        ang_dist *= self.max_ang_dist
        lin_dist *= self.max_lin_dist
        ang_vel = ang_dist / self.time_step
        lin_vel = lin_dist / self.time_step
        print("action: ", action[0])
        print("ang_vel: ", np.rad2deg(ang_vel), "lin_vel: ", lin_vel)
        print("ang_dist: ", np.rad2deg(ang_dist), "lin_dist: ", lin_dist)
        return ang_vel, lin_vel

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "depth": self._get_depth(),
            "pointgoal_with_gps_compass": self._get_rho_theta(),
        }

    def _get_depth(self) -> np.ndarray:
        images = self.robot.get_camera_images(
            [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
        )
        # Spot is cross-eyed, so right eye is on the left, and vice versa
        img = np.hstack(
            [images[SpotCamIds.FRONTRIGHT_DEPTH], images[SpotCamIds.FRONTLEFT_DEPTH]]
        )
        img = img.astype(np.float32) / 1000.0  # Convert to meters from mm (uint16)
        # Filter the image and re-scale based on max depth limit (self.max_depth)
        img = filter_depth(
            img, clip_far_thresh=self.max_depth, set_black_value=self.max_depth
        )
        img = img / self.max_depth  # Normalize to [0, 1]
        # Down-sample to policy input shape
        img = cv2.resize(
            img,
            (self.depth_shape[1], self.depth_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        # Add a channel dimension
        img = img.reshape(img.shape + (1,))

        return img

    def _get_rho_theta(self) -> np.ndarray:
        curr_pos, yaw = self.robot.xy_yaw
        r_t = rho_theta(curr_pos, yaw, self.goal)
        return np.array(r_t)
