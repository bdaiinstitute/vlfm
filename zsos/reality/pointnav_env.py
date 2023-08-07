import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from depth_camera_filtering import filter_depth

from zsos.mapping.object_map import convert_to_global_frame

from ..utils.geometry_utils import rho_theta
from .robots.base_robot import BaseRobot
from .robots.camera_ids import SpotCamIds


class PointNavEnv:
    """
    Gym environment for doing the PointNav task on the Spot robot in the real world.
    """

    max_depth: float = 3.5  # max depth value in meters policy was trained on
    success_radius: float = 0.425
    goal: Any = None  # dummy values; must be set by reset()
    max_lin_dist: float = 0.25
    max_ang_dist: float = np.deg2rad(30)
    time_step: float = 0.5
    depth_image_shape: Tuple[int, int] = (212, 240)  # height, width
    info: Dict = {}

    def __init__(self, robot: BaseRobot):
        self.robot = robot

    @property
    def done(self) -> bool:
        rho, _ = self._get_rho_theta()
        return rho < self.success_radius

    def reset(self, goal: Any, relative=True, *args, **kwargs) -> Dict[str, np.ndarray]:
        assert isinstance(goal, np.ndarray)
        if relative:
            # Transform (x,y) goal from robot frame to global frame
            pos, yaw = self.robot.xy_yaw
            pos_w_z = np.array([pos[0], pos[1], 0.0])  # inject dummy z value
            goal_w_z = np.array([goal[0], goal[1], 0.0])  # inject dummy z value
            goal = convert_to_global_frame(pos_w_z, yaw, goal_w_z)[:2]  # drop z
        self.goal = goal
        return self._get_obs()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, Dict]:
        ang_vel, lin_vel = self._compute_velocities(action)
        self.robot.command_base_velocity(ang_vel, lin_vel)
        time.sleep(self.time_step)
        self.robot.command_base_velocity(0.0, 0.0)
        return self._get_obs(), 0.0, self.done, {}  # not using info dict yet

    def _compute_velocities(self, action: Dict[str, np.ndarray]) -> Tuple[float, float]:
        velocities = []
        for action_key, max_dist in (
            ["angular_action", self.max_ang_dist],
            ["linear_action", self.max_lin_dist],
        ):
            act_val = action.get(action_key, 0.0)  # default to 0.0 if key not present
            dist = np.clip(act_val, -1.0, 1.0)  # clip to [-1, 1]
            dist *= max_dist  # scale to max distance
            velocities.append(dist / self.time_step)  # convert to velocity
        ang_vel, lin_vel = velocities
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
        img = self._process_depth(img, self.max_depth, self.depth_image_shape)

        return img

    @staticmethod
    def _process_depth(
        img: np.ndarray, max_depth, depth_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        img = img.astype(np.float32) / 1000.0  # Convert to meters from mm (uint16)
        # Filter the image and re-scale based on max depth limit (self.max_depth)
        img = filter_depth(img, clip_far_thresh=max_depth, set_black_value=max_depth)
        img = img / max_depth  # Normalize to [0, 1]
        if depth_shape is not None:
            # Down-sample to policy input shape
            img = cv2.resize(
                img,
                (depth_shape[1], depth_shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        # Ensure a channel dimension
        img = img.reshape(img.shape[0], img.shape[1], 1)

        return img

    def _get_rho_theta(self) -> np.ndarray:
        curr_pos, yaw = self.robot.xy_yaw
        r_t = rho_theta(curr_pos, yaw, self.goal)
        return np.array(r_t)
