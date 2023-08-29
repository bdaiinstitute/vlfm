import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from zsos.reality.robots.base_robot import BaseRobot
from zsos.reality.robots.camera_ids import SpotCamIds
from zsos.utils.geometry_utils import convert_to_global_frame, rho_theta


class PointNavEnv:
    """
    Gym environment for doing the PointNav task on the Spot robot in the real world.
    """

    goal: Any = (None,)  # dummy value; must be set by reset()
    info: Dict = {}

    def __init__(
        self,
        robot: BaseRobot,
        # robot: BDSWRobot,
        max_body_cam_depth: float = 3.5,  # max depth (in meters) for body cameras
        max_lin_dist: float = 0.25,
        max_ang_dist: float = np.deg2rad(30),
        time_step: float = 0.5,
        *args,
        **kwargs,
    ):
        self.robot = robot

        self._max_body_cam_depth = max_body_cam_depth
        self._max_lin_dist = max_lin_dist
        self._max_ang_dist = max_ang_dist
        self._time_step = time_step

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
        # self.robot.command_base_velocity(ang_vel, lin_vel)
        time.sleep(self._time_step)
        self.robot.command_base_velocity(0.0, 0.0)
        done = ang_vel == 0.0 and lin_vel == 0.0
        return self._get_obs(), 0.0, done, {}  # not using info dict yet

    def _compute_velocities(self, action: Dict[str, np.ndarray]) -> Tuple[float, float]:
        velocities = []
        for action_key, max_dist in (
            ["angular", self._max_ang_dist],
            ["linear", self._max_lin_dist],
        ):
            act_val = action.get(action_key, 0.0)  # default to 0.0 if key not present
            dist = np.clip(act_val, -1.0, 1.0)  # clip to [-1, 1]
            dist *= max_dist  # scale to max distance
            velocities.append(dist / self._time_step)  # convert to velocity
        ang_vel, lin_vel = velocities
        return ang_vel, lin_vel

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "depth": self._get_nav_depth(),
            "pointgoal_with_gps_compass": self._get_rho_theta(),
        }

    def _get_nav_depth(self) -> np.ndarray:
        images = self.robot.get_camera_images(
            [SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH]
        )
        img = np.hstack(
            [images[SpotCamIds.FRONTRIGHT_DEPTH], images[SpotCamIds.FRONTLEFT_DEPTH]]
        )
        img = self._norm_depth(img)
        return img

    def _norm_depth(
        self, depth: np.ndarray, max_depth: Optional[float] = None, scale: bool = True
    ) -> np.ndarray:
        if max_depth is None:
            max_depth = self._max_body_cam_depth

        norm_depth = depth.astype(np.float32)

        if scale:
            # Convert to meters from mm (uint16)
            norm_depth = norm_depth / 1000.0

        # Normalize to [0, 1]
        norm_depth = np.clip(norm_depth, 0.0, max_depth) / max_depth

        return norm_depth

    def _get_rho_theta(self) -> np.ndarray:
        curr_pos, yaw = self.robot.xy_yaw
        r_t = rho_theta(curr_pos, yaw, self.goal)
        return np.array(r_t)
