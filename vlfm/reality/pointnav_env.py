# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from vlfm.reality.robots.bdsw_robot import BDSWRobot
from vlfm.reality.robots.camera_ids import SpotCamIds
from vlfm.utils.geometry_utils import (
    convert_to_global_frame,
    pt_from_rho_theta,
    rho_theta,
)


class PointNavEnv:
    """
    Gym environment for doing the PointNav task on the Spot robot in the real world.
    """

    goal: Any = (None,)  # dummy value; must be set by reset()
    info: Dict = {}

    def __init__(
        self,
        robot: BDSWRobot,
        max_body_cam_depth: float = 3.5,  # max depth (in meters) for body cameras
        max_lin_dist: float = 0.25,
        max_ang_dist: float = np.deg2rad(30),
        time_step: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.robot = robot

        self._max_body_cam_depth = max_body_cam_depth
        self._max_lin_dist = max_lin_dist
        self._max_ang_dist = max_ang_dist
        self._time_step = time_step
        self._cmd_id: Union[None, Any] = None
        self._num_steps = 0

    def reset(self, goal: Any, relative: bool = True, *args: Any, **kwargs: Any) -> Dict[str, np.ndarray]:
        assert isinstance(goal, np.ndarray)
        if relative:
            # Transform (x,y) goal from robot frame to global frame
            pos, yaw = self.robot.xy_yaw
            pos_w_z = np.array([pos[0], pos[1], 0.0])  # inject dummy z value
            goal_w_z = np.array([goal[0], goal[1], 0.0])  # inject dummy z value
            goal = convert_to_global_frame(pos_w_z, yaw, goal_w_z)[:2]  # drop z
        self.goal = goal
        return self._get_obs()

    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, Dict]:
        if self._cmd_id is not None:
            cmd_status = 0
            while cmd_status != 1:
                feedback_resp = self.robot.spot.get_cmd_feedback(self._cmd_id)
                cmd_status = (
                    feedback_resp.feedback.synchronized_feedback
                ).mobility_command_feedback.se2_trajectory_feedback.status
                if cmd_status != 1:
                    time.sleep(0.1)

        ang_dist, lin_dist = self._compute_displacements(action)
        done = action["linear"] == 0.0 and action["angular"] == 0.0
        print("ang/lin:", ang_dist, lin_dist)

        if "rho_theta" in action:
            rho, theta = action["rho_theta"]
            x_pos, y_pos = pt_from_rho_theta(rho, theta)
            yaw = theta
            print("RHO", rho)
        else:
            x_pos = lin_dist
            y_pos = 0
            yaw = ang_dist

        if done:
            self.robot.command_base_velocity(0.0, 0.0)

        self._cmd_id = self.robot.spot.set_base_position(
            x_pos=x_pos,
            y_pos=y_pos,
            yaw=yaw,
            end_time=100,
            relative=True,
            max_fwd_vel=0.3,
            max_hor_vel=0.2,
            max_ang_vel=np.deg2rad(60),
            disable_obstacle_avoidance=False,
            blocking=False,
        )
        if "rho_theta" in action:
            self._cmd_id = None

        self._num_steps += 1
        return self._get_obs(), 0.0, done, {}  # not using info dict yet

    def _compute_velocities(self, action: Dict[str, Any]) -> Tuple[float, float]:
        ang_dist, lin_dist = self._compute_displacements(action)
        ang_vel = ang_dist / self._time_step
        lin_vel = lin_dist / self._time_step
        return ang_vel, lin_vel

    def _compute_displacements(self, action: Dict[str, Any]) -> Tuple[float, float]:
        displacements = []
        for action_key, max_dist in (
            ("angular", self._max_ang_dist),
            ("linear", self._max_lin_dist),
        ):
            if action_key not in action:
                displacements.append(0.0)
                continue
            act_val = action[action_key]
            dist = np.clip(act_val, -1.0, 1.0)  # clip to [-1, 1]
            dist *= max_dist  # scale to max distance
            displacements.append(dist)  # convert to velocities
        ang_dist, lin_dist = displacements
        return ang_dist, lin_dist

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "depth": self._get_nav_depth(),
            "pointgoal_with_gps_compass": self._get_rho_theta(),
        }

    def _get_nav_depth(self) -> np.ndarray:
        images = self.robot.get_camera_images([SpotCamIds.FRONTRIGHT_DEPTH, SpotCamIds.FRONTLEFT_DEPTH])
        img = np.hstack([images[SpotCamIds.FRONTRIGHT_DEPTH], images[SpotCamIds.FRONTLEFT_DEPTH]])
        img = self._norm_depth(img)
        return img

    def _norm_depth(self, depth: np.ndarray, max_depth: Optional[float] = None, scale: bool = True) -> np.ndarray:
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
