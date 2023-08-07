from typing import Any, Dict, Tuple

import cv2
import numpy as np

from ..utils.geometry_utils import wrap_heading
from ..vlm.midas import MidasEstimator
from .pointnav_env import PointNavEnv
from .robots.camera_ids import SpotCamIds

LEFT_CROP = 124
RIGHT_CROP = 60
NOMINAL_ARM_POSE = np.deg2rad([0, -170, 120, 0, 75, 0])


class ObjectNavEnv(PointNavEnv):
    """
    Gym environment for doing the ObjectNav task on the Spot robot in the real world.
    """

    tf_episodic_to_world: np.ndarray = None  # must be set in reset()
    episodic_start_yaw: float = None  # must be set in reset()
    target_object: str = None  # must be set in reset()
    time_step: float = 0.75
    gripper_max_depth: float = 7.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = "/home/naoki/repos/MiDaS/weights/dpt_swin2_large_384.pt"
        model_type = "dpt_swin2_large_384"
        self.depth_model = MidasEstimator(model_path, model_type)

    @property
    def done(self) -> bool:
        if self.goal is None:
            return False
        rho, theta = self._get_rho_theta()
        print("rho, theta", rho, np.rad2deg(theta))
        return rho < self.success_radius

    def reset(self, goal: Any, *args, **kwargs) -> Dict[str, np.ndarray]:
        assert isinstance(goal, str)
        self.target_object = goal
        # Transformation matrix from the frame where the robot started to the world
        # frame
        self.tf_episodic_to_world: np.ndarray = self.robot.get_transform()
        self.tf_episodic_to_world[2, 3] = 0.0  # Make z of the tf 0.0
        self.episodic_start_yaw = self.robot.xy_yaw[1]
        return self._get_obs()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, Dict]:
        # Parent class only moves the base; if we want to move the gripper camera,
        # we need to do it here
        if "gripper_camera_pan" in action:
            self._pan_gripper_camera(action["gripper_camera_pan"])
        return super().step(action)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        tf_camera_to_world = self.robot.get_transform("hand")
        tf_camera_to_episodic = (
            np.linalg.inv(self.tf_episodic_to_world) @ tf_camera_to_world
        )
        gripper_rgb, gripper_depth = self._get_gripper_images()
        return {
            "depth": self._get_depth(),
            "rgb": gripper_rgb,
            "objectgoal": self.target_object,
            "gps": self._get_gps(),
            "compass": np.array([self._get_compass()]),
            "tf_camera_to_episodic": tf_camera_to_episodic,
            "camera_depth": gripper_depth,
        }

    def _get_gripper_images(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the RGB and depth image from the robot's gripper. Crop so that they are
        the same size."""
        srcs = [SpotCamIds.HAND_COLOR, SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME]
        images = self.robot.get_camera_images(srcs)
        # Depth camera is mounted 90 degrees from the color camera, so it has black
        # bars on the left and right that need to be lopped off. The color camera
        # doesn't have this problem, but we cut it off anyway to make the images the
        # same size for easier processing of object locations.
        gripper_rgb, gripper_depth = [
            images[src][:, LEFT_CROP:-RIGHT_CROP] for src in srcs
        ]
        # gripper_depth = self._process_depth(gripper_depth, self.gripper_max_depth)

        gripper_depth = self.depth_model.process(gripper_rgb)
        cv2.resize(
            gripper_depth, gripper_rgb.shape[:2][::-1], interpolation=cv2.INTER_CUBIC
        )
        # gripper_depth *= 7.35
        gripper_depth *= 5.0
        # gripper_depth *= 4.0
        gripper_depth = np.clip(gripper_depth, 0, self.gripper_max_depth)
        gripper_depth /= self.gripper_max_depth

        return gripper_rgb, gripper_depth

    def _get_gps(self) -> np.ndarray:
        """
        Get the (x, y) position of the robot's base in the episode frame. x is forward,
        y is left.
        """
        world_xy = self.robot.xy_yaw[0]

        start_xy = self.tf_episodic_to_world[:2, 3]
        offset = world_xy - start_xy
        rotation_matrix = np.array(
            [
                [np.cos(-self.episodic_start_yaw), -np.sin(-self.episodic_start_yaw)],
                [np.sin(-self.episodic_start_yaw), np.cos(-self.episodic_start_yaw)],
            ]
        )
        episodic_xy = rotation_matrix @ offset
        # Need to negate the y value due to Habitat conventions
        episodic_xy[1] *= -1
        return episodic_xy

    def _get_compass(self) -> float:
        """
        Get the yaw of the robot's base in the episode frame. Yaw is measured in radians
        counterclockwise from the z-axis.
        """
        world_yaw = self.robot.xy_yaw[1]
        episodic_yaw = wrap_heading(world_yaw - self.episodic_start_yaw)
        return episodic_yaw

    def _pan_gripper_camera(self, yaw):
        new_pose = np.array(NOMINAL_ARM_POSE)
        new_pose[0] = yaw[0]
        self.robot.set_arm_joints(new_pose, travel_time=self.time_step * 0.9)
