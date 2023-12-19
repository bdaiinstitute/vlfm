# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Tuple

import numpy as np
from spot_wrapper.spot import Spot, image_response_to_cv2

from .base_robot import BaseRobot
from .frame_ids import SpotFrameIds

MAX_CMD_DURATION = 5


class BDSWRobot(BaseRobot):
    def __init__(self, spot: Spot):
        self.spot = spot

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Returns [x, y], yaw"""
        x, y, yaw = self.spot.get_xy_yaw(use_boot_origin=True)
        return np.array([x, y]), yaw

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns current angle for each of the 6 arm joints in radians"""
        arm_proprioception = self.spot.get_arm_proprioception()
        current_positions = np.array([v.position.value for v in arm_proprioception.values()])
        return current_positions

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """Returns a dict of images mapping camera ids to images

        Args:
            camera_source (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        image_responses = self.spot.get_image_responses(camera_source)
        imgs = {
            source: image_response_to_cv2(image_response, reorient=True)
            for source, image_response in zip(camera_source, image_responses)
        }
        return imgs

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        """Commands the base to execute given angular/linear velocities, non-blocking

        Args:
            ang_vel (float): Angular velocity in radians per second
            lin_vel (float): Linear velocity in meters per second
        """
        # Just make the robot stop moving if both velocities are very low
        if np.abs(ang_vel) < 0.01 and np.abs(lin_vel) < 0.01:
            self.spot.stand()
        else:
            self.spot.set_base_velocity(
                lin_vel,
                0.0,  # no horizontal velocity
                ang_vel,
                MAX_CMD_DURATION,
            )

    def get_transform(self, frame: str = SpotFrameIds.BODY) -> np.ndarray:
        """Returns the transformation matrix of the robot's base (body) or a link

        Args:
            frame (str, optional): Frame to get the transform of. Defaults to
                SpotFrameIds.BODY.

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        return self.spot.get_transform(from_frame=frame)

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        """Moves each of the 6 arm joints to the specified angle

        Args:
            joints (np.ndarray): Array of 6 angles in radians
            travel_time (float): Time in seconds to reach the specified angles
        """
        self.spot.set_arm_joint_positions(positions=joints, travel_time=travel_time)

    def open_gripper(self) -> None:
        """Opens the gripper"""
        self.spot.open_gripper()

    def get_camera_data(self, srcs: List[str]) -> Dict[str, Dict[str, Any]]:
        """Returns a dict that maps each camera id to its image, focal lengths, and
        transform matrix (from camera to global frame).

        Args:
            srcs (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        image_responses = self.spot.get_image_responses(srcs)
        imgs = {
            src: self._camera_response_to_data(image_response) for src, image_response in zip(srcs, image_responses)
        }
        return imgs

    def _camera_response_to_data(self, response: Any) -> Dict[str, Any]:
        image: np.ndarray = image_response_to_cv2(response, reorient=False)
        fx: float = response.source.pinhole.intrinsics.focal_length.x
        fy: float = response.source.pinhole.intrinsics.focal_length.y
        tf_snapshot = response.shot.transforms_snapshot
        camera_frame: str = response.shot.frame_name_image_sensor
        return {
            "image": image,
            "fx": fx,
            "fy": fy,
            "tf_camera_to_global": self.spot.get_transform(from_frame=camera_frame, tf_snapshot=tf_snapshot),
        }
