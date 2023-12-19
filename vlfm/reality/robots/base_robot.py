# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, List, Tuple

import numpy as np

from .camera_ids import CAM_ID_TO_SHAPE, SHOULD_ROTATE
from .frame_ids import SpotFrameIds


class BaseRobot:
    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Returns [x, y], yaw"""
        raise NotImplementedError

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns current angle for each of the 6 arm joints in radians"""
        raise NotImplementedError

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """Returns a dict of images mapping camera ids to images

        Args:
            camera_source (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        raise NotImplementedError

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        """Commands the base to execute given angular/linear velocities, non-blocking

        Args:
            ang_vel (float): Angular velocity in radians per second
            lin_vel (float): Linear velocity in meters per second
        """
        raise NotImplementedError

    def get_transform(self, frame: str = SpotFrameIds.BODY) -> np.ndarray:
        """Returns the transformation matrix of the robot's base (body) or a link

        Args:
            frame (str, optional): Frame to get the transform of. Defaults to
                SpotFrameIds.BODY.

        Returns:
            np.ndarray: 4x4 transformation matrix
        """
        raise NotImplementedError

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        """Moves each of the 6 arm joints to the specified angle

        Args:
            joints (np.ndarray): Array of 6 angles in radians
            travel_time (float): Time in seconds to reach the specified angles
        """
        raise NotImplementedError

    def open_gripper(self) -> None:
        """Opens the gripper"""
        raise NotImplementedError

    @staticmethod
    def reorient_images(imgs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Rotate images if necessary.

        Args:
            imgs_dict: Dictionary of images.
        Returns:
            Dictionary of images, rotated if necessary.
        """
        imgs_dict_copy = imgs_dict.copy()
        for camera_id, img in imgs_dict_copy.items():
            if camera_id in SHOULD_ROTATE:
                imgs_dict_copy[camera_id] = np.rot90(img, k=3)
        return imgs_dict_copy


class FakeRobot(BaseRobot):
    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        """Returns a random x, y, yaw"""
        x, y, yaw = np.random.rand(3)
        return np.array([x, y]), yaw

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns a random angle for each of the 7 arm joints"""
        return np.random.rand(6)

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """Return a list of random images. Ensure they are the right shapes.

        Args:
            camera_source (List[str]): List of camera ids to get images from

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping camera ids to images
        """
        for source in camera_source:
            assert source in CAM_ID_TO_SHAPE, f"Invalid camera source: {source}"
        images = {source: np.random.rand(*CAM_ID_TO_SHAPE[source]) for source in camera_source}
        return self.reorient_images(images)

    def command_base_velocity(self, ang_vel: float, lin_vel: float) -> None:
        pass

    def get_transform(self, frame: str = SpotFrameIds.BODY) -> np.ndarray:
        return np.eye(4)

    def set_arm_joints(self, joints: np.ndarray, travel_time: float) -> None:
        pass

    def open_gripper(self) -> None:
        pass

    def get_camera_data(self, src: List[str]) -> Dict[str, np.ndarray]:
        raise NotImplementedError
