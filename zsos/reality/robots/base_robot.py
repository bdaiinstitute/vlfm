from typing import Dict, List

import numpy as np

from zsos.reality.robots.camera_ids import CAM_ID_TO_SHAPE, SHOULD_ROTATE


class BaseRobot:
    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def command_base_velocity(self, ang_vel: float, lin_vel: float):
        raise NotImplementedError

    @property
    def xy_yaw(self) -> np.ndarray:
        """Returns x, y, yaw"""
        raise NotImplementedError

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns current angle for each of the 7 arm joints"""
        raise NotImplementedError

    @staticmethod
    def _reorient_images(imgs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Rotate images if necessary.

        Args:
            imgs_dict: Dictionary of images.
        Returns:
            Dictionary of images, rotated if necessary.
        """
        for camera_id, img in imgs_dict.items():
            if camera_id in SHOULD_ROTATE:
                imgs_dict[camera_id] = np.rot90(img, k=3)
        return imgs_dict


class FakeRobot(BaseRobot):
    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        """
        Return a list of random images. Ensure they are the right shapes. camera_source
        is a list of camera ids that are attributes of SpotCamIds, and its shape is
        stored in CAM_ID_TO_SHAPE.
        """
        for source in camera_source:
            assert source in CAM_ID_TO_SHAPE, f"Invalid camera source: {source}"
        images = {
            source: np.random.rand(*CAM_ID_TO_SHAPE[source]) for source in camera_source
        }
        return self._reorient_images(images)

    def command_base_velocity(self, ang_vel: float, lin_vel: float):
        pass

    @property
    def xy_yaw(self) -> np.ndarray:
        """Returns a random x, y, yaw"""
        return np.random.rand(3)

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns a random angle for each of the 7 arm joints"""
        return np.random.rand(7)
