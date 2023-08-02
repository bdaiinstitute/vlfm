from typing import Dict, List, Tuple

import numpy as np
from spot_wrapper.spot import Spot, image_response_to_cv2

from zsos.reality.robots.base_robot import BaseRobot

MAX_CMD_DURATION = 5


class BDSWRobot(BaseRobot):
    def __init__(self, spot: Spot):
        self.spot = spot

    def get_camera_images(self, camera_source: List[str]) -> Dict[str, np.ndarray]:
        # Get Spot camera image
        image_responses = self.spot.get_image_responses(camera_source)
        imgs = {
            source: image_response_to_cv2(image_response, reorient=True)
            for source, image_response in zip(camera_source, image_responses)
        }
        return imgs

    def command_base_velocity(self, ang_vel: float, lin_vel: float):
        self.spot.set_base_velocity(
            lin_vel,
            0.0,  # no horizontal velocity
            ang_vel,
            MAX_CMD_DURATION,
        )

    @property
    def xy_yaw(self) -> Tuple[np.ndarray, float]:
        robot_state = self.spot.get_robot_state()
        x, y, yaw = self.spot.get_xy_yaw(robot_state=robot_state)
        return np.array([x, y]), yaw

    @property
    def arm_joints(self) -> np.ndarray:
        """Returns a random angle for each of the 7 arm joints"""
        raise NotImplementedError
