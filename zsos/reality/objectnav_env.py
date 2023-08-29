import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from depth_camera_filtering import filter_depth

from zsos.reality.pointnav_env import PointNavEnv
from zsos.reality.robots.camera_ids import SpotCamIds
from zsos.utils.geometry_utils import get_fov, wrap_heading
from zsos.utils.img_utils import reorient_rescale_map, resize_images

LEFT_CROP = 124
RIGHT_CROP = 60
NOMINAL_ARM_POSE = np.deg2rad([0, -170, 120, 0, 75, 0])

VALUE_MAP_CAMS = [
    SpotCamIds.BACK_FISHEYE,
    SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.LEFT_FISHEYE,
    SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.RIGHT_FISHEYE,
    SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.HAND_COLOR,
]

POINT_CLOUD_CAMS = [
    SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME,
    SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME,
    # SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME,
]

ALL_CAMS = list(set(VALUE_MAP_CAMS + POINT_CLOUD_CAMS))


class ObjectNavEnv(PointNavEnv):
    """
    Gym environment for doing the ObjectNav task on the Spot robot in the real world.
    """

    tf_episodic_to_global: np.ndarray = None  # must be set in reset()
    tf_global_to_episodic: np.ndarray = None  # must be set in reset()
    episodic_start_yaw: float = None  # must be set in reset()
    target_object: str = None  # must be set in reset()

    def __init__(self, max_gripper_cam_depth: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_gripper_cam_depth = max_gripper_cam_depth
        # Get the current date and time
        now = datetime.now()
        # Format it into a string in the format MM-DD-HH-MM-SS
        date_string = now.strftime("%m-%d-%H-%M-%S")
        self._vis_dir = f"{date_string}"
        os.makedirs(f"vis/{self._vis_dir}", exist_ok=True)

    def reset(self, goal: Any, *args, **kwargs) -> Dict[str, np.ndarray]:
        assert isinstance(goal, str)
        self.target_object = goal
        # Transformation matrix from where the robot started to the global frame
        self.tf_episodic_to_global: np.ndarray = self.robot.get_transform()
        self.tf_episodic_to_global[2, 3] = 0.0  # Make z of the tf 0.0
        self.tf_global_to_episodic = np.linalg.inv(self.tf_episodic_to_global)
        self.episodic_start_yaw = self.robot.xy_yaw[1]
        return self._get_obs()

    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, Dict]:
        # Parent class only moves the base; if we want to move the gripper camera,
        # we need to do it here
        vis_imgs = []
        for k in ["annotated_rgb", "annotated_depth", "obstacle_map", "value_map"]:
            img = cv2.cvtColor(action["info"][k], cv2.COLOR_RGB2BGR)
            if "map" in k:
                img = reorient_rescale_map(img)
            if k == "annotated_depth" and np.array_equal(img, np.ones_like(img) * 255):
                # Put text in the middle saying "Target not curently detected"
                text = "Target not currently detected"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
                cv2.putText(
                    img,
                    text,
                    (img.shape[1] // 2 - text_size[0] // 2, img.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    1,
                )
            vis_imgs.append(img)
        vis_img = np.hstack(resize_images(vis_imgs, match_dimension="height"))
        time_id = time.time()
        cv2.imwrite(f"vis/{self._vis_dir}/{time_id}.jpg", vis_img)
        cv2.imshow("Visualization", cv2.resize(vis_img, (0, 0), fx=0.5, fy=0.5))
        cv2.waitKey(1)

        if "gripper_camera_pan" in action:
            self._pan_gripper_camera(action["gripper_camera_pan"])
        return super().step(action)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        robot_xy, robot_heading = self._get_gps(), self._get_compass()
        nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd = (
            self._get_camera_obs()
        )
        return {
            "nav_depth": nav_depth,
            "robot_xy": robot_xy,
            "robot_heading": robot_heading,
            "objectgoal": self.target_object,
            "obstacle_map_depths": obstacle_map_depths,
            "value_map_rgbd": value_map_rgbd,
            "object_map_rgbd": object_map_rgbd,
        }

    def _get_camera_obs(self):
        """
        Poll all necessary cameras on the robot and return their images, focal lengths,
        and transforms to the global frame.
        """
        srcs: List[str] = ALL_CAMS
        cam_data = self.robot.get_camera_data(srcs)
        for src in ALL_CAMS:
            tf = self.tf_global_to_episodic @ cam_data[src]["tf_camera_to_global"]
            # a tf that remaps from camera conventions to xyz conventions
            rotation_matrix = np.array(
                [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
            )
            cam_data[src]["tf_camera_to_global"] = np.dot(tf, rotation_matrix)

            img = cam_data[src]["image"]

            # Normalize and filter depth images; don't filter nav depth yet
            if img.dtype == np.uint16:
                if "hand" in src:
                    max_depth = self._max_gripper_cam_depth
                else:
                    max_depth = self._max_body_cam_depth
                img = self._norm_depth(img, max_depth=max_depth)
                # Don't filter nav depth yet
                if "front" not in src:
                    img = filter_depth(img, blur_type=None, recover_nonzero=False)
                cam_data[src]["image"] = img

            if img.dtype == np.uint8:
                if img.ndim == 2 or img.shape[2] == 1:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    cam_data[src]["image"] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        min_depth = 0

        # Object map output is a list of (rgb, depth, tf_camera_to_episodic, min_depth,
        # max_depth, fx, fy) for each of the cameras in POINT_CLOUD_CAMS
        src = SpotCamIds.HAND_COLOR
        rgb = cam_data[src]["image"]
        hand_depth = np.ones(rgb.shape[:2], dtype=np.float32)
        tf = cam_data[src]["tf_camera_to_global"]
        max_depth = self._max_gripper_cam_depth
        fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
        object_map_rgbd = [(rgb, hand_depth, tf, min_depth, max_depth, fx, fy)]

        # Nav depth requires the front two camera images, and they must be rotated
        # to be upright
        f_left = SpotCamIds.FRONTLEFT_DEPTH
        f_right = SpotCamIds.FRONTRIGHT_DEPTH
        nav_cam_data = self.robot.reorient_images(
            {k: cam_data[k]["image"] for k in [f_left, f_right]}
        )
        nav_depth = np.hstack([nav_cam_data[f_right], nav_cam_data[f_left]])
        nav_depth = filter_depth(nav_depth, blur_type=None, set_black_value=1.0)

        # Obstacle map output is a list of (depth, tf, min_depth, max_depth, fx, fy,
        # topdown_fov) for each of the cameras in POINT_CLOUD_CAMS
        obstacle_map_depths = []
        for src in POINT_CLOUD_CAMS:
            depth = cam_data[src]["image"]
            fx, fy = cam_data[src]["fx"], cam_data[src]["fy"]
            tf = cam_data[src]["tf_camera_to_global"]
            if src in [SpotCamIds.FRONTLEFT_DEPTH, SpotCamIds.FRONTRIGHT_DEPTH]:
                fov = get_fov(fy, depth.shape[0])
            else:
                fov = get_fov(fx, depth.shape[1])
            src_data = (depth, tf, min_depth, self._max_body_cam_depth, fx, fy, fov)
            obstacle_map_depths.append(src_data)

        # Value map output is a list of (rgb, depth, tf_camera_to_episodic, min_depth,
        # max_depth, fov) for each of the cameras in VALUE_MAP_CAMS
        value_map_rgbd = []
        value_cam_srcs: List[str] = VALUE_MAP_CAMS + ["hand_depth_estimated"]
        # RGB cameras are at even indices, depth cameras are at odd indices
        value_rgb_srcs = value_cam_srcs[::2]
        value_depth_srcs = value_cam_srcs[1::2]
        for rgb_src, depth_src in zip(value_rgb_srcs, value_depth_srcs):
            rgb = cam_data[rgb_src]["image"]
            if depth_src == "hand_depth_estimated":
                depth = hand_depth
            else:
                depth = cam_data[src]["image"]
            fx = cam_data[rgb_src]["fx"]
            tf = cam_data[rgb_src]["tf_camera_to_global"]
            fov = get_fov(fx, rgb.shape[1])
            src_data = (rgb, depth, tf, min_depth, max_depth, fov)
            value_map_rgbd.append(src_data)

        return nav_depth, obstacle_map_depths, value_map_rgbd, object_map_rgbd

    def _get_gps(self) -> np.ndarray:
        """
        Get the (x, y) position of the robot's base in the episode frame. x is forward,
        y is left.
        """
        global_xy = self.robot.xy_yaw[0]
        start_xy = self.tf_episodic_to_global[:2, 3]
        offset = global_xy - start_xy
        rotation_matrix = np.array(
            [
                [np.cos(-self.episodic_start_yaw), -np.sin(-self.episodic_start_yaw)],
                [np.sin(-self.episodic_start_yaw), np.cos(-self.episodic_start_yaw)],
            ]
        )
        episodic_xy = rotation_matrix @ offset
        return episodic_xy

    def _get_compass(self) -> float:
        """
        Get the yaw of the robot's base in the episode frame. Yaw is measured in radians
        counterclockwise from the z-axis.
        """
        global_yaw = self.robot.xy_yaw[1]
        episodic_yaw = wrap_heading(global_yaw - self.episodic_start_yaw)
        return episodic_yaw

    def _pan_gripper_camera(self, yaw):
        new_pose = np.array(NOMINAL_ARM_POSE)
        new_pose[0] = yaw[0]
        self.robot.set_arm_joints(new_pose, travel_time=self.time_step * 0.9)
