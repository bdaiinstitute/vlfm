import os
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from zsos.mapping.object_point_cloud_map import ObjectPointCloudMap
from zsos.mapping.obstacle_map import ObstacleMap
from zsos.obs_transformers.utils import image_resize
from zsos.policy.utils.pointnav_policy import (
    WrappedPointNavResNetPolicy,
    rho_theta_from_gps_compass_goal,
)
from zsos.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from zsos.vlm.sam import MobileSAMClient

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from zsos.policy.base_policy import BasePolicy
except ModuleNotFoundError:

    class BasePolicy:
        pass


class BaseObjectNavPolicy(BasePolicy):
    _target_object: str = ""
    _policy_info: Dict[str, Any] = {}
    _detect_target_only: bool = True
    _object_masks: np.ndarray = None  # set by ._update_object_map()
    _stop_action: Tensor = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Union[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None
    ] = None

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        det_conf_threshold: float,
        pointnav_stop_radius: float,
        object_map_min_depth: float,
        object_map_max_depth: float,
        object_map_hfov: float,
        object_map_erosion_size: float,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        *args,
        **kwargs,
    ):
        super().__init__()
        self._object_detector = GroundingDINOClient()
        self._mobile_sam = MobileSAMClient()
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(
            min_depth=object_map_min_depth,
            max_depth=object_map_max_depth,
            hfov=object_map_hfov,
            erosion_size=object_map_erosion_size,
        )
        self._depth_image_shape = tuple(depth_image_shape)
        self._det_conf_threshold = det_conf_threshold
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize

        self._num_steps = 0
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._target_detected = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                fov=object_map_hfov,
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                min_depth=object_map_min_depth,
                max_depth=object_map_max_depth,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
            )

    def _reset(self):
        self._target_object = ""
        self._pointnav_policy.reset()
        self._object_map.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._target_detected = False

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ) -> Tuple[Tensor, Tensor]:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """

        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]

        self._policy_info = {}

        rgb, depth, tf_camera_to_episodic, _ = self._get_object_camera_info(
            observations
        )
        detections = self._update_object_map(rgb, depth, tf_camera_to_episodic)
        position = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        goal = self._get_target_object_location(position)

        if not self._done_initializing:  # Initialize
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            pointnav_action = self._explore(observations)
        else:
            pointnav_action = self._pointnav(
                observations, goal[:2], deterministic=deterministic, stop=True
            )

        self._policy_info = self._get_policy_info(observations, detections)
        self._num_steps += 1

        self._observations_cache = None

        return pointnav_action, rnn_hidden_states

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self, position) -> Union[None, np.ndarray]:
        if self._object_map.has_object(self._target_object):
            self._target_detected = True
            return self._object_map.get_best_object(self._target_object, position)
        else:
            return None

    def _get_policy_info(
        self, observations: "TensorDict", detections: ObjectDetections
    ) -> Dict[str, Any]:
        if self._target_object in self._object_map.clouds:
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": "target: " + self._target_object,
            "gps": str(observations["gps"][0].cpu().numpy()),
            "yaw": np.rad2deg(observations["compass"][0].item()),
            "target_detected": self._target_detected,
            "target_point_cloud": target_point_cloud,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = observations["depth"][0].cpu().numpy() * 255
        annotated_depth = cv2.cvtColor(
            annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )
        if self._object_masks.sum() > 0:
            # If self._object_masks isn't all zero, get the object segmentations and
            # draw them on the rgb and depth images
            contours, _ = cv2.findContours(
                self._object_masks, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            annotated_rgb = cv2.drawContours(
                detections.annotated_frame, contours, -1, (255, 0, 0), 2
            )
            annotated_depth = cv2.drawContours(
                annotated_depth, contours, -1, (255, 0, 0), 2
            )
        else:
            annotated_rgb = observations["rgb"][0].cpu().numpy()
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        detections = self._object_detector.predict(img, visualize=self._visualize)
        if self._detect_target_only:
            detections.filter_by_class([self._target_object])
        detections.filter_by_conf(self._det_conf_threshold)

        return detections

    def _pointnav(
        self,
        observations: "TensorDict",
        goal: np.ndarray,
        deterministic=False,
        stop=False,
    ) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        rho_theta = rho_theta_from_gps_compass_goal(observations, goal)
        obs_pointnav = {
            "depth": image_resize(
                observations["depth"],
                self._depth_image_shape,
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta.unsqueeze(0),
        }
        if rho_theta[0] < self._pointnav_stop_radius and stop:
            return self._stop_action
        action = self._pointnav_policy.act(
            obs_pointnav, masks, deterministic=deterministic
        )
        return action

    def _update_object_map(
        self, rgb: np.ndarray, depth: np.ndarray, tf_camera_to_episodic: np.ndarray
    ) -> ObjectDetections:
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        for idx in range(len(detections.logits)):
            bbox_denorm = detections.boxes[idx] * np.array(
                [width, height, width, height]
            )
            object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())
            self._object_masks[object_mask > 0] = 1
            self._object_map.update_map(
                detections.phrases[idx],
                depth,
                object_mask,
                tf_camera_to_episodic,
            )

        self._object_map.update_explored(tf_camera_to_episodic)

        return detections

    def _get_object_camera_info(
        self, observations: "TensorDict"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The rgb image, depth image, and
                camera transform. The depth image is normalized to be between 0 and 1.
                The camera transform is the transform from the camera to the episodic
                frame, a 4x4 transformation matrix.
        """
        if self._observations_cache is None:
            self._observations_cache = self._extract_from_obs(observations)

        return self._observations_cache

    def _extract_from_obs(
        self, observations: "TensorDict"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError
