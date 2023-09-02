import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from zsos.mapping.object_point_cloud_map import ObjectPointCloudMap
from zsos.mapping.obstacle_map import ObstacleMap
from zsos.obs_transformers.utils import image_resize
from zsos.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from zsos.utils.geometry_utils import rho_theta
from zsos.vlm.coco_classes import COCO_CLASSES
from zsos.vlm.grounding_dino import GroundingDINOClient, ObjectDetections
from zsos.vlm.sam import MobileSAMClient
from zsos.vlm.yolov7 import YOLOv7Client

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
    _observations_cache: Dict[str, Any] = {}

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        det_conf_threshold: float,
        pointnav_stop_radius: float,
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
        self._object_detector = GroundingDINOClient(
            port=os.environ.get("GROUNDING_DINO_PORT", 12181)
        )
        self._coco_object_detector = YOLOv7Client(
            port=os.environ.get("YOLOV7_PORT", 12184)
        )
        self._mobile_sam = MobileSAMClient(port=os.environ.get("SAM_PORT", 12183))
        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._object_map: ObjectPointCloudMap = ObjectPointCloudMap(
            erosion_size=object_map_erosion_size
        )
        self._depth_image_shape = tuple(depth_image_shape)
        self._det_conf_threshold = det_conf_threshold
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._done_initializing = False
        self._target_detected = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
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
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ) -> Tuple[Tensor, Tensor]:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """
        self._pre_step(observations, masks)

        object_map_rgbd = self._observations_cache["object_map_rgbd"]
        detections = [
            self._update_object_map(rgb, depth, tf, min_depth, max_depth, fx, fy)
            for (rgb, depth, tf, min_depth, max_depth, fx, fy) in object_map_rgbd
        ]
        robot_xy = self._observations_cache["robot_xy"]
        goal = self._get_target_object_location(robot_xy)

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            mode = "explore"
            pointnav_action = self._explore(observations)
        else:
            mode = "navigate"
            pointnav_action = self._pointnav(goal[:2], stop=True)

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        self._policy_info = self._get_policy_info(detections[0])  # a little hacky
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._target_object = observations["objectgoal"]
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

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

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        if self._object_map.has_object(self._target_object):
            target_point_cloud = self._object_map.get_target_cloud(self._target_object)
        else:
            target_point_cloud = np.array([])
        policy_info = {
            "target_object": self._target_object,
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "target_detected": self._target_detected,
            "target_point_cloud": target_point_cloud,
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
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
            annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]
        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(
                self._obstacle_map.visualize(), cv2.COLOR_BGR2RGB
            )

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        if self._target_object in COCO_CLASSES:
            detections = self._coco_object_detector.predict(img)
            self._det_conf_threshold = 0.8
        else:
            detections = self._object_detector.predict(img)
            detections.phrases = [
                p.replace("cupboard", "cabinet") for p in detections.phrases
            ]
            if self._target_object == "table" and detections.num_detections == 0:
                detections = self._coco_object_detector.predict(img)
                detections.phrases = [
                    p.replace("dining table", "table") for p in detections.phrases
                ]
            self._det_conf_threshold = 0.6
        if self._detect_target_only:
            detections.filter_by_class([self._target_object])
        detections.filter_by_conf(self._det_conf_threshold)

        return detections

    def _pointnav(self, goal: np.ndarray, stop=False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor(
            [[rho, theta]], device="cuda", dtype=torch.float32
        )
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                self._depth_image_shape,
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        if rho < self._pointnav_stop_radius and stop:
            self._called_stop = True
            return self._stop_action
        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        return action

    def _update_object_map(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> ObjectDetections:
        """
        Updates the object map with the given rgb and depth images, and the given
        transformation matrix from the camera to the episodic coordinate frame.

        Args:
            rgb (np.ndarray): The rgb image to use for updating the object map. Used for
                object detection and Mobile SAM segmentation to extract better object
                point clouds.
            depth (np.ndarray): The depth image to use for updating the object map. It
                is normalized to the range [0, 1] and has a shape of (height, width).
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the
                camera to the episodic coordinate frame.
            min_depth (float): The minimum depth value (in meters) of the depth image.
            max_depth (float): The maximum depth value (in meters) of the depth image.
            fx (float): The focal length of the camera in the x direction.
            fy (float): The focal length of the camera in the y direction.

        Returns:
            ObjectDetections: The object detections from the object detector.
        """
        detections = self._get_object_detections(rgb)
        height, width = rgb.shape[:2]
        self._object_masks = np.zeros((height, width), dtype=np.uint8)
        if np.array_equal(depth, np.ones_like(depth)) and detections.num_detections > 0:
            depth = self._infer_depth(rgb, min_depth, max_depth)
            obs = list(self._observations_cache["object_map_rgbd"][0])
            obs[1] = depth
            self._observations_cache["object_map_rgbd"][0] = tuple(obs)
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
                min_depth,
                max_depth,
                fx,
                fy,
            )

        self._object_map.update_explored(tf_camera_to_episodic)

        return detections

    def _cache_observations(self, observations: "TensorDict"):
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(
        self, rgb: np.ndarray, min_depth: float, max_depth: float
    ) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class ZSOSConfig:
    name: str = "HabitatITMPolicy"
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    depth_image_shape: Tuple[int, int] = (224, 224)
    det_conf_threshold: float = 0.8
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    object_map_erosion_size: int = 5
    exploration_thresh: float = 0.0
    obstacle_map_area_threshold: float = 1.5  # in square meters
    text_prompt: str = "Seems like there is a target_object ahead."
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88

    @classmethod
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(ZSOSConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="zsos_config_base", node=ZSOSConfig())
