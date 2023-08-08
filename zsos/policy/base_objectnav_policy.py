import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from zsos.mapping.object_map import ObjectMap
from zsos.obs_transformers.utils import image_resize
from zsos.policy.utils.pointnav_policy import (
    WrappedPointNavResNetPolicy,
    rho_theta_from_gps_compass_goal,
)
from zsos.utils.geometry_utils import xyz_yaw_to_tf_matrix
from zsos.vlm.grounding_dino import GroundingDINOClient, ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict

    from zsos.policy.base_policy import BasePolicy
except ModuleNotFoundError:

    class BasePolicy:
        pass


class BaseObjectNavPolicy(BasePolicy):
    target_object: str = ""
    camera_height: float = 0.88
    depth_image_shape: Tuple[int, int] = (244, 224)
    det_conf_threshold: float = 0.50
    pointnav_stop_radius: float = 0.85
    visualize: bool = True
    policy_info: Dict[str, Any] = {}
    id_to_padding: Dict[str, float] = {}
    _stop_action: Tensor = None  # must be set by subclass
    # ObjectMap parameters; these must be set by subclass
    min_depth: float = None
    max_depth: float = None
    hfov: float = None
    proximity_threshold: float = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.object_detector = GroundingDINOClient()
        self.pointnav_policy = WrappedPointNavResNetPolicy(
            os.environ["POINTNAV_POLICY_PATH"]
        )
        self.object_map: ObjectMap = ObjectMap(
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            hfov=self.hfov,
            proximity_threshold=self.proximity_threshold,
        )

        self.num_steps = 0
        self.last_goal = np.zeros(2)
        self.done_initializing = False

    def _reset(self):
        self.target_object = ""
        self.pointnav_policy.reset()
        self.object_map.reset()
        self.last_goal = np.zeros(2)
        self.num_steps = 0
        self.done_initializing = False

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
            self.target_object = observations["objectgoal"]

        self.policy_info = {}

        rgb, depth, tf_camera_to_episodic = self._get_detection_camera_info(
            observations
        )
        detections = self._update_object_map(rgb, depth, tf_camera_to_episodic)
        goal = self._get_target_object_location()

        if not self.done_initializing:  # Initialize
            pointnav_action = self._initialize()
        elif goal is None:  # Haven't found target object yet
            pointnav_action = self._explore(observations)
        else:
            pointnav_action = self._pointnav(
                observations, goal[:2], deterministic=deterministic, stop=True
            )

        self.policy_info = self._get_policy_info(observations, detections)
        self.num_steps += 1

        return pointnav_action, rnn_hidden_states

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self) -> Union[None, np.ndarray]:
        try:
            return self.object_map.get_best_object(self.target_object)
        except ValueError:
            # Target object has not been spotted
            return None

    def _get_policy_info(
        self,
        observations: "TensorDict",
        detections: ObjectDetections,
    ) -> Dict[str, Any]:
        seen_objects = set(i.class_name for i in self.object_map.map)
        seen_objects_str = ", ".join(seen_objects)
        policy_info = {
            "target_object": "target: " + self.target_object,
            "visualized_detections": detections.annotated_frame,
            "seen_objects": seen_objects_str,
            "gps": str(observations["gps"][0].cpu().numpy()),
            "yaw": np.rad2deg(observations["compass"][0].item()),
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "target_object",
                "llm_response",
                "seen_objects",
            ],
        }
        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        detections = self.object_detector.predict(img, visualize=self.visualize)
        detections.filter_by_conf(self.det_conf_threshold)

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
        masks = torch.tensor([self.num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self.last_goal):
            self.last_goal = goal
            self.pointnav_policy.reset()
            masks = torch.zeros_like(masks)
        rho_theta = rho_theta_from_gps_compass_goal(observations, goal)
        obs_pointnav = {
            "depth": image_resize(
                observations["depth"],
                self.depth_image_shape,
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta.unsqueeze(0),
        }
        stop_dist = self.pointnav_stop_radius + self.id_to_padding.get(
            self.target_object, 0.0
        )
        if rho_theta[0] < stop_dist and stop:
            return self._stop_action
        action = self.pointnav_policy.act(
            obs_pointnav, masks, deterministic=deterministic
        )
        return action

    def _get_detection_camera_info(self, observations: "TensorDict") -> Tuple:
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_yaw = observations["compass"][0].cpu().item()
        # Habitat GPS makes west negative, so flip y
        camera_position = np.array([x, -y, self.camera_height])
        tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
        return rgb, depth, tf_camera_to_episodic

    def _update_object_map(
        self, rgb: np.ndarray, depth: np.ndarray, tf_camera_to_episodic: np.ndarray
    ) -> ObjectDetections:
        detections = self._get_object_detections(rgb)

        for idx, confidence in enumerate(detections.logits):
            self.object_map.update_map(
                detections.phrases[idx],
                detections.boxes[idx],
                depth,
                tf_camera_to_episodic,
                confidence,
            )

        self.object_map.update_explored(tf_camera_to_episodic)

        return detections
