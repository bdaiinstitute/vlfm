import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from zsos.mapping.object_map import ObjectMap, xyz_yaw_to_tf_matrix
from zsos.obs_transformers.utils import image_resize
from zsos.policy.utils.pointnav_policy import (
    WrappedPointNavResNetPolicy,
    rho_theta_from_gps_compass_goal,
)
from zsos.vlm.grounding_dino import GroundingDINOClient, ObjectDetections

try:
    from habitat_baselines.rl.ppo.policy import PolicyActionData

    from zsos.policy.base_policy import BasePolicy

    HABITAT_BASELINES = True
except ModuleNotFoundError:

    class BasePolicy:
        pass

    HABITAT_BASELINES = False


ID_TO_PADDING = {
    "bed": 0.2,
    "couch": 0.15,
}


class SemanticPolicy(BasePolicy):
    target_object: str = None  # must be set by ._reset() method
    camera_height: float = 0.88
    depth_image_shape: Tuple[int, int] = (224, 224)
    det_conf_threshold: float = 0.50
    pointnav_stop_radius: float = 0.85
    visualize: bool = True
    discrete_actions: bool = True
    # Object map parameters:
    min_depth: float = 0.5
    max_depth: float = 5.0
    hfov: float = 79.0
    proximity_threshold: float = 1.5

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.object_detector = GroundingDINOClient()
        self.pointnav_policy = WrappedPointNavResNetPolicy(
            os.environ["POINTNAV_POLICY_PATH"], discrete_actions=self.discrete_actions
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
        self,
        observations: Dict[str, Tensor],
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic=False,
    ) -> Tensor:
        """First, starts the episode by 'initializing' and allowing robot to get its
        bearings (e.g., spinning in place to get a good view of the scene). Second,
        explores the scene until it finds the target object. Finally, once the target
        object is found, navigate to it.

        Args:
            observations (Dict[str, Tensor]): observations from the environment.
            rnn_hidden_states (Any): Not used but needs to be in the signature for
                Habitat evaluation.
            prev_actions (Any): Not used but needs to be in the signature for Habitat
                evaluation.
            masks (Tensor): 0 or 1 indicating whether the episode just
                started (0 for just started).
        """

        assert masks.shape[1] == 1, "Only one environment at a time is supported."
        if masks[0] == 0:
            self._reset()
            self.target_object = observations["objectgoal"]

        self._update_object_map(observations)
        goal = self._get_target_object_location()

        if not self.done_initializing:  # Initialize
            pointnav_action = self._initialize()
        elif goal is None:  # Target object not found yet
            pointnav_action = self._explore(observations)
        else:
            pointnav_action = self._pointnav(
                observations, goal[:2], deterministic=deterministic, stop=True
            )
        self.num_steps += 1

        return pointnav_action

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _explore(self, observations: "TensorDict") -> Tensor:  # noqa: F821
        raise NotImplementedError

    def _get_target_object_location(self) -> Union[None, np.ndarray]:
        try:
            return self.object_map.get_best_object(self.target_object)
        except ValueError:  # Target object has not been spotted
            return None

    def _get_policy_info(
        self,
        observations: "TensorDict",  # noqa: F821
        detections: ObjectDetections,
    ) -> List[Dict]:
        policy_info = []
        num_envs = observations["rgb"].shape[0]
        seen_objects = set(i.class_name for i in self.object_map.map)
        seen_objects_str = ", ".join(seen_objects)
        for env_idx in range(num_envs):
            curr_info = {
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
                curr_info["render_below_images"].append("debug")
                curr_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]
            policy_info.append(curr_info)

        return policy_info

    def _get_object_detections(self, rgb_img: np.ndarray) -> ObjectDetections:
        detections = self.object_detector.predict(rgb_img, visualize=self.visualize)
        detections.filter_by_conf(self.det_conf_threshold)

        return detections

    def _pointnav(
        self,
        observations: "TensorDict",  # noqa: F821
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
        stop_dist = self.pointnav_stop_radius + ID_TO_PADDING.get(
            self.target_object, 0.0
        )

        print(
            "rho",
            rho_theta[0].cpu().item(),
            "theta",
            np.rad2deg(rho_theta[1].cpu().item()),
        )
        # quit()

        if rho_theta[0] < stop_dist and stop:
            return torch.zeros(1, dtype=torch.long, device="cuda")
        action = self.pointnav_policy.act(
            obs_pointnav, masks, deterministic=deterministic
        )
        return action

    def _update_object_map(self, observations: "TensorDict") -> None:  # noqa: F821
        """
        Updates the object map with the detections from the current timestep.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """

        if "tf_camera_to_episodic" in observations:
            # Meant for execution on the real robot, where object-search camera
            # (e.g., gripper) and navigation camera (e.g., front cameras) are not the
            # same
            tf_camera_to_episodic = (
                observations["tf_camera_to_episodic"][0].cpu().numpy()
            )
        else:
            x, y = observations["gps"][0].cpu().numpy()
            camera_position = np.array([x, -y, self.camera_height]) * np.array(
                [1, -1, 1]  # Habitat GPS makes west negative, so flip y
            )
            camera_yaw = observations["compass"][0].cpu().item()
            tf_camera_to_episodic = xyz_yaw_to_tf_matrix(camera_position, camera_yaw)

        if "camera_depth" in observations:
            # Meant for execution on the real robot, where object-search camera
            # (e.g., gripper) and navigation camera (e.g., front cameras) are not the
            # same
            camera_depth = observations["camera_depth"][0].cpu().numpy()
        else:
            camera_depth = observations["depth"][0].cpu().numpy()

        rgb = observations["rgb"][0].cpu().numpy()
        detections = self._get_object_detections(rgb)

        for idx, confidence in enumerate(detections.logits):
            self.object_map.update_map(
                detections.phrases[idx],
                detections.boxes[idx],
                camera_depth,
                tf_camera_to_episodic,
                confidence,
            )

        self.object_map.update_explored(tf_camera_to_episodic)
