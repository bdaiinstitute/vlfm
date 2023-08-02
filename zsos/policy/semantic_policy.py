import os
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from zsos.mapping.object_map import ObjectMap
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


ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]
ID_TO_PADDING = {
    "bed": 0.2,
    "couch": 0.15,
}


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class SemanticPolicy(BasePolicy):
    target_object: str = ""
    camera_height: float = 0.88
    depth_image_shape: Tuple[int, int] = (244, 224)
    det_conf_threshold: float = 0.50
    pointnav_stop_radius: float = 0.85
    visualize: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.object_detector = GroundingDINOClient()
        self.pointnav_policy = WrappedPointNavResNetPolicy(
            os.environ["POINTNAV_POLICY_PATH"]
        )
        self.object_map: ObjectMap = ObjectMap(
            min_depth=0.5, max_depth=5.0, hfov=79.0, image_width=640, image_height=480
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
    ) -> Union["PolicyActionData", Tensor]:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """

        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if masks[0] == 0:
            self._reset()
            object_goal = observations["objectgoal"][0].item()
            if isinstance(object_goal, str):
                self.target_object = object_goal
            elif isinstance(object_goal, int):
                self.target_object = ID_TO_NAME[object_goal]
            else:
                raise ValueError("Invalid object goal")

        detections = self._update_object_map(observations)
        goal = self._get_target_object_location()

        if not self.done_initializing:  # Initialize
            pointnav_action = self._initialize()
        elif goal is not None:  # Found target object
            pointnav_action = self._pointnav(
                observations, goal[:2], deterministic=deterministic, stop=True
            )
        else:
            pointnav_action = self._explore(observations)
        self.num_steps += 1

        if HABITAT_BASELINES:
            action_data = PolicyActionData(
                actions=pointnav_action,
                rnn_hidden_states=rnn_hidden_states,
                policy_info=self._get_policy_info(observations, detections),
            )

            return action_data
        else:
            return pointnav_action  # just return the action

    def _initialize(self) -> Tensor:
        self.done_initializing = not self.num_steps < 11
        return TorchActionIDs.TURN_LEFT

    def _explore(self, observations: "TensorDict") -> Tensor:  # noqa: F821
        raise NotImplementedError

    def _get_target_object_location(self) -> Union[None, np.ndarray]:
        try:
            return self.object_map.get_best_object(self.target_object)
        except ValueError:
            # Target object has not been spotted
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

    def _get_object_detections(self, img: np.ndarray) -> ObjectDetections:
        detections = self.object_detector.predict(img, visualize=self.visualize)
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
        if rho_theta[0] < stop_dist and stop:
            return TorchActionIDs.STOP
        action = self.pointnav_policy.act(
            obs_pointnav, masks, deterministic=deterministic
        )
        return action

    def _update_object_map(
        self, observations: "TensorDict"  # noqa: F821
    ) -> ObjectDetections:
        """
        Updates the object map with the detections from the current timestep.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        rgb = observations["rgb"][0].cpu().numpy()
        depth = observations["depth"][0].cpu().numpy()
        x, y = observations["gps"][0].cpu().numpy()
        camera_coordinates = np.array(
            [x, -y, self.camera_height]  # Habitat GPS makes west negative, so flip y
        )
        yaw = observations["compass"][0].item()

        detections = self._get_object_detections(rgb)

        for idx, confidence in enumerate(detections.logits):
            self.object_map.update_map(
                detections.phrases[idx],
                detections.boxes[idx],
                depth,
                camera_coordinates,
                yaw,
                confidence,
            )
        self.object_map.update_explored(camera_coordinates, yaw)

        return detections
