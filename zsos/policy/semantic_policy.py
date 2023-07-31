import os
from typing import Dict, List, Union

import numpy as np
import torch
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from torch import Tensor

from zsos.mapping.object_map import ObjectMap
from zsos.obs_transformers.utils import image_resize
from zsos.policy.base_policy import BasePolicy
from zsos.policy.utils.pointnav_policy import (
    WrappedPointNavResNetPolicy,
    rho_theta_from_gps_compass_goal,
)
from zsos.vlm.grounding_dino import GroundingDINOClient, ObjectDetections

ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class SemanticPolicy(BasePolicy):
    target_object: str = ""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.object_detector = GroundingDINOClient()
        self.pointnav_policy = WrappedPointNavResNetPolicy(
            os.environ["POINTNAV_POLICY_PATH"]
        )
        self.object_map: ObjectMap = ObjectMap(
            min_depth=0.5, max_depth=5.0, hfov=79.0, image_width=640, image_height=480
        )

        self.start_steps = 0
        self.last_goal = np.zeros(2)
        self.done_initializing = False

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ) -> PolicyActionData:
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, explores the scene until it finds the target object.
        Once the target object is found, it navigates to the object.
        """

        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if masks[0] == 0:
            self._reset()
            self.target_object = ID_TO_NAME[observations[ObjectGoalSensor.cls_uuid][0]]

        detections = self._update_object_map(observations)
        goal = self._get_target_object_location()

        if not self.done_initializing:  # Initialize
            pointnav_action = self._initialize()
        elif goal is not None:  # Found target object
            pointnav_action = self._pointnav(
                observations, masks, goal[:2], deterministic=deterministic, stop=True
            )
        else:
            pointnav_action = self._explore(observations)
        action_data = PolicyActionData(
            actions=pointnav_action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=self._get_policy_info(observations, detections),
        )

        self.start_steps += 1

        return action_data

    def _initialize(self) -> Tensor:
        self.done_initializing = not self.start_steps < 11
        return TorchActionIDs.TURN_LEFT

    def _explore(self, observations: TensorDict) -> Tensor:
        raise NotImplementedError

    def _get_target_object_location(self) -> Union[None, np.ndarray]:
        try:
            return self.object_map.get_best_object(self.target_object)
        except ValueError:
            # Target object has not been spotted
            return None

    def _get_policy_info(
        self,
        observations: TensorDict,
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
        observations: TensorDict,
        masks: Tensor,
        goal: np.ndarray,
        deterministic=False,
        stop=False,
    ) -> Tensor:
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
        if rho_theta[0] < self.pointnav_stop_radius and stop:
            return TorchActionIDs.STOP
        action = self.pointnav_policy.act(
            obs_pointnav, masks, deterministic=deterministic
        )
        return action

    def _update_object_map(self, observations: TensorDict) -> ObjectDetections:
        """
        Updates the object map with the detections from the current timestep.

        Args:
            observations (TensorDict): The observations from the current timestep.
            detections (ObjectDetections): The detections from the current
                timestep.
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

    def _reset(self):
        self.target_object = ""
        self.pointnav_policy.reset()
        self.object_map.reset()
        self.last_goal = np.zeros(2)
        self.start_steps = 0
