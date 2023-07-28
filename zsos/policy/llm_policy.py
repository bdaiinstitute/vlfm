import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from torch import Tensor

from frontier_exploration.policy import FrontierExplorationPolicy
from zsos.llm.llm import BaseLLM, ClientFastChat
from zsos.mapping.object_map import ObjectMap
from zsos.obs_transformers.resize import image_resize
from zsos.policy.utils.pointnav_policy import (
    WrappedPointNavResNetPolicy,
    rho_theta_from_gps_compass_goal,
)
from zsos.vlm.blip2 import BLIP2Client
from zsos.vlm.fiber import FIBERClient
from zsos.vlm.grounding_dino import GroundingDINOClient, ObjectDetections

ID_TO_NAME = ["chair", "bed", "potted_plant", "toilet", "tv", "couch"]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


@baseline_registry.register_policy
class LLMPolicy(FrontierExplorationPolicy):
    llm: BaseLLM = None
    seen_objects = set()
    visualize: bool = True
    target_object: str = ""
    current_best_object: str = ""
    depth_image_shape: Tuple[int, int] = (244, 224)
    camera_height: float = 0.88
    det_conf_threshold: float = 0.5
    pointnav_stop_radius: float = 0.65

    def __init__(self, *args, **kwargs):
        super().__init__()
        # VL models
        self.object_detector = GroundingDINOClient()
        self.vlm = BLIP2Client()
        self.grounding_model = FIBERClient()

        self.object_map: ObjectMap = ObjectMap(
            min_depth=0.5, max_depth=5.0, hfov=79.0, image_width=640, image_height=480
        )
        self.llm = ClientFastChat()
        self.pointnav_policy = WrappedPointNavResNetPolicy(
            os.environ["POINTNAV_POLICY_PATH"]
        )
        self.last_goal = np.zeros(2)
        self.start_steps = 0

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ) -> PolicyActionData:
        """
        Moves the robot towards one of the following goals:
        1. The target object, if it was spotted
        2. The object that the LLM thinks is the best
        3. The frontier that is closest to the robot

        For now, (3) will not invoke the PointNav policy

        """

        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if masks[0] == 0:
            self._reset()
            self.target_object = ID_TO_NAME[observations[ObjectGoalSensor.cls_uuid][0]]

        # Get action_data from FrontierExplorationPolicy
        action_data = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
        )

        detections = self._update_object_map(observations)

        try:
            # Target object has been spotted
            goal = self.object_map.get_best_object(self.target_object)
        except ValueError:
            # Target object has not been spotted
            goal = None

        # baseline = True
        baseline = False

        if self.start_steps < 12:
            self.start_steps += 1
            pointnav_action = TorchActionIDs.TURN_LEFT
            llm_responses = "Spinning..."
        elif goal is not None:
            pointnav_action = self._pointnav(
                observations, masks, goal[:2], deterministic=deterministic, stop=True
            )
            llm_responses = "Beelining to target!"
        else:
            curr_pos = observations["gps"][0].cpu().numpy() * np.array([1, -1])
            llm_responses = "Closest exploration" if baseline else "LLM exploration"
            if np.linalg.norm(self.last_goal - curr_pos) < 0.25:
                frontiers = observations["frontier_sensor"][0].cpu().numpy()
                if baseline:
                    goal = frontiers[0]
                else:
                    # Ask LLM which waypoint to head to next
                    goal, llm_responses = self._get_llm_goal(curr_pos, frontiers)
            else:
                goal = self.last_goal

            pointnav_action = self._pointnav(
                observations, masks, goal[:2], deterministic=deterministic, stop=False
            )

        action_data.actions = pointnav_action

        action_data.policy_info = self._get_policy_info(
            observations, detections, llm_responses
        )
        print(llm_responses)

        return action_data

    def _get_policy_info(
        self,
        observations: TensorDict,
        detections: ObjectDetections,
        llm_responses: str,
    ) -> List[Dict]:
        policy_info = []
        num_envs = observations["rgb"].shape[0]
        seen_objects_str = ", ".join(self.seen_objects)
        for env_idx in range(num_envs):
            curr_info = {
                "target_object": "target: " + self.target_object,
                "llm_response": "best: " + llm_responses,
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

    def _get_llm_goal(
        self, current_pos: np.ndarray, frontiers: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """
        Asks LLM which object or frontier to go to next. self.object_map is used to
        generate the prompt for the LLM.

        Args:
            current_pos (np.ndarray): A 1D array of shape (2,) containing the current
                position of the robot.
            frontiers (np.ndarray): A 2D array of shape (num_frontiers, 2) containing
                the coordinates of the frontiers.

        Returns:
            Tuple[np.ndarray, str]: A tuple containing the goal and the LLM response.
        """
        prompt, waypoints = self.object_map.get_textual_map_prompt(
            self.target_object, current_pos, frontiers
        )
        resp = self.llm.ask(prompt)
        int_resp = extract_integer(resp) - 1

        try:
            waypoint = waypoints[int_resp]
        except IndexError:
            print("Seems like the LLM returned an invalid response:\n")
            print(resp)
            waypoint = waypoints[-1]

        return waypoint, resp

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
        action = self.pointnav_policy.get_actions(
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

        seen_objects = set(i.class_name for i in self.object_map.map)
        self.seen_objects.update(seen_objects)

        return detections

    def _should_explore(self) -> bool:
        return self.target_object not in self.seen_objects

    def _reset(self):
        self.seen_objects = set()
        self.target_object = ""
        self.current_best_object = ""
        self.pointnav_policy.reset()
        self.object_map.reset()
        self.last_goal = np.zeros(2)
        self.start_steps = 0


def extract_integer(llm_resp: str) -> int:
    """
    Extracts the first integer from the given string.

    Args:
        llm_resp (str): The input string from which the integer is to be extracted.

    Returns:
        int: The first integer found in the input string. If no integer is found,
            returns -1.

    Examples:
        >>> extract_integer("abc123def456")
        123
    """
    digits = []
    for c in llm_resp:
        if c.isdigit():
            digits.append(c)
        elif len(digits) > 0:
            break
    if not digits:
        return -1
    return int("".join(digits))
