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

        # Get action_data from FrontierExplorationPolicy
        action_data = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
        )

        self.target_object = ID_TO_NAME[observations[ObjectGoalSensor.cls_uuid][0]]

        image_numpy = observations["rgb"][0].cpu().numpy()
        detections = self._get_object_detections(image_numpy)
        self._update_object_map(observations, detections)
        llm_responses = self._get_llm_responses()

        if self.start_steps < 12:
            self.start_steps += 1
            pointnav_action = TorchActionIDs.TURN_LEFT
        elif self._should_explore():
            pointnav_action = action_data.actions
        else:
            goal = self.object_map.get_best_object(self.target_object)
            # PointNav only cares about x, y
            pointnav_action = self._pointnav(
                observations, masks, goal[:2], deterministic=deterministic
            )
        action_data.actions = pointnav_action

        action_data.policy_info = self._get_policy_info(
            observations, detections, llm_responses
        )

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
        objects = self._extract_detected_names(detections)

        self.seen_objects.update(objects)

        return detections

    def _extract_detected_names(self, detections: ObjectDetections) -> List[str]:
        # Filter out detections that are not verbatim in the classes.txt file
        objects = [
            phrase
            for phrase in detections.phrases
            if phrase in self.object_detector.classes
        ]
        return objects

    def _get_llm_responses(self) -> str:
        """
        Asks LLM which object to go to next, conditioned on the target object.

        Returns:
            List[str]: A list containing the responses generated by the LLM.
        """
        if len(self.seen_objects) == 0:
            return ""

        if self.target_object in self.seen_objects:
            return self.target_object

        choices = list(self.seen_objects)
        choices_str = ""
        for i, category in enumerate(choices):
            choices_str += f"{i}. {category}\n"

        prompt = (
            "Question: Which object category from the following options would be most "
            f"likely to be found near a '{self.target_object}'?\n\n"
            f"{choices_str}"
            "\nYour response must be ONLY ONE integer (ex. '0', '15', etc.).\n"
            "Answer: "
        )

        llm_resp = self.llm.ask(prompt)
        obj_idx = extract_integer(llm_resp)
        if obj_idx != -1:
            self.current_best_object = choices[obj_idx]
            # TODO: IndexError here

        return self.current_best_object

    def _pointnav(
        self,
        observations: TensorDict,
        masks: Tensor,
        goal: np.ndarray,
        deterministic=False,
    ) -> Tensor:
        if not np.array_equal(goal, self.last_goal):
            self.last_goal = goal
            self.pointnav_policy.reset()
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
        if rho_theta[0] < self.pointnav_stop_radius:
            return TorchActionIDs.STOP
        action = self.pointnav_policy.get_actions(
            obs_pointnav, masks, deterministic=deterministic
        )
        return action

    def _update_object_map(
        self, observations: TensorDict, detections: ObjectDetections
    ) -> None:
        """
        Updates the object map with the detections from the current timestep.

        Args:
            observations (TensorDict): The observations from the current timestep.
            detections (ObjectDetections): The detections from the current
                timestep.
        """
        depth = observations["depth"][0].cpu().numpy()
        camera_coordinates = np.array(
            [*observations["gps"][0].cpu().numpy(), self.camera_height]
        )
        yaw = observations["compass"][0].item()

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
