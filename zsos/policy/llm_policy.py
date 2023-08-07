from typing import Tuple

import numpy as np
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from torch import Tensor

from zsos.llm.llm import BaseLLM, ClientFastChat
from zsos.policy.base_objectnav_policy import BaseObjectNavPolicy
from zsos.vlm.blip2 import BLIP2Client
from zsos.vlm.fiber import FIBERClient


@baseline_registry.register_policy
class LLMPolicy(BaseObjectNavPolicy):
    llm: BaseLLM = None
    visualize: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__()
        # VL models
        self.vlm = BLIP2Client()
        self.grounding_model = FIBERClient()
        self.llm = ClientFastChat()

    def _explore(self, observations: TensorDict) -> Tensor:
        curr_pos = observations["gps"][0].cpu().numpy() * np.array([1, -1])
        baseline = True
        if np.linalg.norm(self.last_goal - curr_pos) < 0.25:
            frontiers = observations["frontier_sensor"][0].cpu().numpy()
            if baseline:
                goal = frontiers[0]
            else:
                # Ask LLM which waypoint to head to next
                goal, _ = self._get_llm_goal(curr_pos, frontiers)
        else:
            goal = self.last_goal

        pointnav_action = self._pointnav(
            observations, goal[:2], deterministic=True, stop=False
        )

        return pointnav_action

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
