from typing import Dict, List

import torch
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData

from frontier_exploration.policy import FrontierExplorationPolicy
from zsos.detector.grounding_dino import GroundingDINO, ObjectDetections
from zsos.llm.llm import BaseLLM, ClientFastChat

ID_TO_NAME = ["chair", "bed", "potted_plant", "toilet", "tv", "couch"]


@baseline_registry.register_policy
class LLMPolicy(FrontierExplorationPolicy):
    llm: BaseLLM = None

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.object_detector = GroundingDINO(
            box_threshold=0.35,
            text_threshold=0.25,
            device=torch.device("cuda"),
        )
        # self.llm = ClientVLLM()
        self.llm = ClientFastChat()

        self.seen_objects = set()
        self.visualize = True
        self.target_object = ""
        self.current_best_object = ""

    def act(self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False) -> PolicyActionData:
        if masks[0] == 0:
            self._reset()

        action_data = super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic=deterministic,
        )
        self.target_object = ID_TO_NAME[observations[ObjectGoalSensor.cls_uuid][0]]

        detections = self._get_object_detections(observations)
        llm_responses = self._get_llm_responses()

        action_data.policy_info = self._get_policy_info(observations, detections, llm_responses)

        return action_data

    def _get_policy_info(
        self,
        observations: TensorDict,
        detections: List[ObjectDetections],
        llm_responses: List[str],
    ) -> List[Dict]:
        policy_info = []
        num_envs = observations["rgb"].shape[0]
        seen_objects_str = ", ".join(self.seen_objects)
        for env_idx in range(num_envs):
            policy_info.append(
                {
                    "target_object": "target: " + self.target_object,
                    "visualized_detections": detections[env_idx].annotated_frame,
                    "llm_response": "best: " + llm_responses[env_idx],
                    "seen_objects": seen_objects_str,
                    # don't render these on to the egocentric images when making videos:
                    "render_below_images": [
                        "llm_response",
                        "seen_objects",
                        "target_object",
                    ],
                }
            )

        return policy_info

    def _get_object_detections(self, observations: TensorDict) -> List[ObjectDetections]:
        # observations["rgb"] is shape (N, H, W, 3); we want (N, 3, H, W)
        rgb = observations["rgb"].permute(0, 3, 1, 2)
        rgb = rgb.float() / 255.0  # normalize to [0, 1]

        detections = [self.object_detector.predict(rgb[i], visualize=self.visualize) for i in range(rgb.shape[0])]

        objects = [phrase for det in detections for phrase in det.phrases if phrase in self.object_detector.classes]
        self.seen_objects.update(objects)

        return detections

    def _get_llm_responses(self) -> List[str]:
        if len(self.seen_objects) == 0:
            return [""]

        if self.target_object in self.seen_objects:
            return [self.target_object]

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
        self.current_best_object = choices[extract_integer(llm_resp)]

        return [self.current_best_object]

    def _reset(self):
        self.seen_objects = set()
        self.target_object = ""
        self.current_best_object = ""


def extract_integer(llm_resp: str) -> int:
    """Extracts the first integer from the given string."""
    digits = []
    for c in llm_resp:
        if c.isdigit():
            digits.append(c)
        elif len(digits) > 0:
            break
    return int("".join(digits))
