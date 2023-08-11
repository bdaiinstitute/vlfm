import os
from typing import Any, Dict, Tuple, Union

import cv2
from torch import Tensor

from zsos.mapping.frontier_map import FrontierMap
from zsos.mapping.value_map import ValueMap
from zsos.policy.base_objectnav_policy import BaseObjectNavPolicy
from zsos.vlm.blip2itm import BLIP2ITMClient
from zsos.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except ModuleNotFoundError:
    pass


class ITMPolicy(BaseObjectNavPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.itm = BLIP2ITMClient()
        self.frontier_map: FrontierMap = FrontierMap()
        self.value_map: ValueMap = ValueMap(fov=self.hfov, max_depth=self.max_depth)

    def act(self, observations: TensorDict, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        rgb, depth, tf_camera_to_episodic = self._get_detection_camera_info(
            observations
        )
        text = f"Seems like there is a {self.target_object} ahead."
        curr_cosine = self.frontier_map._encode(rgb, text)
        self.value_map.update_map(depth, tf_camera_to_episodic, curr_cosine)

        return super().act(observations, *args, **kwargs)

    def _reset(self):
        super()._reset()
        self.frontier_map.reset()
        self.value_map.reset()

    def _get_policy_info(
        self,
        observations: "TensorDict",
        detections: ObjectDetections,
    ) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(observations, detections)
        color_coords = []
        for frontier in observations["frontier_sensor"][0].cpu().numpy():
            color_coords.append((frontier[:2], (0, 0, 255)))
        color_coords.append((self.last_goal, (0, 255, 0)))
        policy_info["cost_map"] = cv2.cvtColor(
            self.value_map.visualize(color_coords), cv2.COLOR_RGB2BGR
        )

        return policy_info

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = observations["frontier_sensor"][0].cpu().numpy()
        rgb = observations["rgb"][0].cpu().numpy()
        text = f"Seems like there is a {self.target_object} ahead."
        self.frontier_map.update(frontiers, rgb, text)
        goal, cosine = self.frontier_map.get_best_frontier()
        os.environ["DEBUG_INFO"] = f"Best frontier: {cosine:.3f}"
        print(f"Step: {self.num_steps} Best frontier: {cosine}")
        pointnav_action = self._pointnav(
            observations, goal[:2], deterministic=True, stop=False
        )

        return pointnav_action
