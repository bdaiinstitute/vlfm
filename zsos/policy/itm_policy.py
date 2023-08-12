import os
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
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
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5

    def __init__(
        self, value_map_max_depth: float, value_map_hfov: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.itm = BLIP2ITMClient()
        self.frontier_map: FrontierMap = FrontierMap()
        self.value_map: ValueMap = ValueMap(
            fov=value_map_hfov, max_depth=value_map_max_depth
        )

    def act(self, observations: TensorDict, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        rgb, depth, tf_camera_to_episodic = self._get_object_camera_info(observations)
        text = f"Seems like there is a {self._target_object} ahead."
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

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        base_kwargs = {
            "radius": self._circle_marker_radius,
            "thickness": self._circle_marker_thickness,
        }
        frontiers = observations["frontier_sensor"][0].cpu().numpy()
        for frontier in frontiers:
            marker_kwargs = {"color": self._frontier_color, **base_kwargs}
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected__frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {"color": color, **base_kwargs}
            markers.append((self._last_goal, marker_kwargs))
        policy_info["cost_map"] = cv2.cvtColor(
            self.value_map.visualize(markers), cv2.COLOR_BGR2RGB
        )

        return policy_info

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = observations["frontier_sensor"][0].cpu().numpy()
        rgb = observations["rgb"][0].cpu().numpy()
        text = f"Seems like there is a {self._target_object} ahead."
        self.frontier_map.update(frontiers, rgb, text)
        goal, cosine = self.frontier_map.get_best_frontier()
        os.environ["DEBUG_INFO"] = f"Best frontier: {cosine:.3f}"
        print(f"Step: {self._num_steps} Best frontier: {cosine}")
        pointnav_action = self._pointnav(
            observations, goal[:2], deterministic=True, stop=False
        )

        return pointnav_action
