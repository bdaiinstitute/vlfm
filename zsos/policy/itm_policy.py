import os
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from zsos.mapping.frontier_map import FrontierMap
from zsos.mapping.value_map import ValueMap
from zsos.policy.base_objectnav_policy import BaseObjectNavPolicy
from zsos.policy.utils.acyclic_enforcer import AcyclicEnforcer
from zsos.vlm.blip2itm import BLIP2ITMClient
from zsos.vlm.detections import ObjectDetections

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except ModuleNotFoundError:
    pass


class BaseITMPolicy(BaseObjectNavPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected__frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5

    def __init__(
        self, value_map_max_depth: float, value_map_hfov: float, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient()
        self._value_map: ValueMap = ValueMap(
            fov=value_map_hfov, max_depth=value_map_max_depth
        )

    def _reset(self):
        super()._reset()
        self._value_map.reset()

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
            self._value_map.visualize(markers), cv2.COLOR_BGR2RGB
        )

        return policy_info

    def _update_value_map(self, observations: "TensorDict"):
        # This policy only uses the value map for visualization.
        rgb, depth, tf_camera_to_episodic = self._get_object_camera_info(observations)
        text = f"Seems like there is a {self._target_object} ahead."
        curr_cosine = self._itm.cosine(rgb, text)
        self._value_map.update_map(depth, tf_camera_to_episodic, curr_cosine)


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontier_map: FrontierMap = FrontierMap()

    def act(self, observations: "TensorDict", *args, **kwargs) -> Tuple[Tensor, Tensor]:
        if self._visualize:
            self._update_value_map(observations)
        return super().act(observations, *args, **kwargs)

    def _reset(self):
        super()._reset()
        self.frontier_map.reset()

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


class ITMPolicyV2(BaseITMPolicy):
    _acyclic_enforcer: AcyclicEnforcer = None  # must be set by ._reset()

    def act(self, observations: "TensorDict", *args, **kwargs) -> Tuple[Tensor, Tensor]:
        self._update_value_map(observations)
        return super().act(observations, *args, **kwargs)

    def _reset(self):
        super()._reset()
        self._acyclic_enforcer = AcyclicEnforcer()

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = observations["frontier_sensor"][0].cpu().numpy()
        if np.array_equal(frontiers, np.zeros((1, 2))):
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Step: {self._num_steps} Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(
            observations, best_frontier, deterministic=True, stop=False
        )

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment. Must contain "gps"
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        sorted_pts, sorted_values = self._value_map.sort_waypoints(frontiers, 0.5)

        position = observations["gps"].squeeze(1).cpu().numpy()[0]
        best_frontier, best_value = None, None
        for frontier, value in zip(sorted_pts, sorted_values):
            cyclic = self._acyclic_enforcer.check_cyclic(position, frontier)
            if not cyclic:
                best_frontier, best_value = frontier, value
                break

        if best_frontier is None:
            print("All frontiers are cyclic. Choosing the closest one.")
            best_idx = max(
                range(len(frontiers)),
                key=lambda i: np.linalg.norm(frontiers[i] - position),
            )
            best_frontier, best_value = (frontiers[best_idx], sorted_values[best_idx])

        self._acyclic_enforcer.add_state_action(position, best_frontier)

        return best_frontier, best_value
