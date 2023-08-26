import os
from typing import Any, Callable, Dict, List, Tuple, Union

import cv2
import numpy as np
from torch import Tensor

from zsos.mapping.frontier_map import FrontierMap
from zsos.mapping.value_map import ValueMap
from zsos.policy.base_objectnav_policy import BaseObjectNavPolicy
from zsos.policy.utils.acyclic_enforcer import AcyclicEnforcer
from zsos.utils.geometry_utils import closest_point_within_threshold
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
    _acyclic_enforcer: AcyclicEnforcer = None  # must be set by ._reset()
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)
    _second_best_thresh: float = 0.9

    @staticmethod
    def _vis_reduce_fn(i):
        return np.max(i, axis=-1)

    def __init__(
        self,
        text_prompt: str,
        use_max_confidence: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._itm = BLIP2ITMClient()
        self._text_prompt = text_prompt
        self._value_map: ValueMap = ValueMap(
            value_channels=len(text_prompt.split("\n")),
            use_max_confidence=use_max_confidence,
        )

    def _reset(self):
        super()._reset()
        self._value_map.reset()
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_frontier = np.zeros(2)

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = self._observations_cache["frontier_sensor"]
        if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
            print("No frontiers found during exploration, stopping.")
            return self._stop_action
        best_frontier, best_value = self._get_best_frontier(observations, frontiers)
        os.environ["DEBUG_INFO"] = f"Best value: {best_value*100:.2f}%"
        print(f"Best value: {best_value*100:.2f}%")
        pointnav_action = self._pointnav(best_frontier, deterministic=True, stop=False)

        return pointnav_action

    def _get_best_frontier(
        self,
        observations: Union[Dict[str, Tensor], "TensorDict"],
        frontiers: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Returns the best frontier and its value based on self._value_map.

        Args:
            observations (Union[Dict[str, Tensor], "TensorDict"]): The observations from
                the environment.
            frontiers (np.ndarray): The frontiers to choose from, array of 2D points.

        Returns:
            Tuple[np.ndarray, float]: The best frontier and its value.
        """
        sorted_pts, sorted_values = self._sort_frontiers_by_value(
            observations, frontiers
        )
        best_frontier, best_value = None, None

        robot_xy = self._observations_cache["robot_xy"]

        if self._last_value > 0.0:
            closest_index = closest_point_within_threshold(
                sorted_pts, self._last_frontier, threshold=0.5
            )
        else:
            closest_index = -1

        if (
            closest_index != -1
            and self._last_value
            > sorted_values[closest_index] * self._second_best_thresh
        ):
            best_frontier, best_value = (
                sorted_pts[closest_index],
                sorted_values[closest_index],
            )
        else:
            for frontier, value in zip(sorted_pts, sorted_values):
                cyclic = self._acyclic_enforcer.check_cyclic(robot_xy, frontier)
                if not cyclic:
                    best_frontier, best_value = frontier, value
                    break
                print("Suppressed cyclic frontier.")

            if best_frontier is None:
                print("All frontiers are cyclic. Choosing the closest one.")
                best_idx = max(
                    range(len(frontiers)),
                    key=lambda i: np.linalg.norm(frontiers[i] - robot_xy),
                )

                best_frontier, best_value = (
                    frontiers[best_idx],
                    sorted_values[best_idx],
                )

        self._acyclic_enforcer.add_state_action(robot_xy, best_frontier)
        self._last_value = best_value
        self._last_frontier = best_frontier

        return best_frontier, best_value

    def _get_policy_info(self, detections: ObjectDetections) -> Dict[str, Any]:
        policy_info = super()._get_policy_info(detections)

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        base_kwargs = {
            "radius": self._circle_marker_radius,
            "thickness": self._circle_marker_thickness,
        }
        frontiers = self._observations_cache["frontier_sensor"]
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
        policy_info["value_map"] = cv2.cvtColor(
            self._value_map.visualize(markers, reduce_fn=self._vis_reduce_fn),
            cv2.COLOR_BGR2RGB,
        )

        return policy_info

    def _update_value_map(self):
        all_rgb = [i[0] for i in self._observations_cache["value_map_rgbd"]]
        cosines = [
            [
                self._itm.cosine(rgb, p.replace("target_object", self._target_object))
                for p in self._text_prompt.split("\n")
            ]
            for rgb in all_rgb
        ]
        for cosine, (rgb, depth, tf, min_depth, max_depth, fov) in zip(
            cosines, self._observations_cache["value_map_rgbd"]
        ):
            self._value_map.update_map(
                np.array(cosine), depth, tf, min_depth, max_depth, fov
            )

        self._value_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        raise NotImplementedError


class ITMPolicy(BaseITMPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frontier_map: FrontierMap = FrontierMap()

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        if self._visualize:
            self._update_value_map()
        return super().act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    def _reset(self):
        super()._reset()
        self._frontier_map.reset()

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        rgb = self._observations_cache["object_map_rgbd"][0][0]
        text = self._text_prompt.replace("target_object", self._target_object)
        self._frontier_map.update(frontiers, rgb, text)  # type: ignore
        return self._frontier_map.sort_waypoints()


class ITMPolicyV2(BaseITMPolicy):
    _reduce_fn: Callable = np.max

    def act(
        self, observations, rnn_hidden_states, prev_actions, masks, deterministic=False
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        self._update_value_map()
        return super().act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    def _sort_frontiers_by_value(
        self, observations: "TensorDict", frontiers: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        sorted_frontiers, sorted_values = self._value_map.sort_waypoints(
            frontiers, 0.5, reduce_fn=self._reduce_fn
        )
        return sorted_frontiers, sorted_values


class ITMPolicyV3(ITMPolicyV2):
    def __init__(self, exploration_thresh: float, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def select_value(values: Tuple[float, float]) -> float:
            return max(values) if values[0] < exploration_thresh else values[0]

        def visualize_value_map(arr: np.ndarray):
            # Get the values in the first channel
            first_channel = arr[:, :, 0]
            # Get the max values across the two channels
            max_values = np.max(arr, axis=2)
            # Create a boolean mask where the first channel is above the threshold
            mask = first_channel > exploration_thresh
            # Use the mask to select from the first channel or max values
            result = np.where(mask, first_channel, max_values)

            return result

        self._reduce_fn = select_value
        self._vis_reduce_fn = visualize_value_map
