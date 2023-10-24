# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.vlfmap import VLFMap

from .base_vln_policy import BaseVLNPolicy


class BasePathPolicy(BaseVLNPolicy):
    _replan_interval = 20

    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected_frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._path_to_follow: List[List[float]] = []
        self._path_vals: List[float] = []
        self._cur_path_idx = 0
        self._last_plan_step = 0

        if self._compute_frontiers:
            self._vl_map = VLFMap(
                obstacle_map=self._obstacle_map,
                min_dist_goal=self._pointnav_stop_radius,
            )

    def _reset(self) -> None:
        super()._reset()
        self._path_to_follow = []
        self._path_vals = []
        self._cur_path_idx = 0
        self._last_plan_step = 0

        self._vl_map.reset()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        self._update_vl_map()
        return super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic,
        )

    def _parse_instruction(self, instruction: str) -> List[str]:
        split_strs = ["\r\n", "\n", "."]
        parsed_instruct = []

        working_list = [instruction]

        for ss in split_strs:
            working_list_new = []
            for instruct in working_list:
                parsed = instruct.split(ss)
                working_list_new += parsed
            working_list = working_list_new

        for instruct in working_list:
            instruct = instruct.strip()
            if instruct != "":
                parsed_instruct += [instruct]

        print("PARSING: ", instruction)
        print("OUPUT: ", parsed_instruct)
        return parsed_instruct

    def _plan(self) -> Tuple[np.ndarray, bool]:
        if self._reached_goal:
            self._cur_path_idx += 1
            self._reached_goal = False

        replan = False
        if self._num_steps > (self._last_plan_step + self._replan_interval):
            replan = True
        if len(self._path_to_follow) < self._cur_path_idx + 1:
            replan = True

        if replan:
            robot_xy = self._observations_cache["robot_xy"]
            frontiers = self._observations_cache["frontier_sensor"]

            if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                print("No frontiers found during exploration, stopping.")
                return robot_xy, True

            cur_instruct = self._instruction_parts[self._curr_instruction_idx]
            if len(self._instruction_parts) > (self._curr_instruction_idx + 1):
                next_instruct = self._instruction_parts[self._curr_instruction_idx + 1]
                last_instruction = False
            else:
                next_instruct = ""
                last_instruction = True

            if len(self._path_vals) > 0:
                cur_path_val = self._path_vals[
                    min(self._cur_path_idx, len(self._path_vals) - 1)
                ]
                cur_path_len = min(self._cur_path_idx, len(self._path_vals) - 1) + 1
                cur_path = self._path_to_follow
            else:
                cur_path_val = 0.0
                cur_path_len = 0
                cur_path = []

            path, path_vals, switch_or_stop = self._vl_map.get_goal_for_instruction(
                robot_xy,
                frontiers,
                cur_instruct,
                next_instruct,
                cur_path_val,
                cur_path_len,
                cur_path,
            )

            if path is None:  # No valid paths found
                if len(self._path_to_follow) > (self._cur_path_idx + 1):
                    # continue on previously chosen path
                    self._cur_path_idx += 1
                    return self._path_to_follow[self._cur_path_idx], False
                return None, False

            self._path_to_follow = path
            self._path_vals = path_vals

            self._cur_path_idx = 0

            self._last_plan_step = self._num_steps

            if switch_or_stop:
                if last_instruction:
                    return robot_xy, True  # stop
                else:
                    self._curr_instruction_idx += 1

        return self._path_to_follow[self._cur_path_idx], False

    def _update_vl_map(self) -> None:
        for rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache[
            "vl_map_rgbd"
        ]:
            self._vl_map.update_map(rgb, depth, tf, min_depth, max_depth, fov)

        self._vl_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _get_policy_info(self) -> Dict[str, Any]:
        policy_info = super()._get_policy_info()

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected_frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))

        policy_info["vl_map"] = cv2.cvtColor(
            self._vl_map.visualize(markers, gt_traj=self.gt_path_for_viz),
            cv2.COLOR_BGR2RGB,
        )

        policy_info["render_below_images"] += ["current instruction part"]
        policy_info["current instruction part"] = self._instruction_parts[
            self._curr_instruction_idx
        ]

        return policy_info
