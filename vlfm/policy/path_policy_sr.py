# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Tuple

import numpy as np

from vlfm.text_processing.singleresolution import VLPathSelectorSR
from vlfm.text_processing.utils import parse_instruction

from .path_policy import BasePathPolicy


class PathPolicySR(BasePathPolicy):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._cur_path_idx = 0

        self._path_selector: VLPathSelectorSR = VLPathSelectorSR(
            self.args, self._vl_map, min_dist_goal=self._pointnav_stop_radius
        )

    def _reset(self) -> None:
        super()._reset()
        self._cur_path_idx = 0

        self._path_selector.reset()

    def _parse_instruction(self, instruction: str) -> List[str]:
        parsed_instruct = parse_instruction(
            instruction,
            split_strs=["\r\n", "\n", ".", ",", " and ", " then "],
            # instruction, split_strs=["\r\n", "\n", ".", ",", " then "]
        )

        print("PARSING: ", instruction)
        print("OUPUT: ", parsed_instruct)
        return parsed_instruct

    def _plan(self) -> Tuple[np.ndarray, bool, bool]:
        ###Stair logic, just for working out if we need to switch the level on the map
        if self._vl_map.enable_stairs:
            self._stair_preplan_step()

        robot_xy = self._observations_cache["robot_xy"]

        if self._look_at_frontiers:
            if not (
                np.array_equal(self.frontiers_at_plan, np.zeros((1, 2)))
                or len(self.frontiers_at_plan) == 0
            ):
                dist = np.sqrt(
                    np.sum(
                        np.square(
                            self.frontiers_at_plan
                            - np.tile(
                                robot_xy.reshape(1, 2),
                                (self.frontiers_at_plan.shape[0], 1),
                            )
                        ),
                        axis=1,
                    )
                )
                if np.min(dist) <= self._frontier_dist_thresh:
                    # Reached a frontier. Stop and look around!
                    self._path_to_follow = []
                    return robot_xy, False, True

        if self._look_at_goal and self._reached_goal:
            self._path_to_follow = []
            self._reached_goal = False
            return robot_xy, False, True

        replan, force_dont_stop, idx_path = self._pre_plan_logic()

        ###Path planning
        if replan:
            frontiers = self._observations_cache["frontier_sensor"]
            yaw = self._observations_cache["robot_heading"]

            if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                print("No frontiers found during exploration, stopping.")
                self.why_stop = "No frontiers found"
                return robot_xy, True, False

            cur_instruct = self._instruction_parts[self._curr_instruction_idx]
            if len(self._instruction_parts) > (self._curr_instruction_idx + 1):
                next_instruct = self._instruction_parts[self._curr_instruction_idx + 1]
                last_instruction = False
            else:
                next_instruct = ""
                last_instruction = True

            path, path_vals, switch_or_stop = (
                self._path_selector.get_goal_for_instruction(
                    robot_xy,
                    yaw,
                    frontiers,
                    cur_instruct,
                    next_instruct,
                    return_full_path=self.args.use_path_waypoints,
                    yaw=yaw,
                )
            )

            if path is None:  # No valid paths found
                if len(self._path_to_follow) > (self._cur_path_idx + 1):
                    # continue on previously chosen path
                    self._cur_path_idx += 1
                    return self._path_to_follow[self._cur_path_idx], False, False
                else:
                    self.times_no_paths += 1
                if self.times_no_paths > 10:
                    print(
                        "STOPPING because cannot find paths"
                        f" {self.times_no_paths} times"
                    )
                    self.why_stop = f"No paths found {self.times_no_paths} times"
                    return None, True, False
                else:
                    return None, False, False
            else:
                self.times_no_paths = 0

            self._path_to_follow = path
            self._path_vals = path_vals

            self._cur_path_idx = 0

            self._last_plan_step = self._num_steps

            if switch_or_stop:
                if last_instruction:
                    if (not force_dont_stop) and (
                        self._num_steps > self._force_dont_stop_until
                    ):
                        print("STOPPING (in planner)")
                        self.why_stop = "Path value didn't increase enough"
                        return robot_xy, True, False
                    else:
                        print("Forced not to stop!")
                else:
                    self._curr_instruction_idx += 1
            elif (
                last_instruction
                and (not force_dont_stop)
                and (self._num_steps > self._force_dont_stop_until)
            ):
                if (
                    np.sqrt(
                        np.sum(
                            np.square(
                                self._path_to_follow[len(self._path_to_follow) - 1]
                                - robot_xy
                            )
                        )
                    )
                    < self.args.replanning.goal_stop_dist
                ):
                    print("STOPPING (in planner) because goal is current location")
                    self.why_stop = "Planner chose current location as goal"
                    return robot_xy, True, False

        return self._path_to_follow[self._cur_path_idx], False, False
