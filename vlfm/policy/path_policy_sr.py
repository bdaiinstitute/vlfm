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
        parsed_instruct = parse_instruction(instruction, split_strs=["\r\n", "\n", "."])

        print("PARSING: ", instruction)
        print("OUPUT: ", parsed_instruct)
        return parsed_instruct

    def _plan(self) -> Tuple[np.ndarray, bool]:
        ###Stair logic, just for working out if we need to switch the level on the map
        if self._vl_map.enable_stairs:
            self._stair_preplan_step()

        replan, force_dont_stop, idx_path = self._pre_plan_logic()

        ###Path planning
        if replan:
            robot_xy = self._observations_cache["robot_xy"]
            frontiers = self._observations_cache["frontier_sensor"]
            yaw = self._observations_cache["robot_heading"]

            if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                print("No frontiers found during exploration, stopping.")
                self.why_stop = "No frontiers found"
                return robot_xy, True

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
                )
            )

            if path is None:  # No valid paths found
                if len(self._path_to_follow) > (self._cur_path_idx + 1):
                    # continue on previously chosen path
                    self._cur_path_idx += 1
                    return self._path_to_follow[self._cur_path_idx], False
                else:
                    self.times_no_paths += 1
                if self.times_no_paths > 10:
                    print(
                        "STOPPING because cannot find paths"
                        f" {self.times_no_paths} times"
                    )
                    self.why_stop = f"No paths found {self.times_no_paths} times"
                    return None, True
                else:
                    return None, False
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
                        return robot_xy, True  # stop
                    else:
                        print("Forced not to stop!")
                else:
                    self._curr_instruction_idx += 1

        return self._path_to_follow[self._cur_path_idx], False
