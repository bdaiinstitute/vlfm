# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Tuple

import numpy as np

from vlfm.mapping.vlmap import ENABLE_STAIRS
from vlfm.text_processing.multiresolution import VLPathSelectorMR

from .path_policy import FORCE_DONT_STOP_UNTIL, BasePathPolicy


class PathPolicyMR(BasePathPolicy):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._cur_path_idx = 0

        self._path_selector: VLPathSelectorMR = VLPathSelectorMR(
            self._vl_map, min_dist_goal=self._pointnav_stop_radius
        )

    def _reset(self) -> None:
        super()._reset()
        self._cur_path_idx = 0

        self._path_selector.reset()

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
        ###Stair logic, just for working out if we need to switch the level on the map
        if ENABLE_STAIRS:
            self._stair_preplan_step()

        replan, force_dont_stop, idx_path = self._pre_plan_logic()

        ###Path planning
        if replan:
            robot_xy = self._observations_cache["robot_xy"]
            frontiers = self._observations_cache["frontier_sensor"]

            if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                print("No frontiers found during exploration, stopping.")
                self.why_stop = "No frontiers found"
                return robot_xy, True

            if (len(self._path_vals) > 0) and (idx_path != -1):
                cur_path_val = self._path_vals[min(idx_path, len(self._path_vals) - 1)]
                cur_path_len = min(idx_path, len(self._path_vals) - 1) + 1
                cur_path = self._path_to_follow
            else:
                cur_path_val = 0.0
                cur_path_len = 0
                cur_path = []

            path, path_vals, stop = self._path_selector.get_goal_for_instruction(
                robot_xy,
                frontiers,
                self._instruction,
                cur_path_val,
                cur_path_len,
                cur_path,
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

            if stop:
                if (not force_dont_stop) and (self._num_steps > FORCE_DONT_STOP_UNTIL):
                    print("STOPPING (in planner)")
                    self.why_stop = "Path value didn't increase enough"
                    return robot_xy, True  # stop
                else:
                    print("Forced not to stop!")

        return self._path_to_follow[self._cur_path_idx], False
