# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Tuple

import numpy as np

from vlfm.text_processing.multiresolution import VLPathSelectorMR

from .path_policy import BasePathPolicy


class PathPolicyMR(BasePathPolicy):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._cur_path_idx = 0
        self._pos_since_last: List[List[float]] = []

        self._path_selector: VLPathSelectorMR = VLPathSelectorMR(
            self.args, self._vl_map, min_dist_goal=self._pointnav_stop_radius
        )

    def _reset(self) -> None:
        super()._reset()
        self._path_selector.reset()
        self._cur_path_idx = 0
        self._pos_since_last = []

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
        if self._vl_map.enable_stairs:
            self._stair_preplan_step()

        replan, force_dont_stop, idx_path = self._pre_plan_logic()

        robot_xy = self._observations_cache["robot_xy"]

        self._pos_since_last += [robot_xy]

        ###Path planning
        if replan:
            frontiers = self._observations_cache["frontier_sensor"]
            yaw = self._observations_cache["robot_heading"]

            if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                print("No frontiers found during exploration, stopping.")
                self.why_stop = "No frontiers found"
                return robot_xy, True

            path, path_vals, stop = self._path_selector.get_goal_for_instruction(
                robot_xy,
                yaw,
                frontiers,
                self._instruction,
                np.array(self._pos_since_last),
                force_dont_stop,
                return_full_path=self.args.use_path_waypoints,
            )

            self._pos_since_last = []

            if path is None:  # No valid paths found
                if len(self._path_to_follow) > (self._cur_path_idx + 1):
                    # continue on previously chosen path
                    if self.args.use_path_waypoints:
                        self._cur_path_idx += 1
                    else:
                        self._cur_path_idx = len(self._path_to_follow) - 1
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

            if self.args.use_path_waypoints:
                self._cur_path_idx = 0
            else:
                self._cur_path_idx = len(self._path_to_follow) - 1

            self._last_plan_step = self._num_steps

            if stop:
                if (not force_dont_stop) and (
                    self._num_steps > self._force_dont_stop_until
                ):
                    print("STOPPING (in planner)")
                    self.why_stop = "Path value didn't increase enough"
                    return robot_xy, True  # stop
                else:
                    print("Forced not to stop!")

            elif np.sqrt(np.sum(np.square(self._path_to_follow[len(self._path_to_follow)-1] - robot_xy))) < self._pointnav_stop_radius: 
                print("STOPPING (in planner) because goal is current location")
                self.why_stop = "Planner chose current location as goal"
                return robot_xy, True

        return self._path_to_follow[self._cur_path_idx], False
