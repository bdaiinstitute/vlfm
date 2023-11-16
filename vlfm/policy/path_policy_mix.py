# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, List, Tuple

import numpy as np

from vlfm.text_processing.multiresolution import VLPathSelectorMR
from vlfm.text_processing.singleresolution import VLPathSelectorSR
from vlfm.text_processing.utils import parse_instruction

from .path_policy import BasePathPolicy


class PathPolicyMix(BasePathPolicy):
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

        self._multires_path_selector: VLPathSelectorMR = VLPathSelectorMR(
            self.args, self._vl_map, min_dist_goal=self._pointnav_stop_radius
        )

        self._path_bits: List[np.ndarray] = []
        self._best_val_so_far = 0.0
        self._best_goal = np.array([0.0, 0.0])
        self._stop_at_goal = False
        self._n_times_worse = 0
        self._n_times_comp = 0

    def _reset(self) -> None:
        super()._reset()
        self._cur_path_idx = 0

        self._path_selector.reset()
        self._multires_path_selector.reset()

        self._path_bits = []
        self._best_val_so_far = 0.0
        self._best_goal = np.array([0.0, 0.0])
        self._stop_at_goal = False
        self._n_times_worse = 0
        self._n_times_comp = 0

    def _parse_instruction(self, instruction: str) -> List[str]:
        parsed_instruct = parse_instruction(instruction, split_strs=["\r\n", "\n", "."])

        print("PARSING: ", instruction)
        print("OUPUT: ", parsed_instruct)

        # print("INSTRUCTION: ", instruction)
        self._multires_path_selector.make_instruction_tree(instruction)

        return parsed_instruct

    def _decide_stop(self) -> None:
        path = np.array([])
        for i in range(len(self._path_bits)):
            path = np.append(path.reshape(-1, 2), self._path_bits[i], axis=0)

        agent_pos = self._observations_cache["robot_xy"]

        if path.shape[0] > 0:
            path_to_curr_loc = self._multires_path_selector.generate_paths(
                path[-1, :], agent_pos.reshape(1, 2), one_path=True
            )
        else:
            path_to_curr_loc = self._multires_path_selector.generate_paths(
                np.array([0.0, 0.0]), agent_pos.reshape(1, 2), one_path=True
            )

        if len(path_to_curr_loc) > 0:
            path1 = np.append(path.reshape(-1, 2), path_to_curr_loc[0], axis=0)

            value, _, _ = self._multires_path_selector.get_path_value_main_loop(
                path1, self._instruction
            )

            self._n_times_comp += 1

            print(
                f"VALUE: {value}, PREV_BEST: {self._best_val_so_far}, RATIO:"
                f" {(self._best_val_so_far - value)/self._best_val_so_far}"
            )

            if (self._n_times_comp >= 4) and (
                value - self._best_val_so_far
            ) / self._best_val_so_far > self.args.mixpolicy.far_thresh:
                self._stop_at_goal = True

            if value > self._best_val_so_far:
                self._best_goal = agent_pos
                self._best_val_so_far = value
                self._n_times_worse = 0

            else:
                # Update old best_value with latest map
                if path.shape[0] > 0:
                    path_to_goal_loc = self._multires_path_selector.generate_paths(
                        path[-1, :], self._best_goal.reshape(1, 2), one_path=True
                    )
                else:
                    path_to_goal_loc = self._multires_path_selector.generate_paths(
                        np.array([0.0, 0.0]),
                        self._best_goal.reshape(1, 2),
                        one_path=True,
                    )

                if len(path_to_goal_loc) > 0:
                    path2 = np.append(path.reshape(-1, 2), path_to_goal_loc[0], axis=0)

                    value2, _, _ = (
                        self._multires_path_selector.get_path_value_main_loop(
                            path2, self._instruction
                        )
                    )

                    if value2 > value:
                        self._best_val_so_far = value2
                    else:
                        self._best_val_so_far = value
                        self._best_goal = agent_pos
                        self._n_times_worse = 0

                if (
                    self._best_val_so_far - value
                ) / self._best_val_so_far > self.args.mixpolicy.far_thresh:
                    self._stop_at_goal = True
                elif (
                    self._best_val_so_far - value
                ) / self._best_val_so_far > self.args.mixpolicy.close_thresh:
                    self._n_times_worse += 1

                    # If we are worse enough times then stop
                    if self._n_times_worse >= self.args.mixpolicy.n_times_worse:
                        self._stop_at_goal = True
                else:
                    self._n_times_worse = 0

    def _plan(self) -> Tuple[np.ndarray, bool]:
        # For last steps just head back to best goal we found
        if (self._num_steps > self.args.mixpolicy.n_steps_goto_goal) and (
            not self._stop_at_goal
        ):
            self._stop_at_goal = True
            self._reached_goal = False

        if self._stop_at_goal:
            print(f"HEADING TO {self._best_goal} THEN WILL STOP!")
            if self._reached_goal:
                return None, True
            return self._best_goal, False

        ###Stair logic, just for working out if we need to switch the level on the map
        if self._vl_map.enable_stairs:
            self._stair_preplan_step()

        replan, force_dont_stop, idx_path = self._pre_plan_logic()

        ###Path planning
        if replan:
            robot_xy = self._observations_cache["robot_xy"]
            frontiers = self._observations_cache["frontier_sensor"]
            yaw = self._observations_cache["robot_heading"]

            ###First check if we should stop
            # I tried this outside the replan but the values are too noisy and it's very slow
            if self._num_steps > self._force_dont_stop_until:
                if not self._stop_at_goal:
                    self._decide_stop()
                    if self._stop_at_goal:
                        self._reached_goal = False  # type:ignore[unreachable]
                if self._stop_at_goal:
                    print(  # type:ignore[unreachable]
                        f"HEADING TO {self._best_goal} THEN WILL STOP!"
                    )
                    if self._reached_goal:  # type:ignore[unreachable]
                        return None, True  # type:ignore[unreachable]
                    return self._best_goal, False

            if np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0:
                print("No frontiers found during exploration, stopping.")
                if self._best_val_so_far > 0:
                    self._stop_at_goal = True
                    return self._best_goal, False
                else:
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
                    return_full_path=True,  # self.args.use_path_waypoints,
                )
            )

            if path is None:  # No valid paths found
                if len(self._path_to_follow) > (self._cur_path_idx + 1):
                    # continue on previously chosen path
                    self._cur_path_idx += 1
                    return self._path_to_follow[self._cur_path_idx], False
                else:
                    self.times_no_paths += 1
                if self.times_no_paths > 5:
                    print(
                        "STOPPING because cannot find paths"
                        f" {self.times_no_paths} times"
                    )
                    if self._best_val_so_far > 0:
                        self._stop_at_goal = True
                        return self._best_goal, False
                    else:
                        return None, True
                else:
                    return None, False
            else:
                self.times_no_paths = 0

            if self.args.use_path_waypoints:
                self._path_to_follow = path
                self._path_vals = path_vals
            else:
                self._path_to_follow = path[-1, :].reshape(1, 2)
                self._path_vals = path_vals[-1]

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
                    self._path_bits += [path]
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
                    # TODO: switch this to go to best goal instead?
                    print("STOPPING (in planner) because goal is current location")
                    self.why_stop = "Planner chose current location as goal"
                    return robot_xy, True

        return self._path_to_follow[self._cur_path_idx], False
