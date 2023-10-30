# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from argparse import Namespace
from typing import List, Tuple

import numpy as np

from vlfm.mapping.vlfmap import VLFMap

from .base import VLPathSelector


class VLPathSelectorSR(VLPathSelector):
    def __init__(self, options: Namespace, vl_map: VLFMap, min_dist_goal: float = 0.4):
        super().__init__(options, vl_map, min_dist_goal)

    def reset(self) -> None:
        super().reset()

    def get_best_path_instruction(
        self, instruction: str, paths: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns best path, goal, waypoint for local planner, value for path"""
        # Get text embeddings
        if instruction in self._cached_text_embeddings.keys():
            text_embed = self._cached_text_embeddings[instruction]
        else:
            text_embed = self._vl_map._vl_model.get_text_embedding(instruction)
            self._cached_text_embeddings[instruction] = text_embed

        # Note cannot easily vectorize across paths as the paths can have different numbers of points...
        max_value = 0.0
        best_path = None
        best_path_vals = None

        for i in range(len(paths)):
            path = paths[i]

            # Get image embeddings along path
            image_embeddings = self._vl_map.get_embeddings_path(path)

            value, c_similarity, peak_i = self.get_similarity(
                path, image_embeddings, text_embed
            )

            # Update
            if value > max_value:
                max_value = value
                best_path = path[: peak_i + 1].copy()
                best_path_vals = c_similarity[: peak_i + 1]

        # print("BEST PATH: ", best_path)
        # print("BEST SIMILARITY: ", best_path_vals)
        # print("BEST VALUE: ", max_value)

        return best_path, best_path_vals, max_value

    def get_goal_for_instruction(
        self,
        agent_pos: np.ndarray,
        waypoints: np.ndarray,
        cur_instruct: str,
        next_instruct: str,
        last_path_val: float,
        last_path_len: int,
        last_path: List[List[float]],
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            agent_pos (Tuple[float,float]): current agent position
            waypoints (np.ndarray): An array of 2D waypoints to make paths to
            cur_instruct (str): The part of the instruction the agent is currently
                trying to follow
            next_instruct (str): The part of the instruction the agent should follow
                after the current part (empty string if there is no next instruction)
            last_path_val (float): The value for the part of the path we travelled
                since the last time this function was called
            last_path_len (int): The length of the part of the path we travelled
                since the last time this function was called

        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: A tuple of the path,
            the value for the path up to each point along the path,
            and whether to start using the next instruction (or stop if no next)
        """
        paths = self.generate_paths(agent_pos, waypoints)

        if len(paths) == 0:
            return None, None, False

        best_path_curr, best_path_vals_curr, max_value_curr = (
            self.get_best_path_instruction(cur_instruct, paths)
        )

        if best_path_curr is None:
            if next_instruct == "":
                return None, None, False
            best_path_next, best_path_vals_next, max_value_next = (
                self.get_best_path_instruction(next_instruct, paths)
            )
            if best_path_next is None:
                return None, None, False
            return best_path_next, best_path_vals_next, True

        len_curr = len(best_path_vals_curr)

        self._cur_path_val += last_path_val * last_path_len
        self._cur_path_len += last_path_len

        if last_path_len > 0:
            self.ignore_locs = np.append(
                self.ignore_locs.reshape(-1, 2),
                (np.array(last_path)[:last_path_len, :]).reshape(-1, 2),
                axis=0,
            )

        if next_instruct == "":  # Current instruction is the final one
            if self._cur_path_val != 0:
                val_with_part = (max_value_curr * len_curr + self._cur_path_val) / (
                    len_curr + self._cur_path_len
                )
                val_without_part = self._cur_path_val / self._cur_path_len

                should_stop = (
                    (val_with_part - val_without_part) / val_without_part
                ) <= self._thresh_stop
            else:
                should_stop = False
            self._vl_map.set_paths_for_viz([best_path_curr], [(255, 0, 0)])
            return best_path_curr, best_path_vals_curr, should_stop

        else:
            best_path_next, best_path_vals_next, max_value_next = (
                self.get_best_path_instruction(next_instruct, paths)
            )

            switch = False

            if max_value_next > (
                max_value_curr * len_curr + self._cur_path_val * self._prev_val_weight
            ) / (len_curr + self._cur_path_len):
                switch = True
            # We also check if current instruction's best path will not improve much,
            # in case there is a difference in the scale of the value between the
            # current and next instruction that makes it hard to switch with the above check
            elif self._cur_path_val != 0:
                val_with_part = (max_value_curr * len_curr + self._cur_path_val) / (
                    len_curr + self._cur_path_len
                )
                val_without_part = self._cur_path_val / self._cur_path_len

                if (
                    (val_with_part - val_without_part) / val_without_part
                ) <= self._thresh_switch:
                    switch = True

            if switch:
                self._points_started_instructions[next_instruct] = agent_pos
                self._cur_path_val = 0.0
                self._cur_path_len = 0
                self.ignore_locs = np.array([])
                self._vl_map.set_paths_for_viz([best_path_next], [(255, 0, 0)])
                return best_path_next, best_path_vals_next, True
            else:
                self._vl_map.set_paths_for_viz([best_path_curr], [(255, 0, 0)])
                return best_path_curr, best_path_vals_curr, False
