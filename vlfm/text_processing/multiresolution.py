# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

from argparse import Namespace
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch

from vlfm.mapping.vlfmap import VLFMap

from .base import VLPathSelector
from .utils import get_closest_vals, get_dist, parse_instruction


class TextType(Enum):
    INSTRUCTION = 0
    SENTENCE = 1
    PART = 2
    WORD = 3


class InstructionTree:
    def __init__(self, text: str, level: TextType) -> None:
        self.text = text
        self.text_type = level
        self.children: List[InstructionTree] = []

    def add_child(self, child: InstructionTree) -> None:
        self.children += [child]


class VLPathSelectorMR(VLPathSelector):
    _loop_dist_prev_path = 0.2
    _loop_dist = 0.3

    def __init__(self, options: Namespace, vl_map: VLFMap, min_dist_goal: float = 0.4):
        super().__init__(options, vl_map, min_dist_goal)
        self.instruction_tree: Optional[InstructionTree] = None
        self.path: np.ndarray = np.array([])

        self._weight_path = self.args.multiresolution.path_weight_path
        self._weight_sentence = self.args.multiresolution.path_weight_sentence
        self._weight_parts = self.args.multiresolution.path_weight_parts
        self._weight_words = self.args.multiresolution.path_weight_words

        self._thresh_peak_parts_val = (
            self.args.multiresolution.path_thresh_peak_parts_val
        )
        self._thresh_peak_parts_switch = (
            self.args.multiresolution.path_thresh_peak_parts_switch
        )

        self._enable_log_success_thresh = self.args.logging.enable_log_success_thresh

        if self._enable_log_success_thresh:
            self.past_thresh: List[Tuple[float, float]] = []  # percentage, abs
            self.past_thresh_is_updated = False

        self._calculate_path_from_origin = (
            self.args.multiresolution.calculate_path_from_origin
        )

        if self._calculate_path_from_origin:
            self.prev_path_value = 1.0

        self._store_points_on_paths = self.args.multiresolution.store_prev_points

        if self._store_points_on_paths:
            self._n_pts_store = self.args.multiresolution.n_points_store
            self._extra_waypoints: np.ndarray = np.array([])

    def reset(self) -> None:
        super().reset()
        self.instruction_tree = None
        self.path = np.array([])
        if self._enable_log_success_thresh:
            self.past_thresh = []
            self.past_thresh_is_updated = False
        if self._calculate_path_from_origin:
            self.prev_path_value = 1.0
        if self._store_points_on_paths:
            self._extra_waypoints = np.array([])

    def get_best_values_for_words(
        self, words: List[str], image_embeddings: torch.tensor
    ) -> np.ndarray:
        # TODO: change this to allow for soft thresholding
        # Instead of a max for each should be max up to each point
        vals = []
        for word in words:
            if word in self._cached_text_embeddings.keys():
                text_embed = self._cached_text_embeddings[word]
            else:
                text_embed = self._vl_map._vl_model.get_text_embedding(word)
                self._cached_text_embeddings[word] = text_embed
            similarity = self._vl_map._vl_model.get_similarity_batch(
                image_embeddings=image_embeddings, txt_embedding=text_embed
            )
            if similarity.size > 1:
                vals += [np.max(similarity)]
            elif similarity.size > 0:
                vals += [similarity[0]]

        return np.array(vals)

    def get_values_instruction_part(
        self, instruction: str, path: np.ndarray, image_embeddings: np.ndarry
    ) -> Tuple[np.ndarray, int, int]:
        # Get text embeddings
        if instruction in self._cached_text_embeddings.keys():
            text_embed = self._cached_text_embeddings[instruction]
        else:
            text_embed = self._vl_map._vl_model.get_text_embedding(instruction)
            self._cached_text_embeddings[instruction] = text_embed

        value, c_similarity, peak_i = self.get_similarity(
            path, image_embeddings, text_embed, thresh=1.0
        )

        peak_i_l = peak_i
        peak_i_h = peak_i

        if c_similarity.size > 0:
            # Get first idx where it is over the threshold
            where_over = c_similarity > value * self._thresh_peak_parts_switch
            if np.any(where_over):
                peak_i_l = np.where(where_over)[0][0]

            where_over = c_similarity > value * self._thresh_peak_parts_val
            if np.any(where_over):
                peak_i_h = np.where(where_over)[0][0]

        return c_similarity, peak_i_h, peak_i_l

    def get_path_value_main_loop(
        self, path: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        # Get image embeddings along path
        image_embeddings = self._vl_map.get_embeddings_path(path)

        assert self.instruction_tree is not None

        best_path_vals_top, _, _ = self.get_values_instruction_part(
            self.instruction_tree.text, path, image_embeddings
        )

        # TODO: allow trade-off in start/stop rather than hard start where previous stopped
        # Although it is hard to work out how to implement, especially for average embeddings

        start_i_s = 0

        total_value = best_path_vals_top * self._weight_path

        # print("PATH")

        for sentence in self.instruction_tree.children:
            bvs, peak_i_h, peak_i_l = self.get_values_instruction_part(
                sentence.text, path[start_i_s:], image_embeddings[start_i_s:, ...]
            )

            start_i_p = start_i_s
            stop_i_p = start_i_s + peak_i_l + 1

            total_value[start_i_s : start_i_s + peak_i_h + 1] += (
                bvs[: peak_i_h + 1] * self._weight_sentence
            )

            # print("SENT: ", peak_i, path.shape[0])

            for part in sentence.children:
                bvp, peak_i_h, peak_i_l = self.get_values_instruction_part(
                    sentence.text,
                    path[start_i_p:stop_i_p],
                    image_embeddings[start_i_p:stop_i_p, ...],
                )
                stop_i_w = start_i_p + peak_i_l + 1

                total_value[start_i_p : start_i_p + peak_i_h + 1] += (
                    bvp[: peak_i_h + 1] * self._weight_parts
                )

                # print("VALS: ", peak_i, path.shape[0])

                words = [child.text for child in part.children]
                bvw = self.get_best_values_for_words(
                    words, image_embeddings[start_i_p:stop_i_w, ...]
                )

                start_i_p = stop_i_w

                total_value[start_i_p:stop_i_w] += np.mean(bvw) * self._weight_words

                # print("VALS WORD: ", bvw)

                if start_i_p >= path.shape[0] or start_i_p >= stop_i_p:
                    # print("breaking! Parts: ", start_i_p, stop_i_p, path.shape[0])
                    break

            start_i_s = stop_i_p

            if start_i_s >= path.shape[0]:
                # print("breaking! Sentences: ", start_i_s, path.shape[0])
                break

        peak_i = np.argmax(total_value)
        value = total_value[peak_i]

        if self.args.similarity_calc.enable_peak_threshold:
            # Get first idx where it is over the threshold
            where_over = total_value > value * self._thresh_peak
            if np.any(where_over):
                peak_i = np.where(where_over)[0][0]
                value = total_value[peak_i]

        return total_value, peak_i, value

    def get_best_path_instruction_full(
        self,
        instruction: str,
        paths: List[np.ndarray],
        path_to_curr_loc: Optional[np.ndarry] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Returns best path, goal, waypoint for local planner, value for path"""

        if self.instruction_tree is None:
            # print("INSTRUCTION: ", instruction)
            self.instruction_tree = InstructionTree(instruction, TextType.INSTRUCTION)
            sentences = parse_instruction(instruction, split_strs=["\r\n", "\n", "."])
            # print("SENTENCES: ", sentences)
            for sentence_s in sentences:
                sentence_it = InstructionTree(sentence_s, TextType.SENTENCE)
                parts = parse_instruction(sentence_s, split_strs=[",", ";"])
                # print("PARTS: ", parts)
                for part_s in parts:
                    part_it = InstructionTree(part_s, TextType.PART)
                    words = parse_instruction(part_s, split_strs=[" "])
                    # print("WORDS: ", words)
                    for word in words:
                        word_it = InstructionTree(word, TextType.WORD)
                        part_it.add_child(word_it)
                    sentence_it.add_child(part_it)
                self.instruction_tree.add_child(sentence_it)

        # Get value for previous part
        if self._calculate_path_from_origin:
            _, _, value_prev_path = self.get_path_value_main_loop(path_to_curr_loc)
        else:
            _, _, value_prev_path = self.get_path_value_main_loop(self.path)

        # Note cannot easily vectorize across paths as the paths can have different numbers of points...
        max_value = 0.0
        best_path = None
        best_path_vals = None

        for i in range(len(paths)):
            path = paths[i]

            if not self._calculate_path_from_origin:
                # Check if path doubles back to a previously visited location
                loop_removal_flag = False
                ignore_idx = 4  # Ignore the first bit as we might be close to the end of the path at the start
                if (self.path.size > 0) and (path.shape[0] > ignore_idx + 1):
                    pp_i, cp_i, dist = get_closest_vals(
                        self.path, path[ignore_idx:].reshape(-1, 2)
                    )
                    if dist < self._loop_dist:
                        full_path = full_path = np.append(
                            self.path[:pp_i, :].reshape(-1, 2),
                            path[: cp_i + ignore_idx, :].reshape(-1, 2),
                            axis=0,
                        )
                        loop_removal_flag = True
                        print(
                            "PATH HAS LOOP! ",
                            full_path.shape,
                            self.path.shape,
                            path.shape,
                        )
                    else:
                        full_path = np.append(
                            self.path.reshape(-1, 2), path.reshape(-1, 2), axis=0
                        )
                else:
                    full_path = np.append(
                        self.path.reshape(-1, 2), path.reshape(-1, 2), axis=0
                    )
            else:
                full_path = path

            total_value, peak_i, value = self.get_path_value_main_loop(full_path)

            # Update
            if value > max_value:
                if self._calculate_path_from_origin:
                    opi = 0
                elif loop_removal_flag:
                    opi = pp_i
                else:
                    opi = self.path.shape[0]

                if peak_i > opi:
                    max_value = value
                    best_path = path[: peak_i - opi + 1, :]
                    best_path_vals = total_value[opi : peak_i + 1]

        return best_path, best_path_vals, max_value, value_prev_path

    def remove_loop(self) -> bool:
        for i in range(self.path.shape[0] - 1, -1, -1):
            for j in range(i):
                dist = get_dist(self.path[i], self.path[j])
                if dist < self._loop_dist_prev_path:
                    self.path = np.append(
                        self.path[:j].reshape(-1, 2),
                        self.path[i:].reshape(-1, 2),
                        axis=0,
                    )
                    return True

        return False

    def get_goal_for_instruction(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        waypoints: np.ndarray,
        instruction: str,
        last_path: np.ndarray,
        force_no_stop: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            agent_pos (np.ndarray): current agent position
            waypoints (np.ndarray): An array of 2D waypoints to make paths to
            instruction (str): The full instruction the agent is trying to follow
            last_path_val (float): The value for the part of the path we travelled
                since the last time this function was called
            last_path_len (int): The length of the part of the path we travelled
                since the last time this function was called

        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: A tuple of the path,
            the value for the path up to each point along the path,
            and whether to start using the next instruction (or stop if no next)
        """
        if self._store_points_on_paths and (self._extra_waypoints.size) > 0:
            waypoints = np.append(
                waypoints, self._extra_waypoints.reshape(-1, 2), axis=0
            )

        if self._calculate_path_from_origin:
            # Add waypoints
            if self._add_directional_waypoints:
                dir_waypoints = self.get_directional_waypoints(agent_pos, agent_yaw)
                waypoints = np.append(waypoints, dir_waypoints, axis=0)

            paths = self.generate_paths(np.array([0.0, 0.0]), waypoints)

        else:
            if last_path.size > 0:
                self.path = np.append(
                    self.path.reshape(-1, 2), last_path.reshape(-1, 2), axis=0
                )
                # Cannot use ignore_locs otherwise previous path value is always 0
                # So need another way of preventing backtracking...

                # Remove loops from saved path
                removing_loops = True
                # print("BEFORE REMOVING LOOPS, PATH: ", self.path)
                while removing_loops:
                    removing_loops = self.remove_loop()

                # print("AFTER REMOVING LOOPS, PATH: ", self.path)

            if self._add_directional_waypoints:
                dir_waypoints = self.get_directional_waypoints(agent_pos, agent_yaw)
                waypoints = np.append(waypoints, dir_waypoints, axis=0)

            paths = self.generate_paths(agent_pos, waypoints)

        if len(paths) == 0:
            print("NO PATHS GENERATED!")
            return None, None, False

        if self._calculate_path_from_origin:
            print("Generating path from origin to current loc")
            path_to_curr_loc = self.generate_paths(
                np.array([0.0, 0.0]), agent_pos.reshape(1, 2), one_path=True
            )
            print("done")
            if len(path_to_curr_loc) > 0:
                best_path_curr, best_path_vals_curr, val_with_part, val_without_part = (
                    self.get_best_path_instruction_full(
                        instruction, paths, path_to_curr_loc[0]
                    )
                )
            else:
                best_path_curr, best_path_vals_curr, val_with_part, val_without_part = (
                    self.get_best_path_instruction_full(
                        instruction, paths, np.array([0.0, 0.0]).reshape(1, 2)
                    )
                )
            val_without_part = (
                self.prev_path_value
            )  # TODO: check if better with this here or in else
            self.prev_path_value = val_with_part

        else:
            best_path_curr, best_path_vals_curr, val_with_part, val_without_part = (
                self.get_best_path_instruction_full(instruction, paths)
            )

        if best_path_curr is None:
            print("NO BEST PATH CURR")
            return None, None, False

        if best_path_curr.size == 0:
            print("PATH SIZE 0!")
            return None, None, False

        if self._store_points_on_paths:
            if best_path_curr.shape[0] - 1 <= self._n_pts_store:
                extra_points = best_path_curr[1:, :]
            else:
                interval = int(np.floor(best_path_curr.shape[0] / (self._n_pts_store)))
                idx = [(i + 1) * interval - 1 for i in range(self._n_pts_store)]
                extra_points = best_path_curr[idx, :]
            self._extra_waypoints = np.append(
                self._extra_waypoints.reshape(-1, 2), extra_points, axis=0
            )

        if self._store_points_on_paths or self._add_directional_waypoints:
            extra_waypoints = np.array([])
            if self._store_points_on_paths:
                extra_waypoints = np.append(
                    extra_waypoints.reshape(-1, 2),
                    self._extra_waypoints.reshape(-1, 2),
                    axis=0,
                )
            if self._add_directional_waypoints:
                extra_waypoints = np.append(
                    extra_waypoints.reshape(-1, 2), dir_waypoints.reshape(-1, 2), axis=0
                )

            self._vl_map.set_extra_waypoints(extra_waypoints)

        if self._enable_log_success_thresh and (not force_no_stop):
            # Check if less than previous values, if not would have already stopped!
            val_a = val_with_part - val_without_part
            val_p = val_a / val_without_part
            is_less = True
            if len(self.past_thresh) > 0:
                for i in range(len(self.past_thresh)):
                    if (
                        val_p > self.past_thresh[i][0]
                        and val_a > self.past_thresh[i][1]
                    ):
                        is_less = False
                        break
            if is_less:
                self.past_thresh += [(val_p, val_a)]
                self.past_thresh_is_updated = True

        print("PATH VAL DEBUG: ", val_with_part, val_without_part, self.path.shape)

        if val_without_part != 0:
            should_stop = (
                (val_with_part - val_without_part) / val_without_part
            ) <= self._thresh_stop and (
                val_with_part - val_without_part
            ) <= self._path_thresh_stop_abs
            print(
                "STOP THRESH: ",
                (val_with_part - val_without_part) / val_without_part,
                self._thresh_stop,
                val_with_part - val_without_part,
            )
        else:
            should_stop = False

        if self._calculate_path_from_origin:
            print("Generating path from current loc to new goal")
            path_to_best_list = self.generate_paths(
                agent_pos, best_path_curr[-1, :].reshape(1, 2), one_path=True
            )
            print("done")
            if len(path_to_best_list) > 0:
                path_to_best = path_to_best_list[0]
                if path_to_best.shape[0] > 1:
                    path_to_best = (path_to_best[1:]).reshape(-1, 2)
            else:
                if len(path_to_curr_loc) > 0:
                    path_to_best = np.append(
                        np.flip(path_to_curr_loc, 0), best_path_curr[1:, :], axis=0
                    )
                else:
                    path_to_best = best_path_curr[-1, :].reshape(1, 2)
            self._vl_map.set_paths_for_viz(
                [best_path_curr, path_to_best], [(255, 0, 0), (0, 0, 255)]
            )

            return path_to_best, best_path_vals_curr, should_stop

        else:
            self._vl_map.set_paths_for_viz([best_path_curr], [(255, 0, 0)])
            return best_path_curr, best_path_vals_curr, should_stop

    def get_last_thresh(self) -> Optional[Tuple[float, float]]:
        if self.past_thresh_is_updated:
            self.past_thresh_is_updated = False
            return self.past_thresh[-1]
        self.past_thresh_is_updated = False
        return None
