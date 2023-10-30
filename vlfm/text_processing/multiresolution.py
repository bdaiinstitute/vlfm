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

        self._weight_path = self.args.path_weight_path
        self._weight_sentence = self.args.path_weight_sentence
        self._weight_parts = self.args.path_weight_parts
        self._weight_words = self.args.path_weight_words

        self._thresh_peak_parts_val = self.args.path_thresh_peak_parts_val
        self._thresh_peak_parts_switch = self.args.path_thresh_peak_parts_switch

    def reset(self) -> None:
        super().reset()
        self.instruction_tree = None
        self.path = np.array([])

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

        if self.args.enable_peak_threshold:
            # Get first idx where it is over the threshold
            where_over = total_value > value * self._thresh_peak
            if np.any(where_over):
                peak_i = np.where(where_over)[0][0]
                value = total_value[peak_i]

        return total_value, peak_i, value

    def get_best_path_instruction_full(
        self, instruction: str, paths: List[np.ndarray]
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
        _, _, value_prev_path = self.get_path_value_main_loop(self.path)

        # Note cannot easily vectorize across paths as the paths can have different numbers of points...
        max_value = 0.0
        best_path = None
        best_path_vals = None

        for i in range(len(paths)):
            path = paths[i]

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
                        "PATH HAS LOOP! ", full_path.shape, self.path.shape, path.shape
                    )
                else:
                    full_path = np.append(
                        self.path.reshape(-1, 2), path.reshape(-1, 2), axis=0
                    )
            else:
                full_path = np.append(
                    self.path.reshape(-1, 2), path.reshape(-1, 2), axis=0
                )

            total_value, peak_i, value = self.get_path_value_main_loop(full_path)

            # Update
            if value > max_value:
                if loop_removal_flag:
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
        waypoints: np.ndarray,
        instruction: str,
        last_path: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            agent_pos (Tuple[float,float]): current agent position
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

        paths = self.generate_paths(agent_pos, waypoints)

        if len(paths) == 0:
            print("NO PATHS GENERATED!")
            return None, None, False

        best_path_curr, best_path_vals_curr, val_with_part, val_without_part = (
            self.get_best_path_instruction_full(instruction, paths)
        )

        if best_path_curr is None:
            print("NO BEST PATH CURR")
            return None, None, False

        if best_path_curr.size == 0:
            print("PATH SIZE 0!")
            return None, None, False

        print("PATH VAL DEBUG: ", val_with_part, val_without_part, self.path.shape)

        if val_without_part != 0:
            should_stop = (
                (val_with_part - val_without_part) / val_without_part
            ) <= self._thresh_stop
            print(
                "STOP THRESH: ",
                (val_with_part - val_without_part) / val_without_part,
                self._thresh_stop,
            )
        else:
            should_stop = False

        self._vl_map.set_paths_for_viz([best_path_curr], [(255, 0, 0)])
        return best_path_curr, best_path_vals_curr, should_stop
