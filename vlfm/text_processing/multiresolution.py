# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import torch

from vlfm.mapping.vlfmap import VLFMap

from .base import USE_PEAK_THRESHOLD, VLPathSelector
from .utils import parse_instruction


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
    _weight_path = 1.0
    _weight_sentence = 0.6
    _weight_parts = 0.3
    _weight_words = 0.6

    _thresh_peak_parts = 0.7

    def __init__(self, vl_map: VLFMap, min_dist_goal: float = 0.4):
        super().__init__(vl_map, min_dist_goal)
        self.instruction_tree: Optional[InstructionTree] = None

    def reset(self) -> None:
        super().reset()
        self.instruction_tree = None

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
    ) -> Tuple[np.ndarray, int]:
        # Get text embeddings
        if instruction in self._cached_text_embeddings.keys():
            text_embed = self._cached_text_embeddings[instruction]
        else:
            text_embed = self._vl_map._vl_model.get_text_embedding(instruction)
            self._cached_text_embeddings[instruction] = text_embed

        _, c_similarity, peak_i = self.get_similarity(
            path, image_embeddings, text_embed, thresh=self._thresh_peak_parts
        )

        return c_similarity, peak_i

    def get_best_path_instruction_full(
        self, instruction: str, paths: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
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

        # Note cannot easily vectorize across paths as the paths can have different numbers of points...
        max_value = 0.0
        best_path = None
        best_path_vals = None

        for i in range(len(paths)):
            path = paths[i]
            # Get image embeddings along path
            image_embeddings = self._vl_map.get_embeddings_path(path)

            best_path_vals_top, _ = self.get_values_instruction_part(
                self.instruction_tree.text, path, image_embeddings
            )

            np.array([])
            np.array([])

            # TODO: allow trade-off in start/stop rather than hard start where previous stopped
            # Although it is hard to work out how to implement, especially for average embeddings

            start_i_s = 0

            total_value = best_path_vals_top * self._weight_path

            # print("PATH")

            for sentence in self.instruction_tree.children:
                bvs, peak_i = self.get_values_instruction_part(
                    sentence.text, path[start_i_s:], image_embeddings[start_i_s:, ...]
                )
                np.array([])
                np.array([])

                start_i_p = start_i_s
                stop_i_p = start_i_s + peak_i + 1

                total_value[start_i_s:stop_i_p] += (
                    bvs[: peak_i + 1] * self._weight_sentence
                )

                # print("SENT: ", peak_i, path.shape[0])

                for part in sentence.children:
                    bvp, peak_i = self.get_values_instruction_part(
                        sentence.text,
                        path[start_i_p:stop_i_p],
                        image_embeddings[start_i_p:stop_i_p, ...],
                    )
                    stop_i_w = start_i_p + peak_i + 1

                    total_value[start_i_p:stop_i_w] += (
                        bvp[: peak_i + 1] * self._weight_parts
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

            if USE_PEAK_THRESHOLD:
                # Get first idx where it is over the threshold
                where_over = total_value > value * self._thresh_peak
                if np.any(where_over):
                    peak_i = np.where(where_over)[0][0]
                    value = total_value[peak_i]

            # Update
            if value > max_value:
                max_value = value
                best_path = path[:peak_i, :]
                best_path_vals = total_value[:peak_i]

        return best_path, best_path_vals, max_value

    def get_goal_for_instruction(
        self,
        agent_pos: np.ndarray,
        waypoints: np.ndarray,
        instruction: str,
        last_path_val: float,
        last_path_len: int,
        last_path: List[List[float]],
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
        paths = self.generate_paths(agent_pos, waypoints)

        if len(paths) == 0:
            return None, None, False

        best_path_curr, best_path_vals_curr, max_value_curr = (
            self.get_best_path_instruction_full(instruction, paths)
        )

        if best_path_curr is None:
            return None, None, False

        len_curr = len(best_path_vals_curr)

        self._cur_path_val += last_path_val * last_path_len
        self._cur_path_len += last_path_len

        if last_path_len > 0:
            self.ignore_locs = np.append(
                self.ignore_locs.reshape(-1, 2),
                (np.array(last_path)[:last_path_len, :]).reshape(-1, 2),
                axis=0,
            )

        if self._cur_path_val != 0:
            val_with_part = (max_value_curr * len_curr + self._cur_path_val) / (
                len_curr + self._cur_path_len
            )
            val_without_part = self._cur_path_val / self._cur_path_len

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
