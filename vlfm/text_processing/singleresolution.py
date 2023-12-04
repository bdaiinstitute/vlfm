# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from argparse import Namespace
from typing import Tuple

import numpy as np

from vlfm.mapping.vlfmap import VLFMap

from .base import VLPathSelector


class VLPathSelectorSR(VLPathSelector):
    def __init__(self, options: Namespace, vl_map: VLFMap, min_dist_goal: float = 0.4):
        super().__init__(options, vl_map, min_dist_goal)

    def reset(self) -> None:
        super().reset()

    def get_path_value_main_loop(
        self, path: np.ndarray, instruction: str
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        image_embeddings = self._vl_map.get_embeddings_path(path)

        if instruction in self._cached_text_embeddings.keys():
            text_embed = self._cached_text_embeddings[instruction]
        else:
            text_embed = self._vl_map._vl_model.get_text_embedding(
                instruction, head="embed"
            )
            self._cached_text_embeddings[instruction] = text_embed

        return self.get_similarity(path, image_embeddings, text_embed)

    def get_goal_for_instruction(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        waypoints: np.ndarray,
        cur_instruct: str,
        next_instruct: str,
        return_full_path: bool = False,
        return_chosen_path: bool = False,
        yaw: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            agent_pos (Tuple[float,float]): current agent position
            waypoints (np.ndarray): An array of 2D waypoints to make paths to
            cur_instruct (str): The part of the instruction the agent is currently
                trying to follow
            next_instruct (str): The part of the instruction the agent should follow
                after the current part (empty string if there is no next instruction)

        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: A tuple of the path,
            the value for the path up to each point along the path,
            and whether to start using the next instruction (or stop if no next)
        """
        # Add waypoints
        if self._add_directional_waypoints:
            dir_waypoints = self.get_directional_waypoints(agent_pos, agent_yaw)
            waypoints = np.append(waypoints, dir_waypoints.reshape(-1, 2), axis=0)

            # Visualize
            self._vl_map.set_extra_waypoints(dir_waypoints.reshape(-1, 2))

        paths = self.generate_paths(np.array([0.0, 0.0]), waypoints)

        if len(paths) == 0:
            print("NO PATHS GENERATED!")
            return None, None, False

        if self._enable_shape_sim:
            caption = (
                "Map showing the path (in pink) that you'd take to follow this"
                f" instruction: {cur_instruct}"
            )
            if caption in self._cached_text_embeddings.keys():
                text_embed_shape = self._cached_text_embeddings[caption]
            else:
                text_embed_shape = self._vl_map._vl_model.get_text_embedding(
                    caption, head="img"
                )
                self._cached_text_embeddings[caption] = text_embed_shape
            obstacle_map_clean = self.get_obstacle_map_clean()
        else:
            obstacle_map_clean = None
            text_embed_shape = None

        if cur_instruct in self._points_started_instructions.keys():
            path_to_curr_loc = self.generate_paths(
                self._points_started_instructions[cur_instruct],
                agent_pos.reshape(1, 2),
                one_path=True,
            )
        else:
            path_to_curr_loc = self.generate_paths(
                np.array([0.0, 0.0]), agent_pos.reshape(1, 2), one_path=True
            )

        if len(path_to_curr_loc) > 0:
            best_path_curr, best_path_vals_curr, val_with_part, val_without_part = (
                self.get_best_path_instruction(
                    cur_instruct,
                    paths,
                    path_to_curr_loc[0],
                    obstacle_map_clean,
                    text_embed_shape,
                )
            )
        else:
            best_path_curr, best_path_vals_curr, val_with_part, val_without_part = (
                self.get_best_path_instruction(
                    cur_instruct,
                    paths,
                    np.array([0.0, 0.0]).reshape(1, 2),
                    obstacle_map_clean,
                    text_embed_shape,
                )
            )
            val_without_part = (
                self.prev_path_value
            )  # TODO: check if better with this here or in else
        self.prev_path_value = val_with_part

        if best_path_curr is None:
            print("NO BEST PATH CURR")
            return None, None, False

        if best_path_curr.size == 0:
            print("PATH SIZE 0!")
            return None, None, False

        if next_instruct == "":  # Current instruction is the final one
            if val_without_part != 0:
                should_stop = (
                    (val_with_part - val_without_part) / val_without_part
                ) <= self._thresh_stop and (
                    val_with_part - val_without_part
                ) <= self._path_thresh_stop_abs
                print(
                    "STOP THRESH: ",
                    val_with_part,
                    val_without_part,
                    (val_with_part - val_without_part) / val_without_part,
                    self._thresh_stop,
                )
            else:
                should_stop = False

            path_to_best, best_path_vals_curr = self.get_path_to_return(
                agent_pos,
                best_path_curr,
                best_path_vals_curr,
                path_to_curr_loc,
                return_full_path,
                not should_stop,
                return_chosen_path,
            )

            return path_to_best, best_path_vals_curr, should_stop

        else:
            if self._enable_shape_sim:
                caption = (
                    "Map showing the path (in pink) that you'd take to follow this"
                    f" instruction: {next_instruct}"
                )
                if caption in self._cached_text_embeddings.keys():
                    text_embed_shape = self._cached_text_embeddings[caption]
                else:
                    text_embed_shape = self._vl_map._vl_model.get_text_embedding(
                        caption, head="img"
                    )
                    self._cached_text_embeddings[caption] = text_embed_shape

            # Update VL Map
            if self._vl_map.use_direction_embedding:
                masks = self._vl_map.update_direction_embeddings(
                    agent_pos, yaw, update_masks=False
                )

            best_path_next, best_path_vals_next, max_value_next, _ = (
                self.get_best_path_instruction(
                    next_instruct,
                    paths,
                    np.array([0.0, 0.0]).reshape(1, 2),
                    obstacle_map_clean,
                    text_embed_shape,
                )
            )

            switch = False

            if max_value_next > val_with_part:
                switch = True

            # We also check if current instruction's best path will not improve much,
            # in case there is a difference in the scale of the value between the
            # current and next instruction that makes it hard to switch with the above check
            else:
                if val_without_part != 0:
                    if (
                        (val_with_part - val_without_part) / val_without_part
                    ) <= self._thresh_switch:
                        switch = True

                    print(
                        "SWICTH THRESH: ",
                        val_with_part,
                        val_without_part,
                        (val_with_part - val_without_part) / val_without_part,
                        self._thresh_switch,
                    )

            if (
                switch
                and (best_path_next is not None)
                and (not (best_path_next.size == 0))
            ):
                self._points_started_instructions[next_instruct] = agent_pos

                assert masks is not None, "Returned masks is None!"
                self._vl_map.prev_masks = masks

                path_to_best, best_path_vals_next = self.get_path_to_return(
                    agent_pos,
                    best_path_next,
                    best_path_vals_next,
                    path_to_curr_loc,
                    return_full_path,
                    True,
                    return_chosen_path,
                )
                return path_to_best, best_path_vals_next, True
            else:
                if self._vl_map.use_direction_embedding:
                    # Put old embedding back
                    assert masks is not None, "Returned masks is None!"
                    self._vl_map.revert_direction_embeddings(agent_pos, yaw, masks)

                path_to_best, best_path_vals_curr = self.get_path_to_return(
                    agent_pos,
                    best_path_curr,
                    best_path_vals_curr,
                    path_to_curr_loc,
                    return_full_path,
                    True,
                    return_chosen_path,
                )
                return path_to_best, best_path_vals_curr, False
