# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from vlfm.mapping.vlfmap import VLFMap
from vlfm.mapping.vlmap import Stair
from vlfm.text_processing.multiresolution import VLPathSelectorMR

from .base_vln_policy import BaseVLNPolicy


class TestingPolicy(BaseVLNPolicy):
    _target_object_color: Tuple[int, int, int] = (0, 255, 0)
    _selected_frontier_color: Tuple[int, int, int] = (0, 255, 255)
    _frontier_color: Tuple[int, int, int] = (0, 0, 255)
    _circle_marker_thickness: int = 2
    _circle_marker_radius: int = 5
    _last_value: float = float("-inf")
    _last_frontier: np.ndarray = np.zeros(2)

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._cur_path_idx = 0
        self._last_plan_step = 0

        self._vl_map: VLFMap = VLFMap(
            vl_model_type=self.args.map.vl_feature_type,
            size=self.args.map.map_size,
            obstacle_map=self._obstacle_map,
            enable_stairs=self.args.map.enable_stairs,
            use_adapter=self.args.map.use_adapter,
        )

        if self._vl_map.enable_stairs:
            self.on_stairs = False
            self.point_entered_stairs = (0, 0)
            self.stair: Optional[Stair] = None

        self.episode_counter = 0
        self.start_ep = True

        self._path: np.ndarray = np.array([])
        self._text_embed_cache: Optional[List[torch.tensor]] = None

        self._path_selector: VLPathSelectorMR = VLPathSelectorMR(
            self.args, self._vl_map, min_dist_goal=self._pointnav_stop_radius
        )

    def _reset(self) -> None:
        super()._reset()
        self._cur_path_idx = 0
        self._last_plan_step = 0

        self._vl_map.reset()

        if self._vl_map.enable_stairs:
            self.on_stairs = False
            self.point_entered_stairs = (0, 0)
            self.stair = None

        self.episode_counter += 1
        self.start_ep = True

        self._path = np.array([])
        self._text_embed_cache = None

        self._path_selector.reset()

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self._pre_step(observations, masks)
        self._update_vl_map()
        return super().act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            deterministic,
        )

    def _parse_instruction(self, instruction: str) -> List[str]:
        words = instruction.split(" ")
        parsed_instruct = [" ".join(words[: i + 1]) for i in range(len(words))]

        print("PARSING: ", instruction)
        print("OUPUT: ", parsed_instruct)
        return parsed_instruct

    def _plan(self) -> Tuple[np.ndarray, bool, bool]:
        assert (
            self.gt_path_for_viz is not None
        ), "Need to set gt_path_for_viz for collecting data!"

        if self._text_embed_cache is None:
            # start_str = "Let me give you a tour: "
            self._text_embed_cache = []
            for i in range(len(self._instruction_parts)):
                # text_embed = self._vl_map._vl_model.get_text_embedding
                # (start_str + '"' + self._instruction_parts[i] + '"')
                text_embed = self._vl_map._vl_model.get_text_embedding(
                    self._instruction_parts[i], head="embed"
                )
                self._text_embed_cache += [text_embed]

        self._path = np.append(
            self._path.reshape(-1, 2),
            self._observations_cache["robot_xy"].reshape(1, 2),
            axis=0,
        )

        if not ((self._num_steps - 24) % 10):
            # show all proposed paths
            frontiers = self._observations_cache["frontier_sensor"]
            if not (np.array_equal(frontiers, np.zeros((1, 2))) or len(frontiers) == 0):
                yaw = self._observations_cache["robot_heading"]
                robot_xy = self._observations_cache["robot_xy"]

                chosen_path, _, _ = self._path_selector.get_goal_for_instruction(
                    robot_xy,
                    yaw,
                    frontiers,
                    self._instruction,
                    False,
                    return_full_path=self.args.use_path_waypoints,
                )

                # waypoints = frontiers

                # if self._path_selector._store_points_on_paths and (self._path_selector._extra_waypoints.size) > 0:
                #     bad_idx = []
                #     for i in range(self._path_selector._extra_waypoints.shape[0]):
                #         pt = self._path_selector._extra_waypoints[i, :]
                #         if self._vl_map.is_on_obstacle(pt):
                #             bad_idx += [i]
                #     if len(bad_idx) > 0:
                #         print("Deleting cached waypoints on obstacles!")
                #         self._path_selector._extra_waypoints = np.delete(
                #             self._path_selector._extra_waypoints, bad_idx, axis=0
                #         )

                #     waypoints = np.append(waypoints, self._path_selector._extra_waypoints, axis=0)

                # # Add waypoints
                # if self._path_selector._add_directional_waypoints:
                #     dir_waypoints = self._path_selector.get_directional_waypoints(agent_pos, agent_yaw)
                #     waypoints = np.append(waypoints, dir_waypoints, axis=0)

                # prop_paths = self._path_selector.generate_paths(np.array([0.0, 0.0]), waypoints)

                # self._vl_map._path_positions = []
                # self._vl_map._path_cols = []
                # for path in prop_paths:
                #     self._vl_map._path_positions += [path]
                #     self._vl_map._path_cols += [(255, 0, 0)]

                self._vl_map._path_positions += [chosen_path]
                self._vl_map._path_cols += [(0, 0, 255)]

        # Get part of instruction that best matches the GT path
        best_val = 0.0

        for i in range(len(self._instruction_parts)):
            embeddings_along_path = self._vl_map.get_embeddings_path(self._path)
            text_embed = self._text_embed_cache[i]
            val = np.mean(
                self._vl_map._vl_model.get_similarity_batch(
                    embeddings_along_path, text_embed
                ),
                axis=0,
            )

            if val >= best_val:
                best_val = val
                self._curr_instruction_idx = i

        if not ((self._num_steps - 24) % 10):
            ###Evaluate value of chosen_path compared to GT path
            embeddings_along_path = self._vl_map.get_embeddings_path(chosen_path)
            val_chosen = np.mean(
                self._vl_map._vl_model.get_similarity_batch(
                    embeddings_along_path, text_embed
                ),
                axis=0,
            )
            print(f"GT path val: {val} ({best_val}), Chosen path val: {val_chosen}")

        if self._reached_goal:
            self._reached_goal = False
            self._cur_path_idx += 1
            if self._cur_path_idx >= self.gt_path_for_viz.shape[0]:
                return self.gt_path_for_viz[-1, :], True, False
        return self.gt_path_for_viz[self._cur_path_idx, :], False, False

    def _update_vl_map(self) -> None:
        for rgb, depth, tf, min_depth, max_depth, fov in self._observations_cache[
            "vl_map_rgbd"
        ]:
            self._vl_map.update_map(rgb, depth, tf, min_depth, max_depth, fov)

        self._vl_map.update_agent_traj(
            self._observations_cache["robot_xy"],
            self._observations_cache["robot_heading"],
        )

    def _get_policy_info(self) -> Dict[str, Any]:
        policy_info = super()._get_policy_info()

        if not self._visualize:
            return policy_info

        markers = []

        # Draw frontiers on to the cost map
        frontiers = self._observations_cache["frontier_sensor"]
        for frontier in frontiers:
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": self._frontier_color,
            }
            markers.append((frontier[:2], marker_kwargs))

        if not np.array_equal(self._last_goal, np.zeros(2)):
            # Draw the pointnav goal on to the cost map
            if any(np.array_equal(self._last_goal, frontier) for frontier in frontiers):
                color = self._selected_frontier_color
            else:
                color = self._target_object_color
            marker_kwargs = {
                "radius": self._circle_marker_radius,
                "thickness": self._circle_marker_thickness,
                "color": color,
            }
            markers.append((self._last_goal, marker_kwargs))

        policy_info["vl_map"] = cv2.cvtColor(
            self._vl_map.visualize(
                markers, gt_traj=self.gt_path_for_viz, instruction=self._instruction
            ),
            cv2.COLOR_BGR2RGB,
        )

        # policy_info["render_below_images"] += ["current instruction part"]
        policy_info["current instruction part"] = self._instruction_parts[
            self._curr_instruction_idx
        ]

        if self._vl_map.enable_stairs:
            policy_info["render_below_images"] += ["on stairs"]
            if self.on_stairs:
                assert isinstance(self.stair, Stair)
                policy_info["on stairs"] = (
                    f"On stairs from {self.stair.lower_floor} to"
                    f" {self.stair.higher_floor}, current floor"
                    f" {self._vl_map._current_floor}"
                )
            else:
                policy_info["on stairs"] = "Not on stairs"

        return policy_info
