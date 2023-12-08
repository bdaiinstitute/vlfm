# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from vlfm.mapping.vlfmap import VLFMap
from vlfm.mapping.vlmap import Stair

from .base_vln_policy import BaseVLNPolicy


class DataCollectionPolicy(BaseVLNPolicy):
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
        )

        if self._vl_map.enable_stairs:
            self.on_stairs = False
            self.point_entered_stairs = (0, 0)
            self.stair: Optional[Stair] = None

        self.save_freq = 10
        self.img_save_folder = "data_for_training"

        os.makedirs(f"{self.img_save_folder}", exist_ok=True)

        self.episode_counter = 0
        self.start_ep = True

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
        return [instruction]

    def _plan(self) -> Tuple[np.ndarray, bool, bool]:
        if not (self._num_steps % self.save_freq):
            self.save_data()

        assert (
            self.gt_path_for_viz is not None
        ), "Need to set gt_path_for_viz for collecting data!"
        if self._reached_goal:
            self._reached_goal = False
            self._cur_path_idx += 1
            if self._cur_path_idx >= self.gt_path_for_viz.shape[0]:
                return self.gt_path_for_viz[-1, :], True, False
        return self.gt_path_for_viz[self._cur_path_idx, :], False, False

    def save_data(self) -> None:
        assert self._vl_map._obstacle_map is not None

        fd_n = f"{self.img_save_folder}/{self.episode_counter:04}"

        if self.start_ep:
            os.makedirs(fd_n, exist_ok=True)
            file_log = open(f"{fd_n}/instruction.txt", "w")
            file_log.write(self._instruction)
            file_log.close()

            os.makedirs(f"{fd_n}/gt_traj_im/", exist_ok=True)
            os.makedirs(f"{fd_n}/gt_embeddings/", exist_ok=True)

        obstacle_map_clean = (
            np.ones((*self._vl_map._obstacle_map._map.shape[:2], 3), dtype=np.uint8)
            * 255
        )
        # Draw unnavigable areas in gray
        obstacle_map_clean[self._vl_map._obstacle_map._navigable_map == 0] = (
            self._vl_map._obstacle_map.radius_padding_color
        )
        # Draw obstacles in black
        obstacle_map_clean[self._vl_map._obstacle_map._map == 1] = (0, 0, 0)
        obstacle_map_clean = cv2.flip(obstacle_map_clean, 0)

        if self.start_ep and (self.gt_path_world_coord is not None):
            np.save(
                f"{fd_n}/gt_traj_world_coord.npy",
                self.gt_path_world_coord,
            )

        if self.gt_path_for_viz is not None:
            obstacle_map_gt = obstacle_map_clean
            self._vl_map._obstacle_map._traj_vis.draw_gt_trajectory(
                obstacle_map_gt, self.gt_path_for_viz
            )

            cv2.imwrite(f"{fd_n}/gt_traj_im/{self._num_steps:03}.png", obstacle_map_gt)
            if self.start_ep:
                np.save(f"{fd_n}/gt_traj_np.npy", self.gt_path_for_viz)

            embeddings_path = self._vl_map.get_embeddings_path(self.gt_path_for_viz)
            np.save(
                f"{fd_n}/gt_embeddings/{self._num_steps:03}.npy",
                torch.mean(embeddings_path, dim=0).cpu().numpy(),
            )
        self.start_ep = False

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
            self._vl_map.visualize(markers, gt_traj=self.gt_path_for_viz),
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
