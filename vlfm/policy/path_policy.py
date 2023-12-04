# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch import Tensor

from vlfm.mapping.vlfmap import VLFMap
from vlfm.mapping.vlmap import Stair

from .base_vln_policy import BaseVLNPolicy


class BasePathPolicy(BaseVLNPolicy):
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
        self._path_to_follow: List[List[float]] = []
        self._path_vals: List[float] = []
        self._cur_path_idx = 0
        self._last_plan_step = 0

        self._vl_map: VLFMap = VLFMap(
            vl_model_type=self.args.map.vl_feature_type,
            size=self.args.map.map_size,
            obstacle_map=self._obstacle_map,
            enable_stairs=self.args.map.enable_stairs,
            use_adapter=self.args.map.use_adapter,
            use_max_confidence=True,  # False
        )

        if self._vl_map.enable_stairs:
            self.on_stairs = False
            self.point_entered_stairs = (0, 0)
            self.stair: Optional[Stair] = None

        if self.args.replanning.enable_replan_when_stuck:
            self.last_xy = np.array([0, 0])
            self.n_at_xy = 0

        self.n_steps_goal = 0
        self.times_no_paths = 0
        self.why_stop = "no stop"

        self._replan_interval = self.args.replanning.replan_interval
        self._force_dont_stop_until = self.args.replanning.force_dont_stop_until
        self.force_dont_stop_after_stuck = (
            self.args.replanning.force_dont_stop_after_stuck
        )

        self._look_at_frontiers = self.args.replanning.look_at_frontiers
        self._frontier_dist_thresh = self.args.replanning.frontier_dist_thresh
        self.frontiers_at_plan = np.array([])

        self._look_at_goal = self.args.replanning.look_at_goal

    def _reset(self) -> None:
        super()._reset()
        self._path_to_follow = []
        self._path_vals = []
        self._cur_path_idx = 0
        self._last_plan_step = 0

        self._vl_map.reset()

        if self._vl_map.enable_stairs:
            self.on_stairs = False
            self.point_entered_stairs = (0, 0)
            self.stair = None

        if self.args.replanning.enable_replan_when_stuck:
            self.last_xy = np.array([0, 0])
            self.n_at_xy = 0

        self.n_steps_goal = 0
        self.times_no_paths = 0
        self.why_stop = "no stop"

        self.frontiers_at_plan = np.array([])

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
        raise NotImplementedError

    def _stair_preplan_step(self) -> None:
        xy = self._observations_cache["robot_xy"]
        px = self._vl_map._xy_to_px(xy.reshape(1, 2)).reshape(2)
        stair = self._vl_map.loc_get_stairs((px[0], px[1]))

        if stair is None:
            if self.on_stairs:
                # check if we've gone far enough that we think we're gone up/down the stairs
                # TODO: change this check to have a visual component?
                # Then could also fix the floor logic
                if (
                    np.sqrt(
                        float((px[0] - self.point_entered_stairs[0]) ** 2)
                        + float((px[1] - self.point_entered_stairs[1]) ** 2)
                    )
                    / self.pixels_per_meter
                    > 2.0
                ):
                    assert isinstance(self.stair, Stair)
                    if self._vl_map._current_floor == self.stair.lower_floor:
                        self._vl_map.change_floors(self.stair.higher_floor)
                    elif self._vl_map._current_floor == self.stair.higher_floor:
                        self._vl_map.change_floors(self.stair.lower_floor)
                    else:
                        raise Exception(
                            "Something is wrong with the floor logic. Tried to go"
                            + f" from {self._vl_map._current_floor} on stair from"
                            + f" {self.stair.lower_floor} to {self.stair.higher_floor}"
                        )
                self.on_stairs = False
                self.point_entered_stairs = (0, 0)
                self.stair = None

        else:
            if not self.on_stairs:
                self.on_stairs = True
                self.point_entered_stairs = px
                self.stair = stair

    def _pre_plan_logic(self) -> Tuple[bool, bool, int]:
        replan = False
        force_dont_stop = False

        xy = self._observations_cache["robot_xy"]

        if self.args.use_path_waypoints:
            if self._reached_goal:
                self._cur_path_idx += 1
                # self._reached_goal = False
                self.n_steps_goal = 0
            else:
                self.n_steps_goal += 1

            if self.n_steps_goal > 20:
                print("CAN'T GET TO CURRENT SUBGOAL, ADVANCING")
                if self.force_dont_stop_after_stuck:
                    force_dont_stop = True
                self._cur_path_idx += 1
                self.n_steps_goal = 0
            # Check if actually close to position we will give as input
            idx_path = self._cur_path_idx
            if len(self._path_to_follow) > self._cur_path_idx:
                path_pos = self._path_to_follow[self._cur_path_idx]
                if np.sqrt(np.sum(np.square(path_pos - xy))) > 1.0:
                    # if we are close to any point on the path assume we followed up to then
                    dists = np.sqrt(
                        np.sum(np.square(self._path_to_follow - xy), axis=1)
                    )
                    # print("TOO FAR FROM LAST POS, DISTS: ", dists)
                    if np.any(dists < 1.0):
                        idx_path = np.argmin(dists)
                    # otherwise set to -1 and we will check for this
                    else:
                        idx_path = -1

            if len(self._path_to_follow) < self._cur_path_idx + 1:
                print("REPLAN because ran out of waypoints")
                replan = True

        else:
            idx_path = -1
            if len(self._path_to_follow) == 0:
                print("REPLAN because no path to follow")
                replan = True
            if self._reached_goal:
                print("REPLAN because reached goal")
                replan = True
                # self._reached_goal = False

        if self.args.replanning.enable_replan_at_steps:
            if self._num_steps > (self._last_plan_step + self._replan_interval):
                print("REPLAN at steps")
                replan = True

        # Work out if end goal is on an obstacle, if so replan:
        if len(self._path_to_follow) > 0:
            if self._vl_map.is_on_obstacle(
                np.array(self._path_to_follow[len(self._path_to_follow) - 1])
            ):
                if self.force_dont_stop_after_stuck:
                    force_dont_stop = True
                print("REPLAN because goal is on obstacle")
                replan = True

        if self.args.replanning.enable_replan_when_stuck:
            if np.sqrt(np.sum(np.square(self.last_xy - xy))) < 0.05:
                self.n_at_xy += 1
            else:
                self.n_at_xy = 0
            if self.n_at_xy > 10:  # might stay here to turn
                print("IS STUCK! Replanning")
                replan = True
                self.n_at_xy = 0
                if self.force_dont_stop_after_stuck:
                    force_dont_stop = True

            self.last_xy = xy

        if self._look_at_frontiers and replan:
            self.frontiers_at_plan = self._observations_cache["frontier_sensor"]
            if self.should_turn and not (self._reached_goal):
                self.should_turn = False
                force_dont_stop = True

        self._reached_goal = False

        return replan, force_dont_stop, idx_path

    def _plan(self) -> Tuple[np.ndarray, bool, bool]:
        raise NotImplementedError

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
                markers,
                gt_traj=self.gt_path_for_viz,
                instruction=self._instruction_parts[self._curr_instruction_idx],
            ),
            cv2.COLOR_BGR2RGB,
        )

        # cv2.imwrite(f"map_viz/valuemap_{self._num_steps}.png", policy_info["vl_map"])

        policy_info["render_below_images"] += ["current instruction part"]
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
