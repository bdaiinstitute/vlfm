# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from argparse import Namespace
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_dilation

from vlfm.mapping.vlfmap import VLFMap
from vlfm.path_planning.path_planner import get_paths
from vlfm.path_planning.utils import get_agent_radius_in_px


class VLPathSelector:
    def __init__(self, options: Namespace, vl_map: VLFMap, min_dist_goal: float = 0.4):
        self.args = options
        self._vl_map = vl_map

        self._thresh_switch = options.similarity_calc.path_thresh_switch
        self._thresh_stop = options.similarity_calc.path_thresh_stop

        self._path_thresh_stop_abs = options.similarity_calc.path_thresh_stop_abs

        self._use_peak_threshold = options.similarity_calc.enable_peak_threshold
        self._thresh_peak = options.similarity_calc.path_thresh_peak

        self._add_directional_waypoints = (
            options.similarity_calc.enable_directional_waypoints
        )

        self._cur_path_val = 0.0
        self._cur_path_len = 0
        self.min_dist_goal = min_dist_goal
        # When the agent started following
        # each instruction part, used for backtracking
        self._points_started_instructions: Dict[str, np.ndarray] = {}
        self._cached_text_embeddings: Dict[str, np.ndarray] = {}

        self.prev_path_value = 1.0

        self._limit_waypoint_radius = options.similarity_calc.limit_waypoint_radius
        self._waypoints_radius_limit = options.similarity_calc.waypoints_radius_limit

    def reset(self) -> None:
        self._cur_path_val = 0.0
        self._cur_path_len = 0

        self._points_started_instructions = {}
        self._cached_text_embeddings = {}

        self.prev_path_value = 1.0

    def similarity_main_loop(
        self,
        image_embeddings: torch.tensor,
        text_embed: torch.tensor,
        denom: np.ndarray,
        thresh: float = -1.0,
    ) -> Tuple[float, np.ndarray, int]:
        similarity = self._vl_map._vl_model.get_similarity_batch(
            image_embeddings=image_embeddings, txt_embedding=text_embed
        )

        c_similarity = np.cumsum(similarity) / denom
        peak_i = np.argmax(c_similarity)
        value = c_similarity[peak_i]

        if self._use_peak_threshold:
            if thresh == -1.0:
                thresh = self._thresh_peak
            if c_similarity.size > 0:
                # Get first idx where it is over the threshold
                where_over = c_similarity > value * thresh
                if np.any(where_over):
                    peak_i = np.where(where_over)[0][0]
                    value = c_similarity[peak_i]

        return value, c_similarity, peak_i

    def get_similarity(
        self,
        path: np.ndarray,
        image_embeddings: torch.tensor,
        text_embed: torch.tensor,
        method: str = "weighted average embeddings",
        thresh: float = -1.0,
    ) -> Tuple[float, np.ndarray, int]:
        # Get original indices of zeros
        orig_nz = [
            torch.any(image_embeddings[i, ...] != 0)
            for i in range(image_embeddings.shape[0])
        ]

        # Denom that ignores the original zeros
        denom_l = []
        denom_i = 1
        for j in range(image_embeddings.shape[0]):
            if orig_nz[j]:
                denom_i += 1
            denom_l += [denom_i]
        denom = np.array(denom_l)

        if thresh == -1.0:
            thresh = self._thresh_peak

        if method == "average sim":
            return self.similarity_main_loop(
                image_embeddings, text_embed, denom, thresh=thresh
            )
        elif method == "average embeddings":
            image_embeddings = torch.cumsum(image_embeddings, dim=0) / denom
            return self.similarity_main_loop(
                image_embeddings, text_embed, np.ones(denom.shape), thresh=thresh
            )
        elif method == "weighted average embeddings":
            tau = 0.1  # CLIP-Hitchhiker found optimal between 0.05-0.15

            similarity = self._vl_map._vl_model.get_similarity_batch(
                image_embeddings=image_embeddings, txt_embedding=text_embed
            )

            exp_sim = np.exp(similarity / tau)
            exp_sim[not orig_nz] = (
                0  # Set similarity to 0 for ones with no embedding to not mess up denom2
            )
            denom2 = np.cumsum(exp_sim)
            denom2[denom2 == 0] = 1  # Prevent divide by 0
            w = torch.tensor(exp_sim / denom2, device=image_embeddings.device)

            rep = list(image_embeddings.shape)[1:] + [1]

            image_embeddings = image_embeddings * torch.movedim(w.repeat(rep), -1, 0)

            image_embeddings = torch.cumsum(image_embeddings, dim=0)

            return self.similarity_main_loop(
                image_embeddings, text_embed, np.ones(denom.shape), thresh=thresh
            )

        else:
            raise Exception(f"Invalid method {method} for get_similarity")

    def get_path_value_main_loop(
        self, path: np.ndarray, instruction: str
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_best_path_instruction(
        self,
        instruction: str,
        paths: List[np.ndarray],
        path_to_curr_loc: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Returns best path, goal, waypoint for local planner, value for path"""
        # Get value for previous part
        value_prev_path, _, _ = self.get_path_value_main_loop(
            path_to_curr_loc, instruction
        )

        # Note cannot easily vectorize across paths as the paths can have different numbers of points...
        max_value = 0.0
        best_path = None
        best_path_vals = None

        for i in range(len(paths)):
            path = paths[i]

            full_path = np.append(path_to_curr_loc, path, axis=0)

            value, total_value, peak_i = self.get_path_value_main_loop(
                full_path, instruction
            )

            # Update
            if value > max_value:
                max_value = value
                best_path = path[: peak_i + 1, :]
                best_path_vals = total_value[: peak_i + 1]

        return best_path, best_path_vals, max_value, value_prev_path

    def get_path_to_return(
        self,
        agent_pos: np.ndarray,
        best_path_curr: np.ndarray,
        best_path_vals_curr: np.ndarray,
        path_to_curr_loc: np.ndarray,
        return_full_path: bool,
        vis_path: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if return_full_path:
            path_to_best_list = self.generate_paths(
                agent_pos, best_path_curr[-1, :].reshape(1, 2), one_path=True
            )
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
            if vis_path:
                self._vl_map.set_paths_for_viz(
                    [best_path_curr, path_to_best], [(255, 0, 0), (0, 0, 255)]
                )
        else:
            path_to_best = best_path_curr[-1, :].reshape(1, 2)
            best_path_vals_curr = best_path_vals_curr[-1]
            if vis_path:
                self._vl_map.set_paths_for_viz([best_path_curr], [(255, 0, 0)])

        return path_to_best, best_path_vals_curr

    def generate_paths(
        self, agent_pos: np.ndarray, waypoints: np.ndarray, one_path: bool = False
    ) -> List[np.ndarray]:
        # Only use waypoints that are inside the radius limit
        if self._limit_waypoint_radius and (not one_path):
            dists = np.sqrt(
                np.sum(np.square(waypoints - agent_pos.reshape(1, 2)), axis=1)
            )
            print("N waypoints before radius enforcement: ", waypoints.shape[0])
            waypoints = waypoints[dists <= self._waypoints_radius_limit]
            print("N waypoints after radius enforcement: ", waypoints.shape[0])

        # Make paths to the waypoints
        robot_radius_px = get_agent_radius_in_px(self._vl_map.pixels_per_meter)
        agent_pos_px = self._vl_map._xy_to_cvpx(agent_pos.reshape(1, 2))[0, :]

        # Only go in explored areas that are navigable
        # Everything else is assumed occupied
        # We increase the explorea area by a bit,
        # otherwise we wouldn't be able to go to the frontiers
        if self._vl_map._obstacle_map is not None:
            explored_area = binary_dilation(
                self._vl_map._obstacle_map.explored_area, iterations=10
            )
            occ_map = 1 - (self._vl_map._obstacle_map._navigable_map * explored_area)

            occ_map = np.flipud(occ_map)
        else:
            raise Exception("ObstacleMap for VLFMap cannot be none when using paths!")

        # Get bounds of unoccupied area
        nz = np.nonzero(1 - occ_map)
        rand_area = [min(min(nz[0]), min(nz[1])), max(max(nz[0]), max(nz[1]))]

        # convert to (x,y) in pixel space
        waypoints_px = self._vl_map._xy_to_cvpx(waypoints)

        paths_px = get_paths(
            agent_pos_px,
            waypoints_px,
            occ_map,
            rand_area,
            robot_radius_px,
            method="both",
            one_path=one_path,
        )

        if len(paths_px) == 0:
            print(f"No valid paths found! One_path: {one_path}")
            return []

        # print("PX: ", paths_px)

        paths = []

        for path_px in paths_px:
            # convert back to (x,y) in metric space
            path = self._vl_map._cvpx_to_xy(np.rint(path_px).astype(int))

            # print("PX: ", path_px, "MET: ", path)

            if path.shape[0] > 1:
                if np.all(path[-1, :] == path[-2, :]):
                    path = path[:-1, :]

            if one_path or (
                np.sqrt(np.sum(np.square(path[-1, :] - agent_pos)))
                >= self.min_dist_goal
            ):
                paths += [path]

        return paths

    def get_directional_waypoints(
        self, agent_pos: np.ndarray, agent_yaw: float
    ) -> np.ndarray:
        # Get point above agent
        up_p = agent_pos.copy()
        up_p[1] -= self.args.similarity_calc.direction_waypoints_dist

        waypoints = []
        ax = agent_pos[0]
        ay = agent_pos[1]
        # Rotate into appropriate yaw:
        for i in range(self.args.similarity_calc.direction_waypoints_n):
            yaw = (
                agent_yaw
                + np.pi * 2 * i / self.args.similarity_calc.direction_waypoints_n
            )
            s = np.sin(yaw)
            c = np.cos(yaw)

            pt = np.array(
                [
                    ax + c * (up_p[0] - ax) - s * (up_p[1] - ay),
                    ay + s * (up_p[0] - ax) + c * (up_p[1] - ay),
                ]
            )

            # Check if in bounds
            pt_px = self._vl_map._xy_to_px(pt.reshape(1, 2))[0]
            if (
                0 <= pt_px[0]
                and pt_px[0] < self._vl_map.size
                and 0 <= pt_px[1]
                and pt_px[1] < self._vl_map.size
            ):
                # Check that not on obstacle
                if not self._vl_map.is_on_obstacle(pt):
                    waypoints += [pt]

        return np.array(waypoints)
