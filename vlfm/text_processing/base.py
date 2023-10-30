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

        self._thresh_switch = options.path_thresh_switch
        self._thresh_stop = options.path_thresh_stop

        self._use_peak_threshold = options.enable_peak_threshold
        self._thresh_peak = options.path_thresh_peak

        self._prev_val_weight = options.path_prev_val_weight

        self._cur_path_val = 0.0
        self._cur_path_len = 0
        self.min_dist_goal = min_dist_goal
        # When the agent started following
        # each instruction part, used for backtracking
        self._points_started_instructions: Dict[str, np.ndarray] = {}
        self._cached_text_embeddings: Dict[str, np.ndarray] = {}

        self.ignore_locs: np.ndarray = np.array([])
        self.ignore_radius = 0.5

    def reset(self) -> None:
        self._cur_path_val = 0.0
        self._cur_path_len = 0

        self._points_started_instructions = {}
        self._cached_text_embeddings = {}

        self.ignore_locs = np.array([])

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

        # Set backtracking to zero
        if self.ignore_locs.shape[0] > 0:
            for j in range(len(path)):
                if np.any(
                    np.sqrt(
                        np.sum(
                            np.square(path[j, :].reshape(-1, 2) - self.ignore_locs),
                            axis=1,
                        )
                    )
                    < self.ignore_radius
                ):
                    image_embeddings[j, ...] = 0

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

    def generate_paths(
        self, agent_pos: np.ndarray, waypoints: np.ndarray
    ) -> List[np.ndarray]:
        # Make paths to the waypoints
        robot_radius_px = get_agent_radius_in_px(self._vl_map.pixels_per_meter)
        agent_pos_px = self._vl_map._xy_to_cvpx(agent_pos.reshape((1, 2)))[0, :]

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
        )

        if len(paths_px) == 0:
            print("No valid paths to frontiers found!")
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

            if (
                np.sqrt(np.sum(np.square(path[-1, :] - agent_pos)))
                >= self.min_dist_goal
            ):
                paths += [path]

        return paths