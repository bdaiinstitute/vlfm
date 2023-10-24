# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from vlfm.path_planning.path_planner import get_paths
from vlfm.path_planning.utils import get_agent_radius_in_px

from .obstacle_map import ObstacleMap
from .vlmap import VLMap


class VLFMap(VLMap):
    """Generates a map with image features, which can be queried with text to find the value map
    with respect to finding and navigating to the target object."""

    _thresh_switch = 0.05  # Not for last instruction part.
    # If change in value under threshold (as percentage of value up until now) then switch
    # Note we also switch when the next instruction is higher than the current
    _thresh_stop = 0.01  # Last instruction part only.
    # If change in value under threshold (as percentage of value up until now) then stop
    _prev_val_weight = (
        1.0  # Weighting for value of previous part of the path when comparing
    )
    # current to next instruction paths
    _path_positions: List[np.ndarray] = []
    # list of paths for visualization, each path is a np array of the points in that path
    _path_cols: List[Tuple[int, int, int]] = []
    # list of colors for the paths to be used in the visualization

    _col_curr: Tuple[int, int, int] = (255, 0, 0)
    _col_next: Tuple[int, int, int] = (0, 0, 255)

    def __init__(
        self,
        feats_sz: List[int] = [32, 256],
        size: int = 1000,
        use_max_confidence: bool = True,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,
        min_dist_goal: float = 0.4,
    ) -> None:
        """
        Args:
            feats_sz: The shape of the image features.
            size: The size of the value map in pixels.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV
        """
        super().__init__(feats_sz, size, use_max_confidence, fusion_type, obstacle_map)
        self._cur_path_val = 0.0
        self._cur_path_len = 0
        self.min_dist_goal = min_dist_goal
        # When the agent started following
        # each instruction part, used for backtracking
        self._cached_text_embeddings: Dict[str, np.ndarray] = {}

        self.viz_counter: int = 0

        self.ignore_locs: np.ndarray = np.array([])
        self.ignore_radius = 0.5

    def reset(self) -> None:
        super().reset()
        self._points_started_instructions = {}
        self._cached_text_embeddings = {}
        self._path_positions = []
        self._path_cols = []

        self._cur_path_val = 0.0
        self._cur_path_len = 0

        self.viz_counter = 0

        self.ignore_locs = np.array([])

    def embedding_value_within_radius(
        self,
        pixel_location: Tuple[int, int],
        radius: int,
        reduction: str = "median",
    ) -> np.ndarray:
        """Returns the maximum pixel value within a given radius of a specified pixel
        location in the given image.

        Args:
            image (np.ndarray): The input image as a 2D numpy array.
            pixel_location (Tuple[int, int]): The location of the pixel as a tuple (row,
                column).
            radius (int): The radius within which to find the maximum pixel value.
            reduction (str, optional): The method to use to reduce the values in the radius
                to a single value. Defaults to "median".

        Returns:
            np.ndarray: The reduced embedding within the radius
        """
        # Ensure that the pixel location is within the image
        assert (
            0 <= pixel_location[0] < self._vl_map.shape[0]
            and 0 <= pixel_location[1] < self._vl_map.shape[1]
        ), "Pixel location is outside the image."

        top_left_x = max(0, pixel_location[0] - radius)
        top_left_y = max(0, pixel_location[1] - radius)
        bottom_right_x = min(self._vl_map.shape[0], pixel_location[0] + radius + 1)
        bottom_right_y = min(self._vl_map.shape[1], pixel_location[1] + radius + 1)
        cropped_image = self._vl_map[
            top_left_x:bottom_right_x, top_left_y:bottom_right_y, ...
        ]

        # Draw a circular mask for the cropped image
        circle_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
        circle_mask = cv2.circle(
            circle_mask,
            (radius, radius),
            radius,
            color=255,
            thickness=-1,
        )

        circle_mask = np.where((circle_mask > 0).flatten())[0]

        overlap_values = cropped_image.reshape(-1, self._feat_channels)[circle_mask, :]
        # Filter out any values that are 0 (i.e. pixels that weren't seen yet)
        overlap_values = overlap_values[overlap_values != 0].reshape(
            -1, self._feat_channels
        )

        if overlap_values.size == 0:
            return np.zeros(cropped_image.shape[2])
        elif reduction == "mean":
            return np.mean(overlap_values, axis=0)  # type: ignore
        elif reduction == "max":
            return np.max(overlap_values, axis=0)
        elif reduction == "median":
            return np.median(overlap_values, axis=0)  # type: ignore
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")

    def get_embeddings_path(self, path: np.ndarray, radius: int = 5) -> np.ndarray:
        """Extracts the image embeddings from the map at the points in the given path"""
        px = self._xy_to_px(path)
        image_embeddings = self._vl_map[px[:, 1], px[:, 0], :]
        # image_embeddings = np.array(
        #     [
        #         self.embedding_value_within_radius(px[i, [1,0]], radius=radius)
        #         for i in range(px.shape[0])
        #     ]
        # )

        # TODO: remove 0's? Very costly to get a 0 with the averaging...

        return image_embeddings.reshape([px.shape[0]] + list(self._feats_sz))

    def get_best_path_instruction(
        self, instruction: str, paths: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns best path, goal, waypoint for local planner, value for path"""
        # Get text embeddings
        if instruction in self._cached_text_embeddings.keys():
            text_embed = self._cached_text_embeddings[instruction]
        else:
            text_embed = self._vl_model.get_text_embedding(instruction)
            self._cached_text_embeddings[instruction] = text_embed

        # Note cannot easily vectorize across paths as the paths can have different numbers of points...
        max_value = 0.0
        best_path = None
        best_path_vals = None

        for i in range(len(paths)):
            path = paths[i]
            # Get image embeddings along path
            image_embeddings = self.get_embeddings_path(path)

            # Score path
            similarity = self._vl_model.get_similarity_batch(
                image_embeddings=image_embeddings, txt_embedding=text_embed
            )

            if self.ignore_locs.shape[0] > 0:
                for j in range(len(path)):
                    if np.any(
                        np.sqrt(
                            np.sum(np.square(path[j, :] - self.ignore_locs), axis=1)
                        )
                        < self.ignore_radius
                    ):
                        print("IGNORING SCORE OF BACKTRACKING: ", path[j, :])
                        similarity[j] = 0

            # stop early if path peaks
            c_similarity = np.cumsum(similarity) / np.array(
                [i + 1 for i in range(len(similarity))]
            )
            peak_i = np.argmax(c_similarity)
            value = c_similarity[peak_i]

            print("PATH: ", path)
            print("SIMILARITY: ", similarity)
            print("C_SIMILARITY: ", c_similarity)
            print("VALUE: ", value)

            # Update
            if value > max_value:
                max_value = value
                best_path = path[: peak_i + 1].copy()
                best_path_vals = c_similarity[: peak_i + 1].copy()

        print("BEST PATH: ", best_path)
        print("BEST SIMILARITY: ", best_path_vals)
        print("BEST VALUE: ", max_value)

        return best_path, best_path_vals, max_value

    def _xy_to_cvpx(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to cv pixel coordinates.

        i.e. (x,y) with origin in top left

        Args:
            points: The array of (x, y) coordinates to convert.

        Returns:
            The array of (x, y) pixel coordinates.
        """
        px = (
            np.rint(points[:, ::-1] * self.pixels_per_meter)
            + self._episode_pixel_origin
        )
        px[:, 0] = self._map.shape[0] - px[:, 0]
        px[:, 1] = self._map.shape[1] - px[:, 1]
        return px.astype(int)

    def _cvpx_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of cv pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()
        # px_copy[:, 0] = self._map.shape[0] + px_copy[:, 0]
        # px_copy[:, 1] = self._map.shape[1] + px_copy[:, 1]
        # px_copy[:, ::-1]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return -points[:, ::-1]

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
        # Make paths to the waypoints
        robot_radius_px = get_agent_radius_in_px(self.pixels_per_meter)
        agent_pos_px = self._xy_to_cvpx(agent_pos.reshape((1, 2)))[0, :]

        # search_area = 150
        # rand_area = [max(min(agent_pos_px)-search_area,0),\
        #     min(max(agent_pos_px)+search_area,self.size)]
        # rand_area= [0, self.size-1]

        # print("AGENT LOC: ", agent_pos)
        # print("AGENT LOC PX: ", agent_pos_px)
        # print("WAYPOINTS: ", waypoints)
        # print("WAYPOINTS PX: ", self._xy_to_px(waypoints))

        # Only go in explored areas that are navigable
        # Everything else is assumed occupied
        # We increase the explorea area by a bit,
        # otherwise we wouldn't be able to go to the frontiers
        if self._obstacle_map is not None:
            explored_area = binary_dilation(
                self._obstacle_map.explored_area, iterations=10
            )
            occ_map = 1 - (self._obstacle_map._navigable_map * explored_area)

            occ_map = np.flipud(occ_map)
        else:
            raise Exception("ObstacleMap for VLFMap cannot be none when using paths!")

        # Get bounds of unoccupied area
        nz = np.nonzero(1 - occ_map)
        rand_area = [min(min(nz[0]), min(nz[1])), max(max(nz[0]), max(nz[1]))]

        # convert to (x,y) in pixel space
        waypoints_px = self._xy_to_cvpx(waypoints)

        paths_px = get_paths(
            agent_pos_px,
            waypoints_px,
            occ_map,
            rand_area,
            robot_radius_px,
            method="rrt",
        )

        if len(paths_px) == 0:
            print("No valid paths to frontiers found!")
            return None, None, False

        # print("PX: ", paths_px)

        paths = []

        for path_px in paths_px:
            # convert back to (x,y) in metric space
            path = self._cvpx_to_xy(np.rint(path_px).astype(int))

            # print("PX: ", path_px, "MET: ", path)

            if path.shape[0] > 1:
                if np.all(path[-1, :] == path[-2, :]):
                    path = path[:-1, :]

            # TODO: Check that last position in the path is further away than the pointnav radius
            if (
                np.sqrt(np.sum(np.square(path[-1, :] - agent_pos)))
                >= self.min_dist_goal
            ):
                paths += [path]

        # print("Waypoints: ", waypoints)

        # print("Paths: ", paths)

        # TODO: Radius? It is in Naoki's code but more costly for us, so ignore for now.
        # Question of if we can do it for the latents before calculating the value?

        best_path_curr, best_path_vals_curr, max_value_curr = (
            self.get_best_path_instruction(cur_instruct, paths)
        )

        len_curr = len(best_path_vals_curr)

        self._path_positions = [best_path_curr]
        self._path_cols = [self._col_curr]

        self._cur_path_val += last_path_val * last_path_len
        self._cur_path_len += last_path_len

        if last_path_len > 0:
            self.ignore_locs = np.append(
                self.ignore_locs, np.array(last_path)[:last_path_len, :]
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
            return best_path_curr, best_path_vals_curr, should_stop

        else:
            best_path_next, best_path_vals_next, max_value_next = (
                self.get_best_path_instruction(next_instruct, paths)
            )

            self._path_positions += [best_path_next]
            self._path_cols += [self._col_next]

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
                return best_path_next, best_path_vals_next, True
            else:
                return best_path_curr, best_path_vals_curr, False

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
        gt_traj: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return an image representation of the map"""
        map_img = super().visualize(markers, obstacle_map, gt_traj)

        # draw the candidate paths
        # (info about these is a property of the clas not an input)

        for i in range(len(self._path_positions)):
            map_img = self._traj_vis._draw_future_path(
                map_img, self._path_positions[i], self._path_cols[i]
            )

        cv2.imwrite(f"map_viz/conf_{self.viz_counter}.png", map_img)

        embed_nz = np.flipud(np.sum(self._vl_map != 0, axis=2))

        cv2.imwrite(f"embeddings_nonzero/{self.viz_counter}.png", embed_nz)

        self.viz_counter += 1

        return map_img
