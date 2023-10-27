# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from vlfm.path_planning.utils import get_agent_radius_in_px

from .obstacle_map import ObstacleMap
from .vlmap import VLMap


class VLFMap(VLMap):
    """Generates a map with image features, which can be queried with text to find the value map
    with respect to finding and navigating to the target object."""

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

        self.viz_counter: int = 0

    def reset(self) -> None:
        super().reset()
        self._path_positions = []
        self._path_cols = []

        self.viz_counter = 0

    def embedding_value_within_radius(
        self,
        pixel_location: Tuple[int, int],
        radius: int,
        reduction: str = "median",
    ) -> torch.tensor:
        """Returns the maximum pixel value within a given radius of a specified pixel
        location in the given image.

        Args:
            pixel_location (Tuple[int, int]): The location of the pixel as a tuple (row,
                column).
            radius (int): The radius within which to find the maximum pixel value.
            reduction (str, optional): The method to use to reduce the values in the radius
                to a single value. Defaults to "median".

        Returns:
            torch.tensor: The reduced embedding within the radius
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

        if overlap_values.shape[0] == 0:
            return torch.zeros(1, self._feat_channels, device=cropped_image.device)
        elif reduction == "mean":
            return torch.mean(overlap_values, dim=0)  # type: ignore
        elif reduction == "max":
            return torch.max(overlap_values, dim=0)[0]
        elif reduction == "median":
            return torch.median(overlap_values, dim=0)[0]  # type: ignore
        else:
            raise ValueError(f"Invalid reduction method: {reduction}")

    def get_embeddings_path(
        self, path: np.ndarray, use_radius: bool = True
    ) -> torch.tensor:
        """Extracts the image embeddings from the map at the points in the given path"""
        px = self._xy_to_px(path)
        if use_radius:
            radius = get_agent_radius_in_px(self.pixels_per_meter)

            image_embeddings = torch.zeros(
                px.shape[0], self._feat_channels, device=self.device
            )
            for i in range(px.shape[0]):
                image_embeddings[i, :] = self.embedding_value_within_radius(
                    px[i, [1, 0]], radius=radius
                )

        else:
            image_embeddings = self._vl_map[px[:, 1], px[:, 0], :]

        return image_embeddings.reshape([px.shape[0]] + list(self._feats_sz))

    def set_paths_for_viz(
        self, paths: List[np.ndarray], path_cols: List[Tuple[int, int, int]]
    ) -> None:
        self._path_positions = paths
        self._path_cols = path_cols

    # TODO: get rid of these and use the _xy_to_px
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

        # cv2.imwrite(f"map_viz/conf_{self.viz_counter}.png", map_img)

        # embed_nz = np.flipud(np.sum(self._vl_map != 0, axis=2))

        # cv2.imwrite(f"embeddings_nonzero/{self.viz_counter}.png", embed_nz)

        # self.viz_counter += 1

        return map_img
