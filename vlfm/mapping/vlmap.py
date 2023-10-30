# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from vlfm.mapping.base_map import BaseMap
from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.utils.geometry_utils import extract_yaw
from vlfm.utils.img_utils import (
    monochannel_to_inferno_rgb,
    place_img_in_img,
    rotate_image,
)
from vlfm.vlm.blip2_unimodal import BLIP2unimodal
from vlfm.vlm.vl_model import BaseVL


class Stair:
    def __init__(
        self, lower_floor: int, higher_floor: int, image_embedding: torch.tensor
    ) -> None:
        # TODO: recovery if initial observation was wrong on up/down?
        self.lower_floor = lower_floor
        self.higher_floor = higher_floor

        self.image_embedding = image_embedding
        self.n = 0

    def add_observation(self, image_embedding: torch.tensor) -> None:
        self.image_embedding = (self.image_embedding * self.n + image_embedding) / (
            self.n + 1
        )
        self.n += 1


class VLMap(BaseMap):
    """Generates a map with image features, which can be queried with text to find the value map
    with respect to finding and navigating to the target object."""

    _confidence_masks: Dict[Tuple[float, float], np.ndarray] = {}
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = 0.0
    _min_confidence: float = 0.25
    _decision_threshold: float = 0.35
    _points_started_instructions: Dict[str, Tuple[float, float]] = {}

    def __init__(
        self,
        vl_model_type: str,
        size: int = 1000,
        use_max_confidence: bool = True,
        fusion_type: str = "default",
        obstacle_map: Optional["ObstacleMap"] = None,
        device: Optional[Any] = None,
        enable_stairs: bool = True,
    ) -> None:
        """
        Args:
            feats_sz: Which VL model to use.
            size: The size of the value map in pixels.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
            fusion_type: The type of fusion to use when combining the value map with the
                obstacle map.
            obstacle_map: An optional obstacle map to use for overriding the occluded
                areas of the FOV
        """
        super().__init__(size)

        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.vl_model_type = vl_model_type
        if vl_model_type == "BLIP2":
            feats_sz = [32, 256]
        elif vl_model_type == "CLIP":
            feats_sz = [512]
        else:
            raise Exception(f"Invalid VL model type {vl_model_type}")

        self.enable_stairs = enable_stairs

        self.device = device

        self._feat_channels = np.prod(feats_sz)
        self._vl_map = torch.zeros(
            (size, size, self._feat_channels), dtype=torch.float32, device=self.device
        )

        self._feats_sz = tuple(feats_sz)
        self._use_max_confidence = use_max_confidence
        self._fusion_type = fusion_type

        self._obstacle_map = obstacle_map
        if self._obstacle_map is not None:
            assert self._obstacle_map.pixels_per_meter == self.pixels_per_meter
            assert self._obstacle_map.size == self.size

        if os.environ.get("MAP_FUSION_TYPE", "") != "":
            self._fusion_type = os.environ["MAP_FUSION_TYPE"]

        self.vl_model = BaseVL()
        self.set_vl_model()  # seperate function to make easier to try different versions

        self.stair_text_cache: List[torch.tensor] = []
        self.upstair_text_cache: List[torch.tensor] = []

        if self.enable_stairs:
            self._stair_dict: Dict[Tuple[int, int], Stair] = {}

            # Current VL map in GPU but other floors will not fit so store in cpu
            self._map_per_level: Dict[int, torch.tensor] = {}
            self._current_floor: int = 0

    def set_vl_model(self) -> None:
        # self._vl_model = BLIP2unimodalClient(
        #     port=int(os.environ.get("VLMODEL_PORT", "12182"))
        # ) #Server not currently working properly because of the datatypes

        if self.vl_model_type == "BLIP2":
            self._vl_model = BLIP2unimodal()
        else:
            raise Exception(f"Invalid VL model type {self.vl_model_type}")

    def reset(self) -> None:
        super().reset()
        self._vl_map.fill_(0)
        if self._obstacle_map is not None:
            self._obstacle_map.reset()
        _last_camera_yaw = 0.0

        self._confidence_masks = {}
        self._min_confidence = 0.25
        self._points_started_instructions = {}

        if self.enable_stairs:
            self._stair_dict = {}
            self._map_per_level = {}
            self._current_floor = 0

    def has_stairs(
        self,
        img_embedding: torch.tensor,
        threshold: float = 0.5,  # TODO: need to check what would be good
    ) -> bool:
        labels = ["staircase", "stairs"]

        if len(self.stair_text_cache) == 0:
            self.stair_text_cache = []
            for lb in labels:
                text_embed = self._vl_model.get_text_embedding(lb)
                self.stair_text_cache += [text_embed]

        similarities = np.array(
            [
                self._vl_model.get_similarity(
                    image_embedding=img_embedding, txt_embedding=text_embed
                )
                for text_embed in self.stair_text_cache
            ]
        )

        return np.any(similarities > threshold)

    def stairs_going_up(
        self,
        img_embedding: torch.tensor,
        threshold: float = 0.5,  # TODO: need to check what would be good
    ) -> bool:
        """
        Determine is stairs are going up (return True) or not (return False)
        """
        labels = ["upstairs", "going up"]

        if len(self.upstair_text_cache) == 0:
            self.upstair_text_cache = []
            for lb in labels:
                text_embed = self._vl_model.get_text_embedding(lb)
                self.upstair_text_cache += [text_embed]

        similarities = np.array(
            [
                self._vl_model.get_similarity(
                    image_embedding=img_embedding, txt_embedding=text_embed
                )
                for text_embed in self.upstair_text_cache
            ]
        )

        return np.any(similarities > threshold)

    def change_floors(self, floor: int) -> None:
        self._map_per_level[self._current_floor] = self._vl_map.cpu().clone()
        if floor in self._map_per_level.keys():
            self._vl_map = self._map_per_level[floor].clone().to(self.device)
        else:
            self._vl_map = torch.zeros(
                (self.size, self.size, self._feat_channels),
                dtype=torch.float32,
                device=self.device,
            )
        self._current_floor = floor

    def loc_get_stairs(self, loc: Tuple[int, int]) -> Union[Stair, None]:
        # TODO: balance similarity and number of observations?
        if loc in self._stair_dict.keys():
            stair = self._stair_dict[loc]
            if stair.n < 5:  # min number of observations
                return None

            if self.has_stairs(stair.image_embedding):
                return stair

        return None

    def update_map(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> None:
        """Updates the VL map with the given image, depth image, and pose to use.

        Args:
            image: The image to use for updating the map.
            depth: The depth image to use for updating the map; expected to be already
                normalized to the range [0, 1].
            tf_camera_to_episodic: The transformation matrix from the episodic frame to
                the camera frame.
            min_depth: The minimum depth value in meters.
            max_depth: The maximum depth value in meters.
            fov: The field of view of the camera in RADIANS.
        """
        image_embedding = self._vl_model.get_image_embedding(image)

        assert image_embedding.shape == self._feats_sz, (
            "Incorrect size of image embedding "
            f"({image_embedding.shape}). Expected {self._feats_sz}."
        )

        curr_map = self._localize_new_data(
            depth, tf_camera_to_episodic, min_depth, max_depth, fov
        )

        if self.enable_stairs:
            # Stairs check
            if self.has_stairs(image_embedding):
                locs = np.nonzero(curr_map != 0)
                for i in range(len(locs[0])):
                    px = (locs[0][i], locs[1][i])
                    if px in self._stair_dict.keys():
                        self._stair_dict[px].add_observation(image_embedding)
                    else:
                        if self.stairs_going_up(image_embedding):
                            stair = Stair(
                                lower_floor=self._current_floor,
                                higher_floor=self._current_floor + 1,
                                image_embedding=image_embedding,
                            )
                        else:
                            stair = Stair(
                                lower_floor=self._current_floor - 1,
                                higher_floor=self._current_floor,
                                image_embedding=image_embedding,
                            )
                        self._stair_dict[px] = stair

        # Fuse the new data with the existing data
        self._fuse_new_data(curr_map, image_embedding.flatten())

    def visualize(
        self,
        markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None,
        obstacle_map: Optional["ObstacleMap"] = None,  # type: ignore # noqa: F821
        gt_traj: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return an image representation of the map"""
        # Visualize on top of confidence
        # (as we don't have a good way of visualizing the embeddings)

        # Must negate the y values to get the correct orientation
        reduced_map = self._map.copy()
        if obstacle_map is not None:
            reduced_map[obstacle_map.explored_area == 0] = 0
        map_img = np.flipud(reduced_map)
        # Make all 0s in the confidence map equal to the max value, so they don't throw off
        # the color mapping (will revert later)
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        # Revert all values that were originally zero to white
        map_img[zero_mask] = (255, 255, 255)
        if len(self._camera_positions) > 0:
            self._traj_vis.draw_trajectory(
                map_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

        if gt_traj is not None:
            self._traj_vis.draw_gt_trajectory(map_img, gt_traj)

        if markers is not None:
            for pos, marker_kwargs in markers:
                map_img = self._traj_vis.draw_circle(map_img, pos, **marker_kwargs)

        return map_img

    def _process_local_data(
        self, depth: np.ndarray, fov: float, min_depth: float, max_depth: float
    ) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.

        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.
        Returns:
            A mask of the visible portion of the FOV.
        """
        # Squeeze out the channel dimension if depth is a 3D array
        if len(depth.shape) == 3:
            depth = depth.squeeze(2)
        # Squash depth image into one row with the max depth value for each column
        depth_row = np.max(depth, axis=0) * (max_depth - min_depth) + min_depth

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        angles = np.linspace(-fov / 2, fov / 2, len(depth_row))

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row
        y = depth_row * np.tan(angles)

        # Get blank cone mask
        cone_mask = self._get_confidence_mask(fov, max_depth)

        # Convert the x, y coordinates to pixel coordinates
        x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

        # Create a contour from the x, y coordinates, with the top left and right
        # corners of the image as the first two points
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)

        # Draw the contour onto the cone mask, in filled-in black
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)  # type: ignore

        return visible_mask

    def _localize_new_data(
        self,
        depth: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fov: float,
    ) -> np.ndarray:
        # Get new portion of the map
        curr_data = self._process_local_data(depth, fov, min_depth, max_depth)

        # Rotate this new data to match the camera's orientation
        yaw = extract_yaw(tf_camera_to_episodic)
        curr_data = rotate_image(curr_data, -yaw)

        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # Overlay the new data onto the map
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)

        return curr_map

    def _get_blank_cone_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        size = int(max_depth * self.pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(  # type: ignore
            cone_mask,
            (size, size),  # center_pixel
            (size, size),  # axes lengths
            0,  # angle circle is rotated
            -np.rad2deg(fov) / 2 + 90,  # start_angle
            np.rad2deg(fov) / 2 + 90,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask

    def _get_confidence_mask(self, fov: float, max_depth: float) -> np.ndarray:
        """Generate a FOV cone with central values weighted more heavily"""
        if (fov, max_depth) in self._confidence_masks:
            return self._confidence_masks[(fov, max_depth)].copy()
        cone_mask = self._get_blank_cone_mask(fov, max_depth)
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self._min_confidence, 1)
                adjusted_mask[row, col] = confidence
        adjusted_mask = adjusted_mask * cone_mask
        self._confidence_masks[(fov, max_depth)] = adjusted_mask.copy()

        return adjusted_mask

    def _fuse_new_data(self, new_map: np.ndarray, image_embedding: np.ndarray) -> None:
        """Fuse the new data with the existing VL and confidence maps.

        Args:
            new_map: The new new_map map data to fuse. Confidences are between
                0 and 1, with 1 being the most confident.
            image_embedding: The image_embedding attributed to the new portion
                of the map.
        """
        assert len(image_embedding) == self._feat_channels, (
            "Incorrect length of image embedding in fuse data (should be flat)"
            f"({len(image_embedding)}). Expected {self._feat_channels}."
        )

        if self._obstacle_map is not None:
            # If an obstacle map is provided, we will use it to mask out the
            # new map
            explored_area = self._obstacle_map.explored_area
            new_map[explored_area == 0] = 0
            self._map[explored_area == 0] = 0
            self._vl_map[explored_area == 0] = 0

        if self._fusion_type == "replace":
            # Ablation. The values from the current observation will overwrite any
            # existing values
            print("VALUE MAP ABLATION:", self._fusion_type)
            new_vl_map = np.zeros_like(self._vl_map)
            new_vl_map[new_map > 0] = image_embedding
            self._map[new_map > 0] = new_map[new_map > 0]
            self._vl_map[new_map > 0] = new_vl_map[new_map > 0]
            return
        elif self._fusion_type == "equal_weighting":
            # Ablation. Updated values will always be the mean of the current and
            # new values, meaning that confidence scores are forced to be the same.
            print("VALUE MAP ABLATION:", self._fusion_type)
            self._map[self._map > 0] = 1
            new_map[new_map > 0] = 1
        else:
            assert (
                self._fusion_type == "default"
            ), f"Unknown fusion type {self._fusion_type}"

        # Any values in the given map that are less confident than
        # self._decision_threshold AND less than the new_map in the existing map
        # will be silenced into 0s
        new_map_mask = np.logical_and(
            new_map < self._decision_threshold, new_map < self._map
        )
        new_map[new_map_mask] = 0

        if self._use_max_confidence:
            # For every pixel that has a higher new_map in the new map than the
            # existing value map, replace the value in the existing value map with
            # the new value
            higher_new_map_mask = new_map > self._map
            self._vl_map[higher_new_map_mask] = image_embedding
            # Update the new_map map with the new new_map values
            self._map[higher_new_map_mask] = new_map[higher_new_map_mask]
        else:
            # Each pixel in the existing value map will be updated with a weighted
            # average of the existing value and the new value. The weight of each value
            # is determined by the current and new new_map values. The new_map map
            # will also be updated with using a weighted average in a similar manner.
            confidence_denominator = self._map + new_map
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                weight_1 = self._map / confidence_denominator
                weight_2 = new_map / confidence_denominator

            weight_1_channeled = np.repeat(
                np.expand_dims(weight_1, axis=2), self._feat_channels, axis=2
            )
            weight_2_channeled = np.repeat(
                np.expand_dims(weight_2, axis=2), self._feat_channels, axis=2
            )

            self._vl_map = (
                self._vl_map * weight_1_channeled + image_embedding * weight_2_channeled
            )
            self._map = self._map * weight_1 + new_map * weight_2

            # Because confidence_denominator can have 0 values, any nans in either the
            # value or confidence maps will be replaced with 0
            self._vl_map = np.nan_to_num(self._vl_map)
            self._map = np.nan_to_num(self._map)


def remap(
    value: float, from_low: float, from_high: float, to_low: float, to_high: float
) -> float:
    """Maps a value from one range to another.

    Args:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low