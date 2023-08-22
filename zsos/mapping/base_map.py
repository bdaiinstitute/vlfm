from typing import List

import numpy as np

from zsos.mapping.traj_visualizer import TrajectoryVisualizer
from zsos.utils.geometry_utils import extract_yaw
from zsos.utils.img_utils import (
    place_img_in_img,
    rotate_image,
)


class BaseMap:
    _confidence_mask: np.ndarray = None
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = None
    _map_dtype: np.dtype = np.float32

    def __init__(
        self,
        fov: float,
        min_depth: float,
        max_depth: float,
        size: int = 1000,
        *args,
        **kwargs,
    ):
        """
        Args:
            num_channels: The number of channels in the value map.
            fov: The field of view of the camera in degrees.
            max_depth: The desired maximum depth of the camera in meters.
            use_max_confidence: Whether to use the maximum confidence value in the value
                map or a weighted average confidence value.
        """
        self.pixels_per_meter = 20

        self._fov = np.deg2rad(fov)
        self._min_depth = min_depth
        self._max_depth = max_depth

        self._map = np.zeros((size, size), dtype=self._map_dtype)
        self._episode_pixel_origin = np.array([size // 2, size // 2])
        self._traj_vis = TrajectoryVisualizer(
            self._episode_pixel_origin, self.pixels_per_meter
        )

    def reset(self):
        self._map.fill(0)
        self._camera_positions = []
        self._traj_vis = TrajectoryVisualizer(
            self._episode_pixel_origin, self.pixels_per_meter
        )

    def _localize_new_data(
        self, depth: np.ndarray, tf_camera_to_episodic: np.ndarray
    ) -> np.ndarray:
        # Get new portion of the map
        curr_data = self._process_local_data(depth, tf_camera_to_episodic)

        # Rotate this new data to match the camera's orientation
        self._last_camera_yaw = yaw = extract_yaw(tf_camera_to_episodic)
        curr_data = rotate_image(curr_data, -yaw)

        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        self._camera_positions.append(np.array([cam_x, cam_y]))

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self._episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self._episode_pixel_origin[1]

        # Overlay the new data onto the map
        curr_map = np.zeros_like(self._map)
        curr_map = place_img_in_img(curr_map, curr_data, px, py)

        return curr_map

    def _process_local_data(
        self, depth: np.ndarray, tf_camera_to_episodic: np.ndarray
    ) -> np.ndarray:
        """Processes the local data (depth image) to be used for updating the map."""
        raise NotImplementedError

    def _xy_to_px(self, points: np.ndarray) -> np.ndarray:
        """Converts an array of (x, y) coordinates to pixel coordinates.

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
        return px.astype(int)

    def _px_to_xy(self, px: np.ndarray) -> np.ndarray:
        """Converts an array of pixel coordinates to (x, y) coordinates.

        Args:
            px: The array of pixel coordinates to convert.

        Returns:
            The array of (x, y) coordinates.
        """
        px_copy = px.copy()
        px_copy[:, 0] = self._map.shape[0] - px_copy[:, 0]
        points = (px_copy - self._episode_pixel_origin) / self.pixels_per_meter
        return points[:, ::-1]
