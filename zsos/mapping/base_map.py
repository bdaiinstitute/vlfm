from typing import List

import numpy as np

from zsos.mapping.traj_visualizer import TrajectoryVisualizer


class BaseMap:
    _confidence_mask: np.ndarray = None
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = None
    _map_dtype: np.dtype = np.float32

    def __init__(self, size: int = 1000, *args, **kwargs):
        """
        Args:
            size: The size of the map in pixels.
        """
        self.pixels_per_meter = 20
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
