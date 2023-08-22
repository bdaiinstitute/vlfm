import json
from typing import Tuple

import cv2
import numpy as np

from zsos.mapping.base_map import BaseMap
from zsos.mapping.value_map import JSON_PATH, KWARGS_JSON
from zsos.utils.geometry_utils import get_point_cloud, transform_points


class ObstacleMap(BaseMap):
    """Generates two maps; one representing the area that the robot has explored so far,
    and another representing the obstacles that the robot has seen so far.
    """

    _map_dtype: np.dtype = bool
    _image_width: int = None  # set upon execution of update_map method

    def __init__(
        self,
        fov: float,
        min_depth: float,
        max_depth: float,
        min_height: float,
        max_height: float,
        agent_radius: float,
        size: int = 1000,
    ):
        super().__init__(fov, min_depth, max_depth, size)
        self._obstacle_map = np.zeros((size, size), dtype=bool)
        self._navigable_map = np.zeros((size, size), dtype=bool)
        self._hfov = np.deg2rad(fov)
        self._min_height = min_height
        self._max_height = max_height
        self.__fx = None
        kernel_size = self.pixels_per_meter * agent_radius
        # round kernel_size to nearest odd number
        kernel_size = int(kernel_size) + (int(kernel_size) % 2 == 0)
        self._navigable_kernel = np.ones((kernel_size, kernel_size), np.uint8)

    @property
    def _fx(self) -> float:
        if self.__fx is None:
            self.__fx = self._image_width / (2 * np.tan(self._hfov / 2))
        return self.__fx

    @property
    def _fy(self) -> float:
        return self._fx

    def update_map(self, depth: np.ndarray, tf_camera_to_episodic: np.ndarray):
        """Adds all obstacles from the current view to the map. Also updates the area
        that the robot has explored so far.
        """
        if self._image_width is None:
            self._image_width = depth.shape[1]
        point_cloud_camera_frame = self._get_local_point_cloud(depth)
        point_cloud_episodic_frame = transform_points(
            tf_camera_to_episodic, point_cloud_camera_frame
        )
        obstacle_cloud = filter_points_by_height(
            point_cloud_episodic_frame, self._min_height, self._max_height
        )
        cloud_to_grid(
            obstacle_cloud,
            self._obstacle_map,
            self._episode_pixel_origin,
            self.pixels_per_meter,
        )

        self._navigable_map = 1 - cv2.dilate(
            self._obstacle_map.astype(np.uint8),
            self._navigable_kernel,
            iterations=1,
        ).astype(bool)

    def get_frontiers(self):
        """Returns the frontiers of the map."""
        raise NotImplementedError

    def visualize(self):
        """Visualizes the map."""
        vis_img = np.ones((*self._obstacle_map.shape[:2], 3), dtype=np.uint8) * 255
        vis_obstacle_map = np.flipud(self._obstacle_map)
        vis_navigable_map = np.flipud(self._navigable_map)
        # Draw unnavigable areas in gray
        vis_img[vis_navigable_map == 0] = (100, 100, 100)
        # Draw obstacles in black
        vis_img[vis_obstacle_map == 1] = (0, 0, 0)
        return vis_img

    def _process_local_data(
        self, depth: np.ndarray, tf_camera_to_episodic: np.ndarray = None
    ) -> np.ndarray:
        """Using the FOV and depth, return the 2D top down map of obstacles within the
        FOV.

        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.
            tf_camera_to_episodic: Currently unused for this subclass.
        Returns:
            A mask of the visible portion of the FOV.
        """

    def _fuse_new_data(self, curr_map: np.ndarray):
        """Fuses the new data with the existing data."""
        raise NotImplementedError

    def _get_local_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        scaled_depth = depth.copy()
        scaled_depth[depth == 0] = 1.0
        scaled_depth = (
            scaled_depth * (self._max_depth - self._min_depth) + self._min_depth
        )
        mask = scaled_depth < self._max_depth
        point_cloud = get_point_cloud(scaled_depth, mask, self._fx, self._fy)
        return point_cloud


def filter_points_by_height(
    points: np.ndarray, min_height: float, max_height: float
) -> np.ndarray:
    return points[(points[:, 2] >= min_height) & (points[:, 2] <= max_height)]


def xy_to_px(
    points: np.ndarray,
    pixels_per_meter: float,
    episode_pixel_origin: Tuple[int, int],
    grid_height: int,
) -> np.ndarray:
    """Converts an array of (x, y) coordinates to pixel coordinates.

    Args:
        points: The array of (x, y) coordinates to convert.

    Returns:
        The array of (x, y) pixel coordinates.
    """
    px = np.rint(points[:, ::-1] * pixels_per_meter) + episode_pixel_origin
    px[:, 0] = grid_height - px[:, 0]
    return px.astype(int)


def cloud_to_grid(
    point_cloud: np.ndarray,
    grid: np.ndarray,
    episode_origin: Tuple[int, int],
    pixels_per_meter: float,
):
    """Flattens a point cloud into a topdown grid.

    Args:
        point_cloud (np.ndarray): A point cloud in the form of a numpy array of shape
            (N, 3) where N is the number of points in the cloud. The first two columns
            are the x and y coordinates of the points in meters, and the third column
            is the z coordinate in meters, which will not be used.
        grid (np.ndarray): A numpy array of shape (H, W) where H and W are the height.
            The xy_to_px function will be used to convert the x and y coordinates of
            the points in the cloud to pixel coordinates, and the pixel values at those
            coordinates will be set to 1.
        episode_origin (Tuple[int, int]): The pixel coordinates of the origin of the
            episode.
        pixels_per_meter (float): The number of pixels per meter.
    """
    # Extract the x and y coordinates of the points in the cloud
    xy_points = point_cloud[:, :2]

    # Convert the x and y coordinates to pixel coordinates
    pixel_points = xy_to_px(xy_points, pixels_per_meter, episode_origin, grid.shape[0])

    # Set the pixel values at the pixel coordinates to 1
    grid[pixel_points[:, 1], pixel_points[:, 0]] = 1


def replay_from_dir():
    with open(KWARGS_JSON, "r") as f:
        kwargs = json.load(f)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    v = ObstacleMap(
        fov=kwargs["fov"],
        min_height=float(kwargs.get("min_height", 0.15)),
        max_height=float(kwargs.get("max_height", 0.88)),
        min_depth=kwargs["min_depth"],
        max_depth=kwargs["max_depth"],
        agent_radius=float(kwargs.get("agent_radius", 0.18)),
        size=kwargs["size"],
    )

    sorted_keys = sorted(list(data.keys()))

    for img_path in sorted_keys:
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        v.update_map(depth, tf_camera_to_episodic)

        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    replay_from_dir()
    quit()
