from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d

from zsos.utils.geometry_utils import calculate_vfov, transform_points
from zsos.vlm.sam import MobileSAMClient


class ObjectPointCloudMap:
    _clouds: Dict[str, np.ndarray] = {}
    _image_width: int = None  # set upon execution of update_map method
    _image_height: int = None  # set upon execution of update_map method
    __vfov: float = None  # set upon execution of update_map method

    def __init__(self, min_depth: float, max_depth: float, hfov: float) -> None:
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._hfov = np.deg2rad(hfov)
        self._mobile_sam = MobileSAMClient()

    @property
    def _vfov(self) -> float:
        if self.__vfov is None:
            self.__vfov = calculate_vfov(
                self._hfov, self._image_width, self._image_height
            )
        return self.__vfov

    def reset(self):
        self._clouds = {}

    def has_object(self, target_class: str) -> bool:
        return target_class in self._clouds

    def update_map(
        self,
        object_name: str,
        bbox: np.ndarray,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        *args,
        **kwargs,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        self._image_height, self._image_width = depth_img.shape[:2]
        local_cloud, too_far = self._extract_object_cloud(rgb_img, depth_img, bbox)
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)

        # Mark all clouds that are close enough by making a new column that is 0 or 1
        # based on whether the point is within range or not
        if too_far:
            within_range = np.zeros((global_cloud.shape[0],))
        else:
            within_range = np.ones((global_cloud.shape[0],))
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        if object_name in self._clouds:
            self._clouds[object_name] = np.concatenate(
                (self._clouds[object_name], global_cloud), axis=0
            )
        else:
            self._clouds[object_name] = global_cloud

    def get_best_object(
        self, target_class: str, curr_position: np.ndarray
    ) -> np.ndarray:
        target_cloud = self._clouds[target_class].copy()

        # Determine whether any points are within range
        within_range_exists: bool = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # Filter out all points that are not within range
            target_cloud = target_cloud[target_cloud[:, -1] == 1]

        # Return the point that is closest to curr_position, which is 2D
        closest_point = target_cloud[
            np.argmin(np.linalg.norm(target_cloud[:, :2] - curr_position, axis=1))
        ]
        closest_point_2d = closest_point[:2]

        return closest_point_2d

    def update_explored(self, *args, **kwargs):
        pass

    def _extract_object_cloud(
        self, rgb: np.ndarray, depth: np.ndarray, object_bbox: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        # Assert that all these values are in the range [0, 1]
        for i in object_bbox:
            assert -0.01 <= i <= 1.01, (
                "Bounding box coordinates must be in the range [0, 1], got:"
                f" {object_bbox}"
            )

        # De-normalize the bounding box coordinates
        bbox_denorm = object_bbox * np.array(
            [
                self._image_width,
                self._image_height,
                self._image_width,
                self._image_height,
            ]
        )
        object_mask = self._mobile_sam.segment_bbox(rgb, bbox_denorm.tolist())
        valid_depth = depth.reshape(depth.shape[:2])

        object_depths = valid_depth[object_mask]
        far_pixels = np.sum(object_depths > 0.85)
        total_pixels = object_depths.shape[0]
        far_pixel_ratio = far_pixels / total_pixels
        center_mask = np.zeros_like(object_mask)
        # Make the middle 50% of center_mask True
        center_mask[
            int(0.25 * self._image_height) : int(0.75 * self._image_height),
            int(0.25 * self._image_width) : int(0.75 * self._image_width),
        ] = True
        by_the_edges = np.sum(object_mask & center_mask) == 0
        too_far = far_pixel_ratio > 0.2 or by_the_edges

        valid_depth[valid_depth == 0] = 1
        valid_depth = (
            valid_depth * (self._max_depth - self._min_depth) + self._min_depth
        )
        cloud = get_point_cloud(valid_depth, object_mask, self._hfov, self._vfov)

        return cloud, too_far


def calculate_3d_coordinates_vectorized(
    hfov: float,
    image_width: int,
    image_height: int,
    depth_values: np.ndarray,
    pixel_x_values: np.ndarray,
    pixel_y_values: np.ndarray,
    vfov: Optional[float] = None,
) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        hfov (float): A float representing the HFOV in radians.
        image_width (int): Width of the image sensor in pixels.
        image_height (int): Height of the image sensor in pixels.
        depth_values (np.ndarray): Array of distances of the points in the image plane
            from the camera center.
        pixel_x_values (np.ndarray): Array of x coordinates of the points in the image
            plane.
        pixel_y_values (np.ndarray): Array of y coordinates of the points in the image
            plane.
        vfov (Optional[float]): A float representing the VFOV in radians. If None, the
            VFOV is calculated from the HFOV, image width, and image height.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    # Calculate angle per pixel in the horizontal and vertical directions
    if vfov is None:
        vfov = calculate_vfov(hfov, image_width, image_height)

    hangle_per_pixel = hfov / image_width
    vangle_per_pixel = vfov / image_height

    # Calculate the horizontal and vertical angles from the center to the given pixels
    theta_values = hangle_per_pixel * (pixel_x_values - image_width / 2)
    phi_values = vangle_per_pixel * (pixel_y_values - image_height / 2)

    hor_distances = depth_values * np.cos(phi_values)
    x_values = hor_distances * np.cos(theta_values)
    y_values = -hor_distances * np.sin(theta_values)
    ver_distances = depth_values * np.sin(theta_values)
    z_values = ver_distances * np.sin(phi_values)

    return np.column_stack((x_values, y_values, z_values))


def get_point_cloud(
    depth_image: np.ndarray,
    mask: np.ndarray,
    hfov: float,
    vfov: Optional[float] = None,
) -> np.ndarray:
    """Calculates the 3D coordinates (x, y, z) of points in the depth image based on
    the horizontal field of view (HFOV), the image width and height, the depth values,
    and the pixel x and y coordinates.

    Args:
        depth_image (np.ndarray): 2D depth image.
        mask (np.ndarray): 2D binary mask identifying relevant pixels.
        hfov (float): A float representing the HFOV in radians.
        vfov (Optional[float]): A float representing the VFOV in radians. If None, the
            VFOV is calculated from the HFOV, image width, and image height.

    Returns:
        np.ndarray: Array of 3D coordinates (x, y, z) of the points in the image plane.
    """
    pixel_y_values, pixel_x_values = np.where(mask)
    depth_values = depth_image[pixel_y_values, pixel_x_values]
    cloud = calculate_3d_coordinates_vectorized(
        hfov,
        depth_image.shape[1],
        depth_image.shape[0],
        depth_values,
        pixel_x_values,
        pixel_y_values,
        vfov,
    )
    cloud = open3d_statistical_outlier_removal(cloud)

    return cloud


def open3d_statistical_outlier_removal(points, nb_neighbors=20, std_ratio=0.5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    return np.asarray(pcd.points)


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str):
    """Visualizes an array of 3D points and saves the visualization as a PNG image.

    Args:
        point_cloud (np.ndarray): Array of 3D points with shape (N, 3).
        save_path (str): Path to save the PNG image.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.savefig(save_path)
    plt.close()
