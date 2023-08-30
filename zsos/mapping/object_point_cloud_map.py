from typing import Dict

import cv2
import numpy as np
import open3d as o3d

from zsos.utils.geometry_utils import get_point_cloud, transform_points


class ObjectPointCloudMap:
    clouds: Dict[str, np.ndarray] = {}
    use_dbscan: bool = False

    def __init__(self, erosion_size: float) -> None:
        self._erosion_size = erosion_size

    def reset(self):
        self.clouds = {}

    def has_object(self, target_class: str) -> bool:
        return target_class in self.clouds

    def update_map(
        self,
        object_name: str,
        depth_img: np.ndarray,
        object_mask: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        local_cloud = self._extract_object_cloud(
            depth_img, object_mask, min_depth, max_depth, fx, fy
        )

        # Mark all points of local_cloud whose distance from the camera is too far
        # as being out of range
        within_range = local_cloud[:, 0] <= max_depth * 0.95  # 5% margin
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        if object_name in self.clouds:
            self.clouds[object_name] = np.concatenate(
                (self.clouds[object_name], global_cloud), axis=0
            )
        else:
            self.clouds[object_name] = global_cloud

    def get_best_object(
        self, target_class: str, curr_position: np.ndarray
    ) -> np.ndarray:
        target_cloud = self.get_target_cloud(target_class)

        if self.use_dbscan:
            # Return the point that is closest to curr_position, which is 2D
            closest_point = target_cloud[
                np.argmin(np.linalg.norm(target_cloud[:, :2] - curr_position, axis=1))
            ]
        else:
            # Calculate the Euclidean distance from each point to the reference point
            ref_point = np.concatenate((curr_position, np.array([0.5])))
            distances = np.linalg.norm(target_cloud[:, :3] - ref_point, axis=1)

            # Use argsort to get the indices that would sort the distances
            sorted_indices = np.argsort(distances)

            # Get the top 20% of points
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(target_cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = target_cloud[median_index]

        closest_point_2d = closest_point[:2]

        return closest_point_2d

    def update_explored(self, *args, **kwargs):
        pass

    def get_target_cloud(self, target_class: str) -> np.ndarray:
        target_cloud = self.clouds[target_class].copy()
        # Determine whether any points are within range
        within_range_exists: bool = np.any(target_cloud[:, -1] == 1)
        if within_range_exists:
            # Filter out all points that are not within range
            target_cloud = target_cloud[target_cloud[:, -1] == 1]
        return target_cloud

    def _extract_object_cloud(
        self,
        depth: np.ndarray,
        object_mask: np.ndarray,
        min_depth: float,
        max_depth: float,
        fx: float,
        fy: float,
    ) -> np.ndarray:
        final_mask = object_mask * 255
        final_mask = cv2.erode(final_mask, None, iterations=self._erosion_size)

        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)

        return cloud


def open3d_dbscan_filtering(
    points, eps: float = 0.2, min_points: int = 100
) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps, min_points))

    # Count the points in each cluster
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # Exclude noise points, which are given the label -1
    non_noise_labels_mask = unique_labels != -1
    non_noise_labels = unique_labels[non_noise_labels_mask]
    non_noise_label_counts = label_counts[non_noise_labels_mask]

    if len(non_noise_labels) == 0:  # only noise was detected
        return np.array([])

    # Find the label of the largest non-noise cluster
    largest_cluster_label = non_noise_labels[np.argmax(non_noise_label_counts)]

    # Get the indices of points in the largest non-noise cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]

    # Get the points in the largest non-noise cluster
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points


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
