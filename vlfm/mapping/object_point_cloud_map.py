# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Dict, Union

import cv2
import numpy as np
import open3d as o3d

from vlfm.utils.geometry_utils import (
    extract_yaw,
    get_point_cloud,
    transform_points,
    within_fov_cone,
)


class ObjectPointCloudMap:
    clouds: Dict[str, np.ndarray] = {}
    use_dbscan: bool = True

    def __init__(self, erosion_size: float) -> None:
        self._erosion_size = erosion_size
        self.last_target_coord: Union[np.ndarray, None] = None

    def reset(self) -> None:
        self.clouds = {}
        self.last_target_coord = None

    def has_object(self, target_class: str) -> bool:
        return target_class in self.clouds and len(self.clouds[target_class]) > 0

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
        if len(local_cloud) == 0:
            return

        # For second-class, bad detections that are too offset or out of range, we
        # assign a random number to the last column of its point cloud that can later
        # be used to identify which points came from the same detection.
        if too_offset(object_mask):
            within_range = np.ones_like(local_cloud[:, 0]) * np.random.rand()
        else:
            # Mark all points of local_cloud whose distance from the camera is too far
            # as being out of range
            within_range = local_cloud[:, 0] <= max_depth * 0.95  # 5% margin
            # All values of 1 in within_range will be considered within range, and all
            # values of 0 will be considered out of range; these 0s need to be
            # assigned with a random number so that they can be identified later.
            within_range = within_range.astype(np.float32)
            within_range[within_range == 0] = np.random.rand()
        global_cloud = transform_points(tf_camera_to_episodic, local_cloud)
        global_cloud = np.concatenate((global_cloud, within_range[:, None]), axis=1)

        curr_position = tf_camera_to_episodic[:3, 3]
        closest_point = self._get_closest_point(global_cloud, curr_position)
        dist = np.linalg.norm(closest_point[:3] - curr_position)
        if dist < 1.0:
            # Object is too close to trust as a valid object
            return

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

        closest_point_2d = self._get_closest_point(target_cloud, curr_position)[:2]

        if self.last_target_coord is None:
            self.last_target_coord = closest_point_2d
        else:
            # Do NOT update self.last_target_coord if:
            # 1. the closest point is only slightly different
            # 2. the closest point is a little different, but the agent is too far for
            #    the difference to matter much
            delta_dist = np.linalg.norm(closest_point_2d - self.last_target_coord)
            if delta_dist < 0.1:
                # closest point is only slightly different
                return self.last_target_coord
            elif (
                delta_dist < 0.5
                and np.linalg.norm(curr_position - closest_point_2d) > 2.0
            ):
                # closest point is a little different, but the agent is too far for
                # the difference to matter much
                return self.last_target_coord
            else:
                self.last_target_coord = closest_point_2d

        return self.last_target_coord

    def update_explored(
        self, tf_camera_to_episodic: np.ndarray, max_depth: float, cone_fov: float
    ) -> None:
        """
        This method will remove all point clouds in self.clouds that were originally
        detected to be out-of-range, but are now within range. This is just a heuristic
        that suppresses ephemeral false positives that we now confirm are not actually
        target objects.

        Args:
            tf_camera_to_episodic: The transform from the camera to the episode frame.
            max_depth: The maximum distance from the camera that we consider to be
                within range.
            cone_fov: The field of view of the camera.
        """
        camera_coordinates = tf_camera_to_episodic[:3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

        for obj in self.clouds:
            within_range = within_fov_cone(
                camera_coordinates,
                camera_yaw,
                cone_fov,
                max_depth * 0.5,
                self.clouds[obj],
            )
            range_ids = set(within_range[..., -1].tolist())
            for range_id in range_ids:
                if range_id == 1:
                    # Detection was originally within range
                    continue
                # Remove all points from self.clouds[obj] that have the same range_id
                self.clouds[obj] = self.clouds[obj][
                    self.clouds[obj][..., -1] != range_id
                ]

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
        final_mask = cv2.erode(  # type: ignore
            final_mask, None, iterations=self._erosion_size
        )

        valid_depth = depth.copy()
        valid_depth[valid_depth == 0] = 1  # set all holes (0) to just be far (1)
        valid_depth = valid_depth * (max_depth - min_depth) + min_depth
        cloud = get_point_cloud(valid_depth, final_mask, fx, fy)
        cloud = get_random_subarray(cloud, 5000)
        if self.use_dbscan:
            cloud = open3d_dbscan_filtering(cloud)

        return cloud

    def _get_closest_point(
        self, cloud: np.ndarray, curr_position: np.ndarray
    ) -> np.ndarray:
        ndim = curr_position.shape[0]
        if self.use_dbscan:
            # Return the point that is closest to curr_position, which is 2D
            closest_point = cloud[
                np.argmin(np.linalg.norm(cloud[:, :ndim] - curr_position, axis=1))
            ]
        else:
            # Calculate the Euclidean distance from each point to the reference point
            if ndim == 2:
                ref_point = np.concatenate((curr_position, np.array([0.5])))
            else:
                ref_point = curr_position
            distances = np.linalg.norm(cloud[:, :3] - ref_point, axis=1)

            # Use argsort to get the indices that would sort the distances
            sorted_indices = np.argsort(distances)

            # Get the top 20% of points
            percent = 0.25
            top_percent = sorted_indices[: int(percent * len(cloud))]
            try:
                median_index = top_percent[int(len(top_percent) / 2)]
            except IndexError:
                median_index = 0
            closest_point = cloud[median_index]
        return closest_point


def open3d_dbscan_filtering(
    points: np.ndarray, eps: float = 0.2, min_points: int = 100
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


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
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


def get_random_subarray(points: np.ndarray, size: int) -> np.ndarray:
    """
    This function returns a subarray of a given 3D points array. The size of the
    subarray is specified by the user. The elements of the subarray are randomly
    selected from the original array. If the size of the original array is smaller than
    the specified size, the function will simply return the original array.

    Args:
        points (numpy array): A numpy array of 3D points. Each element of the array is a
            3D point represented as a numpy array of size 3.
        size (int): The desired size of the subarray.

    Returns:
        numpy array: A subarray of the original points array.
    """
    if len(points) <= size:
        return points
    indices = np.random.choice(len(points), size, replace=False)
    return points[indices]


def too_offset(mask: np.ndarray) -> bool:
    """
    This will return true if the entire bounding rectangle of the mask is either on the
    left or right third of the mask. This is used to determine if the object is too far
    to the side of the image to be a reliable detection.

    Args:
        mask (numpy array): A 2D numpy array of 0s and 1s representing the mask of the
            object.
    Returns:
        bool: True if the object is too offset, False otherwise.
    """
    # Find the bounding rectangle of the mask
    x, y, w, h = cv2.boundingRect(mask)

    # Calculate the thirds of the mask
    third = mask.shape[1] // 3

    # Check if the entire bounding rectangle is in the left or right third of the mask
    if x + w <= third:
        # Check if the leftmost point is at the edge of the image
        # return x == 0
        return x <= int(0.05 * mask.shape[1])
    elif x >= 2 * third:
        # Check if the rightmost point is at the edge of the image
        # return x + w == mask.shape[1]
        return x + w >= int(0.95 * mask.shape[1])
    else:
        return False
