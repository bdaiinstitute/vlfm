import glob
import json
import os
import os.path as osp
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from zsos.mapping.traj_visualizer import TrajectoryVisualizer
from zsos.utils.geometry_utils import extract_yaw, get_rotation_matrix
from zsos.utils.img_utils import (
    max_pixel_value_within_radius,
    monochannel_to_inferno_rgb,
    place_img_in_img,
    rotate_image,
)

DEBUG = False
RECORDING = False
RECORDING_DIR = "value_map_recordings"
JSON_PATH = osp.join(RECORDING_DIR, "data.json")
ARGS_TXT = osp.join(RECORDING_DIR, "args.txt")


class ValueMap:
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_mask: np.ndarray = None
    _camera_positions: List[np.ndarray] = []
    _last_camera_yaw: float = None
    use_max_confidence: bool = False

    def __init__(self, fov: float, max_depth: float):
        """
        Args:
            fov: The field of view of the camera in degrees.
            max_depth: The desired maximum depth of the camera in meters.
        """
        size = 1000
        self.pixels_per_meter = 20

        self.fov = np.deg2rad(fov)
        self.max_depth = max_depth
        self.value_map = np.zeros((size, size), np.float32)
        self.confidence_map = np.zeros((size, size), np.float32)
        self.episode_pixel_origin = np.array([size // 2, size // 2])
        self.min_confidence = 0.25
        self.decision_threshold = 0.35
        self.traj_vis = TrajectoryVisualizer(
            self.episode_pixel_origin, self.pixels_per_meter
        )

        if RECORDING:
            if osp.isdir(RECORDING_DIR):
                warnings.warn(
                    f"Recording directory {RECORDING_DIR} already exists. Deleting it."
                )
                shutil.rmtree(RECORDING_DIR)
            os.mkdir(RECORDING_DIR)
            # Dump all args to a file
            with open(ARGS_TXT, "w") as f:
                f.write(f"{fov},{max_depth}")
            # Create a blank .json file inside for now
            with open(JSON_PATH, "w") as f:
                f.write("{}")

    def reset(self):
        self.value_map.fill(0)
        self.confidence_map.fill(0)
        self._camera_positions = []
        self.traj_vis = TrajectoryVisualizer(
            self.episode_pixel_origin, self.pixels_per_meter
        )

    def update_map(
        self, depth: np.ndarray, tf_camera_to_episodic: np.ndarray, value: float
    ):
        """Updates the value map with the given depth image, pose, and value to use.

        Args:
            depth: The depth image to use for updating the map; expected to be already
                normalized to the range [0, 1].
            tf_camera_to_episodic: The transformation matrix from the episodic frame to
                the camera frame.
            value: The value to use for updating the map.
        """
        # Get new portion of the map
        curr_data = self._get_visible_mask(depth)

        # Rotate this new data to match the camera's orientation
        self._last_camera_yaw = yaw = extract_yaw(tf_camera_to_episodic)
        curr_data = rotate_image(curr_data, -yaw)

        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_camera_to_episodic[:2, 3] / tf_camera_to_episodic[3, 3]
        self._camera_positions.append(np.array([cam_x, cam_y]))

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self.episode_pixel_origin[0]
        py = int(-cam_y * self.pixels_per_meter) + self.episode_pixel_origin[1]

        # Overlay the new data onto the map
        blank_map = np.zeros_like(self.value_map)
        blank_map = place_img_in_img(blank_map, curr_data, px, py)

        # Fuse the new data with the existing data
        self._fuse_new_data(blank_map, value)

        if RECORDING:
            idx = len(glob.glob(osp.join(RECORDING_DIR, "*.png")))
            img_path = osp.join(RECORDING_DIR, f"{idx:04d}.png")
            cv2.imwrite(img_path, (depth * 255).astype(np.uint8))
            with open(JSON_PATH, "r") as f:
                data = json.load(f)
            data[img_path] = {
                "tf_camera_to_episodic": tf_camera_to_episodic.tolist(),
                "value": value,
            }
            with open(JSON_PATH, "w") as f:
                json.dump(data, f)

    def sort_waypoints(
        self, waypoints: np.ndarray, radius: float
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Selects the best waypoint from the given list of waypoints.

        Args:
            waypoints (np.ndarray): An array of 2D waypoints to choose from.

        Returns:
            Tuple[List[np.ndarray], List[float]]: The best waypoint and its associated value.
        """
        radius_px = int(radius * self.pixels_per_meter)

        def get_value(point: np.ndarray) -> float:
            x, y = point
            px = int(-x * self.pixels_per_meter) + self.episode_pixel_origin[0]
            py = int(-y * self.pixels_per_meter) + self.episode_pixel_origin[1]
            point_px = (self.value_map.shape[0] - px, py)
            value = max_pixel_value_within_radius(self.value_map, point_px, radius_px)
            return value

        values = [get_value(point) for point in waypoints]
        # Use np.argsort to get the indices of the sorted values
        sorted_inds = np.argsort([-v for v in values])  # sort in descending order
        sorted_values = [values[i] for i in sorted_inds]
        sorted_frontiers = [waypoints[i] for i in sorted_inds]

        return sorted_frontiers, sorted_values

    def visualize(
        self, markers: Optional[List[Tuple[np.ndarray, Dict[str, Any]]]] = None
    ) -> np.ndarray:
        """Return an image representation of the map"""
        # Must negate the y values to get the correct orientation
        # map_img = np.flipud(self.confidence_map * self.value_map)
        map_img = np.flipud(self.value_map)
        # Make all 0s in the value map equal to the max value, so they don't throw off
        # the color mapping (will revert later)
        zero_mask = map_img == 0
        map_img[zero_mask] = np.max(map_img)
        map_img = monochannel_to_inferno_rgb(map_img)
        # Revert all values that were originally zero to white
        map_img[zero_mask] = (255, 255, 255)
        if len(self._camera_positions) > 0:
            self.traj_vis.draw_trajectory(
                map_img,
                self._camera_positions,
                self._last_camera_yaw,
            )

            if markers is not None:
                for pos, marker_kwargs in markers:
                    map_img = self.traj_vis.draw_circle(map_img, pos, **marker_kwargs)

        return map_img

    def _get_visible_mask(self, depth: np.ndarray) -> np.ndarray:
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
        depth_row = np.max(depth, axis=0) * self.max_depth

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        angles = np.linspace(-self.fov / 2, self.fov / 2, len(depth_row))

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row * np.cos(angles)
        y = depth_row * np.sin(angles)

        # Get blank cone mask
        cone_mask = self._get_confidence_mask()

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
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)

        if DEBUG:
            vis = cv2.cvtColor((cone_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[1], point[0]] = (0, 255, 0)
            cv2.imshow("obstacle mask", vis)
            cv2.waitKey(0)

        return visible_mask

    def _get_blank_cone_mask(self) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        size = int(self.max_depth * self.pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(
            cone_mask,
            (size, size),  # center_pixel
            (size, size),  # axes lengths
            0,  # angle circle is rotated
            -np.rad2deg(self.fov) / 2 + 90,  # start_angle
            np.rad2deg(self.fov) / 2 + 90,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask

    def _get_confidence_mask(self) -> np.ndarray:
        """Generate a FOV cone with central values weighted more heavily"""
        if self._confidence_mask is not None:
            return self._confidence_mask.copy()
        cone_mask = self._get_blank_cone_mask()
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, self.fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self.min_confidence, 1)
                adjusted_mask[row, col] = confidence
        adjusted_mask = adjusted_mask * cone_mask
        self._confidence_mask = adjusted_mask.copy()

        return adjusted_mask

    def _fuse_new_data(self, confidence: np.ndarray, value: float):
        """Fuse the new data with the existing value and confidence map.

        Args:
            confidence: The new confidence map data to fuse. Confidences are between
                0 and 1, with 1 being the most confident.
            value: The value attributed to the new portion of the map.
        """
        # Any values in the given confidence map that are less confident than
        # self.decision_threshold AND less than the confidence in the existing map
        # will be re-assigned with a confidence of 0
        confidence_mask = np.logical_and(
            confidence < self.decision_threshold,
            confidence < self.confidence_map,
        )
        confidence[confidence_mask] = 0

        if self.use_max_confidence:
            # For every pixel that has a higher confidence in the new map than the
            # existing value map, replace the value in the existing value map with
            # the new value
            higher_confidence_mask = confidence > self.confidence_map
            self.value_map[higher_confidence_mask] = value
            # Update the confidence map with the new confidence values
            self.confidence_map[higher_confidence_mask] = confidence[
                higher_confidence_mask
            ]
        else:
            # Each pixel in the existing value map will be updated with a weighted
            # average of the existing value and the new value. The weight of each value
            # is determined by the current and new confidence values. The confidence map
            # will also be updated with using a weighted average in a similar manner.
            confidence_denominator = self.confidence_map + confidence
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                weight_1 = self.confidence_map / confidence_denominator
                weight_2 = confidence / confidence_denominator

            self.value_map = self.value_map * weight_1 + value * weight_2
            self.confidence_map = self.confidence_map * weight_1 + confidence * weight_2

            # Because confidence_denominator can have 0 values, any nans in either the
            # value or confidence maps will be replaced with 0
            self.value_map = np.nan_to_num(self.value_map)
            self.confidence_map = np.nan_to_num(self.confidence_map)


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


def replay_from_dir():
    with open(ARGS_TXT, "r") as f:
        lines = f.readlines()
        fov, max_depth = lines[0].split(",")
        fov, max_depth = float(fov), float(max_depth)
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    v = ValueMap(fov=fov, max_depth=max_depth)

    sorted_keys = sorted(list(data.keys()))

    for img_path in sorted_keys:
        tf_camera_to_episodic = np.array(data[img_path]["tf_camera_to_episodic"])
        value = data[img_path]["value"]
        depth = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        v.update_map(depth, tf_camera_to_episodic, value)

        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break


if __name__ == "__main__":
    # replay_from_dir()
    # quit()

    v = ValueMap(fov=79, max_depth=5.0)
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = v._get_visible_mask(depth)
    cv2.imshow("img", (img * 255).astype(np.uint8))
    cv2.waitKey(0)

    num_points = 20

    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    points = np.stack((x, y), axis=1)

    for pt, angle in zip(points, angles):
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        v.update_map(depth, tf, 1)
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
