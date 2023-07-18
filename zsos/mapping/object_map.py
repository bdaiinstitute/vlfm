import math
from typing import List, Optional, Tuple

import numpy as np


class ObjectMap:
    """
    This class is used to localize objects detected by the agent. The agent has access
    to a depth camera, bounding boxes provided by an object detector, and an estimate of
    its own position and yaw.
    """

    map: List[Tuple[str, np.ndarray, float]] = []  # class_name, location, confidence

    def __init__(
        self,
        min_depth: float = 0.5,
        max_depth: float = 5.0,
        hfov: float = 79.0,
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hfov = np.deg2rad(hfov)
        self.image_width = image_width
        self.image_height = image_height
        self.vfov = calculate_vfov(self.hfov, image_width, image_height)

    def reset(self) -> None:
        """
        Resets the object map.
        """
        self.map = []

    def update_map(
        self,
        object_name: str,
        bbox: np.ndarray,
        depth_img: np.ndarray,
        agent_camera_position: np.ndarray,
        agent_camera_yaw: float,
        confidence: float,
    ) -> None:
        """
        Updates the object map with the latest information from the agent.
        """
        location = self._estimate_object_location(
            bbox, depth_img, agent_camera_position, agent_camera_yaw
        )
        self._add_object(object_name, location, confidence)

    def get_best_object(self, object: str) -> np.ndarray:
        """
        Returns the closest object to the agent that matches the given object name.

        Args:
            object (str): The name of the object to search for.

        Returns:
            np.ndarray: The location of the closest object to the agent that matches the
                given object name [x, y, z].
        """
        best_loc, best_conf = None, -float("inf")
        for name, location, conf in self.map:
            if name == object and conf > best_conf:
                best_loc = location
                best_conf = conf

        if best_loc is None:
            raise ValueError(f"No object of type {object} found in the object map.")

        return best_loc

    def _estimate_object_location(
        self,
        bounding_box: np.ndarray,
        depth_image: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_yaw: float,
    ) -> np.ndarray:
        """
        Estimates the location of a detected object in the global coordinate frame using
        a depth camera and a bounding box.

        Args:
            bounding_box (np.ndarray): The bounding box coordinates of the detected
            object in the image [x_min, y_min, x_max, y_max]. These coordinates are
            normalized to the range [0, 1].
            depth_image (np.ndarray): The depth image captured by the RGBD camera.
            camera_coordinates (np.ndarray): The global coordinates of the camera
            [x, y, z].
            camera_yaw (float): The yaw angle of the camera in radians.

        Returns:
            np.ndarray: The estimated 3D location of the detected object in the global
            coordinate frame [x, y, z].
        """
        # Get the depth value of the object
        pixel_x, pixel_y, depth_value = self._get_object_depth(
            depth_image, bounding_box
        )
        # TODO: Be careful if the depth is the max depth value of the camera
        object_coord_agent = calculate_3d_coordinates(
            self.hfov,
            self.image_width,
            self.image_height,
            depth_value,
            pixel_x,
            pixel_y,
            vfov=self.vfov,
        )
        # Yaw from compass sensor must be negated to work properly
        object_coord_global = convert_to_global_frame(
            camera_coordinates, -camera_yaw, object_coord_agent
        )

        return object_coord_global

    def _get_object_depth(
        self, depth: np.ndarray, object_bbox: np.ndarray
    ) -> Tuple[int, int, float]:
        """
        Gets the depth value of an object in the depth image.

        Args:
            depth (np.ndarray): The depth image captured by the RGBD camera.
            object_bbox (np.ndarray): The bounding box coordinates of the detected
                object in the image [x_min, y_min, x_max, y_max]. These coordinates are
                normalized to the range [0, 1].

        Returns:
            Tuple[int, int, float]: The pixel coordinates of the center of the object
                bounding box and the determined depth value of the object.
        """
        x_min, y_min, x_max, y_max = object_bbox

        # Assert that all these values are in the range [0, 1]
        for i in [x_min, y_min, x_max, y_max]:
            assert -0.01 <= i <= 1.01, (
                "Bounding box coordinates must be in the range [0, 1], got:"
                f" {object_bbox}"
            )

        # In pixel space, calculate the center of the bounding box (integers)
        pixel_x = int((x_min + x_max) / 2 * self.image_width)
        pixel_y = int((y_min + y_max) / 2 * self.image_height)

        # Scale the bounding box to the depth image size using self.image_width and
        # self.image_height
        x_min_int = int(x_min * self.image_width)
        x_max_int = int(x_max * self.image_width)
        y_min_int = int(y_min * self.image_height)
        y_max_int = int(y_max * self.image_height)
        depth_image_chip = depth[y_min_int:y_max_int, x_min_int:x_max_int]

        # De-normalize the depth values
        depth_image_chip = (
            depth_image_chip * (self.max_depth - self.min_depth) + self.min_depth
        )

        depth_value = float(np.median(depth_image_chip))

        return pixel_x, pixel_y, depth_value

    def _add_object(
        self, object_name: str, position: np.ndarray, confidence: float
    ) -> None:
        """
        Adds an object to the map.
        """
        # TODO: do some type of filtering here (like non-max suppression)
        self.map.append((object_name, position, confidence))


def calculate_3d_coordinates(
    hfov: float,
    image_width: int,
    image_height: int,
    depth_value: float,
    pixel_x: int,
    pixel_y: int,
    vfov: Optional[float] = None,
) -> np.ndarray:
    """
    Calculates the 3D coordinates (x, y, z) of a point in the image plane based on the
    horizontal field of view (HFOV), the image width and height, the depth value, and
    the pixel x and y coordinates.

    Args:
        hfov (float): A float representing the HFOV in radians.
        image_width (int): Width of the image sensor in pixels.
        image_height (int): Height of the image sensor in pixels.
        depth_value (float): Distance of the point in the image plane from the camera
            center.
        pixel_x (int): The x coordinate of the point in the image plane.
        pixel_y (int): The y coordinate of the point in the image plane.
        vfov (Optional[float]): A float representing the VFOV in radians. If None, the
            VFOV is calculated from the HFOV, image width, and image height.

    Returns:
        np.ndarray: The 3D coordinates (x, y, z) of the point in the image plane.
    """
    # Calculate angle per pixel in the horizontal and vertical directions
    if vfov is None:
        vfov = calculate_vfov(hfov, image_width, image_height)
    hangle_per_pixel = hfov / image_width
    vangle_per_pixel = vfov / image_height

    # Calculate the horizontal and vertical angles from the center to the given pixel
    theta = hangle_per_pixel * (pixel_x - image_width / 2)
    phi = vangle_per_pixel * (pixel_y - image_height / 2)

    hor_distance = depth_value * math.cos(phi)
    x = hor_distance * math.cos(theta)

    y = hor_distance * math.sin(theta)

    ver_distance = depth_value * math.sin(theta)
    z = ver_distance * math.sin(phi)

    return np.array([x, y, z])


def calculate_vfov(hfov: float, width: int, height: int) -> float:
    """
    Calculates the vertical field of view (VFOV) based on the horizontal field of view
    (HFOV), width, and height of the image sensor.

    Args:
        hfov (float): The HFOV in radians.
        width (int): Width of the image sensor in pixels.
        height (int): Height of the image sensor in pixels.

    Returns:
        A float representing the VFOV in radians.
    """
    # Calculate the diagonal field of view (DFOV)
    dfov = 2 * math.atan(
        math.tan(hfov / 2)
        * math.sqrt((width**2 + height**2) / (width**2 + height**2))
    )

    # Calculate the vertical field of view (VFOV)
    vfov = 2 * math.atan(
        math.tan(dfov / 2) * (height / math.sqrt(width**2 + height**2))
    )

    return vfov


def convert_to_global_frame(
    agent_pos: np.ndarray, agent_yaw: float, local_pos: np.ndarray
) -> np.ndarray:
    """
    Converts a given position from the agent's local frame to the global frame.

    Args:
        agent_pos (np.ndarray): A 3D vector representing the agent's position in their
            local frame.
        agent_yaw (float): The agent's yaw in radians.
        local_pos (np.ndarray): A 3D vector representing the position to be converted in
            the agent's local frame.

    Returns:
        A 3D numpy array representing the position in the global frame.
    """
    # Construct the homogeneous transformation matrix
    x, y, z = agent_pos
    transformation_matrix = np.array(
        [
            [np.cos(agent_yaw), -np.sin(agent_yaw), 0, x],
            [np.sin(agent_yaw), np.cos(agent_yaw), 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ]
    )

    # Append a homogeneous coordinate of 1 to the local position vector
    local_pos_homogeneous = np.append(local_pos, 1)

    # Perform the transformation using matrix multiplication
    global_pos_homogeneous = transformation_matrix.dot(local_pos_homogeneous)
    global_pos_homogeneous = global_pos_homogeneous / global_pos_homogeneous[-1]

    return global_pos_homogeneous[:3]
