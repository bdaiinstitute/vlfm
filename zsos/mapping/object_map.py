import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from zsos.policy.utils.pointnav_policy import wrap_heading


class Object:
    def __init__(self, class_name, location, confidence, too_far):
        self.class_name = class_name
        self.location = location
        self.confidence = confidence
        self.too_far = too_far
        self.explored = False


class ObjectMap:
    """
    This class is used to localize objects detected by the agent. The agent has access
    to a depth camera, bounding boxes provided by an object detector, and an estimate of
    its own position and yaw.
    """

    map: List[Object] = []

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
        self.camera_history = []

    def reset(self) -> None:
        """
        Resets the object map.
        """
        self.map = []
        self.camera_history = []

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
        location, too_far = self._estimate_object_location(
            bbox, depth_img, agent_camera_position, agent_camera_yaw
        )
        new_object = Object(object_name, location, confidence, too_far)
        self._add_object(new_object)

    def get_best_object(self, target_class: str) -> np.ndarray:
        """
        Returns the closest object to the agent that matches the given object name.

        Args:
            target_class (str): The name of the object class to search for.

        Returns:
            np.ndarray: The location of the closest object to the agent that matches the
                given object name [x, y, z].
        """
        best_loc, best_conf = None, -float("inf")
        for object_inst in self.map:
            if (
                target_class == object_inst.class_name
                and object_inst.confidence > best_conf
            ):
                best_loc = object_inst.location
                best_conf = object_inst.confidence

        if best_loc is None:
            raise ValueError(
                f"No object of type {target_class} found in the object map."
            )

        return best_loc

    def update_explored(
        self, camera_coordinates: np.ndarray, camera_yaw: float
    ) -> None:
        self.camera_history.append((camera_coordinates, camera_yaw))
        for obj in self.map:
            if within_fov_cone(
                camera_coordinates,
                camera_yaw,
                self.hfov,
                self.max_depth,
                obj.location,
            ):
                obj.explored = True

    def visualize(self) -> np.ndarray:
        """
        Visualizes the object map by plotting the history of the camera coordinates
        and the location of each object in a 2D top-down view. If the object is
        explored, the object is plotted in a darker color. The map is a cv2 image with
        height and width of 400, with the origin at the center of the image, and each
        pixel representing 0.15 meters.
        """
        # Create black (blank) image of appropriate dimensions.
        visual_map = np.zeros((400, 400, 3), dtype=np.uint8)

        # Set the center of the map as (200,200)
        origin = np.array([200, 200])
        pixels_per_meter = 15

        def plot_circle(coordinates, circle_color):
            position = np.round(coordinates[:2] * pixels_per_meter).astype(int)

            # Add origin offset
            # y-axis in OpenCV goes from top to bottom, so need to invert y coordinate
            position = origin + (position * np.array([1, -1]))

            # Draw the camera on the map
            cv2.circle(visual_map, tuple(position), 2, circle_color, -1)

        for each_obj in self.map:
            # Explored objects are blue, unexplored objects are red
            color = (255, 0, 0) if each_obj.explored else (0, 0, 255)
            plot_circle(each_obj.location, color)

        for camera_c, _ in self.camera_history:
            plot_circle(camera_c, (0, 255, 0))

        visual_map = cv2.flip(visual_map, 0)
        visual_map = cv2.rotate(visual_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return visual_map

    def _estimate_object_location(
        self,
        bounding_box: np.ndarray,
        depth_image: np.ndarray,
        camera_coordinates: np.ndarray,
        camera_yaw: float,
    ) -> Tuple[np.ndarray, bool]:
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
            bool: True if the object is too far away for the depth camera, False
            otherwise.
        """
        # Get the depth value of the object
        pixel_x, pixel_y, depth_value = self._get_object_depth(
            depth_image, bounding_box
        )
        too_far = depth_value >= self.max_depth
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

        return object_coord_global, too_far

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

    def _add_object(self, proposed_object: Object) -> None:
        """
        Updates the map with a proposed Object instance using non-maximal suppression.

        Args:
            proposed_object (Object): The proposed Object to be added to the map.
        """

        updated_list = []
        proximity_threshold = 1.5

        for obj in self.map:
            keep = True
            same_name_and_close = (obj.class_name == proposed_object.class_name) and (
                np.linalg.norm(obj.location - proposed_object.location)
                <= proximity_threshold
            )

            if same_name_and_close:
                if obj.confidence > proposed_object.confidence:
                    # Do not modify map; proposed object was worse than what we already
                    # have in our current map
                    return
                else:
                    keep = False

            if keep:
                updated_list.append(obj)

        # Proposed object is added if no nearby, more confident object exists
        updated_list.append(proposed_object)
        self.map = updated_list


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


def within_fov_cone(
    cone_origin: np.ndarray,
    cone_angle: float,
    cone_fov: float,
    cone_range: float,
    point: np.ndarray,
) -> bool:
    """
    Checks if a point is within a cone of a given origin, angle, fov, and range.

    Args:
        cone_origin (np.ndarray): The origin of the cone.
        cone_angle (float): The angle of the cone in radians.
        cone_fov (float): The field of view of the cone in radians.
        cone_range (float): The range of the cone.
        point (np.ndarray): The point to check.

    """
    direction = point - cone_origin
    dist = np.linalg.norm(direction)
    angle = np.arctan2(direction[1], direction[0])
    angle_diff = wrap_heading(angle - cone_angle)

    return dist <= cone_range and abs(angle_diff) <= cone_fov / 2
