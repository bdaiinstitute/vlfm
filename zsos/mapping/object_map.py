import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from zsos.llm.prompts import get_textual_map_prompt, numbered_list, unnumbered_list
from zsos.utils.geometry_utils import (
    calculate_vfov,
    extract_yaw,
    within_fov_cone,
)


class Object:
    def __init__(self, class_name, location, confidence, too_far):
        self.class_name = class_name
        self.location = location
        self.confidence = confidence
        self.too_far = too_far
        self.explored = False

    def __repr__(self):
        return f"{self.class_name} at {self.location} with confidence {self.confidence}"


class ObjectMap:
    """This class is used to localize objects detected by the agent. The agent has
    access to a depth camera, bounding boxes provided by an object detector, and an
    estimate of its own position and yaw.
    """

    map: List[Object] = []
    image_width: int = None  # set upon execution of update_map method
    image_height: int = None  # set upon execution of update_map method

    def __init__(
        self,
        min_depth: float,
        max_depth: float,
        hfov: float,
        proximity_threshold: float,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.hfov = np.deg2rad(hfov)
        self.proximity_threshold = proximity_threshold
        self.camera_history = []

    @property
    def vfov(self) -> float:
        return calculate_vfov(self.hfov, self.image_width, self.image_height)

    def reset(self) -> None:
        self.map = []
        self.camera_history = []

    def update_map(
        self,
        object_name: str,
        bbox: np.ndarray,
        depth_img: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
        confidence: float,
    ) -> None:
        """Updates the object map with the latest information from the agent."""
        self.image_height, self.image_width = depth_img.shape[:2]
        location, too_far = self._estimate_object_location(
            bbox, depth_img, tf_camera_to_episodic
        )
        new_object = Object(object_name, location, confidence, too_far)
        self._add_object(new_object)

    def get_best_object(self, target_class: str) -> np.ndarray:
        """Returns the closest object to the agent that matches the given object name.
        It will ignore any detections of the target class if they are too far away,
        unless all the detections we have of the target class are too far away.

        Args:
            target_class (str): The name of the object class to search for.

        Returns:
            np.ndarray: The location of the closest object to the agent that matches the
                given object name [x, y, z].
        """
        matches = [obj for obj in self.map if obj.class_name == target_class]
        if len(matches) == 0:
            raise ValueError(
                f"No object of type {target_class} found in the object map."
            )

        ignore_too_far = any([not obj.too_far for obj in matches])
        best_loc, best_conf = None, -float("inf")
        for object_inst in matches:
            if object_inst.confidence > best_conf:
                if ignore_too_far and object_inst.too_far:
                    continue
                best_loc = object_inst.location
                best_conf = object_inst.confidence

        assert best_loc is not None, "This error should never be reached."

        return best_loc

    def update_explored(self, tf_camera_to_episodic: np.ndarray) -> None:
        camera_coordinates = tf_camera_to_episodic[:3, 3] / tf_camera_to_episodic[3, 3]
        camera_yaw = extract_yaw(tf_camera_to_episodic)

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
        # Remove objects that are both too far and explored
        self.map = [obj for obj in self.map if not (obj.explored and obj.too_far)]

    def get_textual_map_prompt(
        self, target: str, current_pos: np.ndarray, frontiers: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        """Returns a textual representation of the object map. The {target} field will
        still be unfilled.
        """
        # 'textual_map' is a list of strings, where each string represents the
        # object's name and location
        textual_map = objects_to_str(self.map, current_pos)
        textual_map = unnumbered_list(textual_map)

        # 'unexplored_objects' is a list of strings, where each string represents the
        # object's name and location
        unexplored_objects = [obj for obj in self.map if not obj.explored]
        unexplored_objects_strs = objects_to_str(unexplored_objects, current_pos)
        # For object_options, only return a list of objects that have not been explored
        object_options = numbered_list(unexplored_objects_strs)

        # 'frontiers_list' is a list of strings, where each string represents the
        # frontier's location
        frontiers_list = [
            f"({frontier[0]:.2f}, {frontier[1]:.2f})" for frontier in frontiers
        ]
        frontier_options = numbered_list(
            frontiers_list, start=len(unexplored_objects_strs) + 1
        )

        curr_pos_str = f"({current_pos[0]:.2f}, {current_pos[1]:.2f})"

        prompt = get_textual_map_prompt(
            target,
            textual_map,
            object_options,
            frontier_options,
            curr_position=curr_pos_str,
        )

        waypoints = []
        for obj in unexplored_objects:
            waypoints.append(obj.location)
        waypoints.extend(list(frontiers))
        waypoints = np.array(waypoints)

        return prompt, waypoints

    def visualize(self, frontiers: np.ndarray) -> np.ndarray:
        """Visualizes the object map by plotting the history of the camera coordinates
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

        for frontier in frontiers:
            if np.all(frontier == 0):  # ignore all zeros frontiers
                continue
            plot_circle(frontier, (255, 255, 255))

        visual_map = cv2.flip(visual_map, 0)
        visual_map = cv2.rotate(visual_map, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return visual_map

    def _estimate_object_location(
        self,
        bounding_box: np.ndarray,
        depth_image: np.ndarray,
        tf_camera_to_episodic: np.ndarray,
    ) -> Tuple[np.ndarray, bool]:
        """Estimates the location of a detected object in the global coordinate frame
        using a depth camera and a bounding box.

        Args:
            bounding_box (np.ndarray): The bounding box coordinates of the detected
                object in the image [x_min, y_min, x_max, y_max]. These coordinates are
                normalized to the range [0, 1].
            depth_image (np.ndarray): The depth image captured by the RGBD camera.
            tf_camera_to_episodic (np.ndarray): The transformation matrix from the camera
                coordinate frame to the episodic coordinate frame.

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
        object_coord_agent = np.append(object_coord_agent, 1)
        object_coord_global = tf_camera_to_episodic @ object_coord_agent
        object_coord_global = object_coord_global[:3] / object_coord_global[3]
        extract_yaw(tf_camera_to_episodic)
        # print("tf_camera_to_episodic", tf_camera_to_episodic)
        # print("camera_yaw", camera_yaw)
        # print("object_coord_agent", object_coord_agent)
        # print("object_coord_global", object_coord_global)

        return object_coord_global, too_far

    def _get_object_depth(
        self, depth: np.ndarray, object_bbox: np.ndarray
    ) -> Tuple[int, int, float]:
        """Gets the depth value of an object in the depth image.

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
        """Updates the map with a proposed Object instance using non-maximal
        suppression.

        Args:
            proposed_object (Object): The proposed Object to be added to the map.
        """
        updated_list = []

        for obj in self.map:
            keep = True
            same_name_and_close = (obj.class_name == proposed_object.class_name) and (
                np.linalg.norm(obj.location - proposed_object.location)
                <= self.proximity_threshold
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
    """Calculates the 3D coordinates (x, y, z) of a point in the image plane based on
    the horizontal field of view (HFOV), the image width and height, the depth value,
    and the pixel x and y coordinates.

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
    y = -hor_distance * math.sin(theta)
    ver_distance = depth_value * math.sin(theta)
    z = ver_distance * math.sin(phi)

    return np.array([x, y, z])


def objects_to_str(objs: List[Object], current_pos: np.ndarray) -> List[str]:
    """This function converts a list of object locations into strings. The list is first
    sorted based on the distance of each object from the agent's current position.

    Args:
        objs (List[Object]): A list of Object instances representing the objects.
        current_pos (np.ndarray): Current position of the agent.

    Returns:
        List[str]: A list where each string represents an object and its location in
        relation to the agent's position.
    """
    objs.sort(key=lambda obj: np.linalg.norm(obj.location[:2] - current_pos))
    objs = [f"{obj.class_name} at {obj_loc_to_str(obj.location)}" for obj in objs]
    return objs


def obj_loc_to_str(arr: np.ndarray) -> str:
    """Converts a numpy array representing an object's location into a string.

    Args:
        arr (np.ndarray): Object's coordinates.

    Returns:
        str: A string representation of the object's location with precision up to two
        decimal places.
    """
    return f"({arr[0]:.2f}, {arr[1]:.2f})"
