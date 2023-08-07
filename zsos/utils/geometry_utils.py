import math
from typing import Tuple, Union

import numpy as np


def rho_theta(
    curr_pos: np.ndarray, curr_heading: float, curr_goal: np.ndarray
) -> Tuple[float, float]:
    """
    Calculates polar coordinates (rho, theta) relative to a given position and heading
    to a given goal position. 'rho' is the distance from the agent to the goal, and
    theta is how many radians the agent must turn (to the left, CCW from above) to face
    the goal. Coordinates are in (x, y), where x is the distance forward/backwards, and
    y is the distance to the left or right (right is negative)

    Args:
        curr_pos (np.ndarray): Array of shape (2,) representing the current position.
        curr_heading (float): The current heading, in radians. It represents how many
            radians  the agent must turn to the left (CCW from above) from its initial
            heading to reach its current heading.
        curr_goal (np.ndarray): Array of shape (2,) representing the goal position.

    Returns:
        Tuple[float, float]: A tuple of floats representing the polar coordinates
            (rho, theta).
    """
    rotation_matrix = np.array(
        [
            [np.cos(-curr_heading), -np.sin(-curr_heading)],
            [np.sin(-curr_heading), np.cos(-curr_heading)],
        ]
    )
    local_goal = curr_goal - curr_pos
    local_goal = rotation_matrix @ local_goal

    rho = np.linalg.norm(local_goal)
    theta = np.arctan2(local_goal[1], local_goal[0])

    return rho, theta


def wrap_heading(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


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
