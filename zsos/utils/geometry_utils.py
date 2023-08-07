from typing import Union, Tuple

import numpy as np


def wrap_heading(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


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
