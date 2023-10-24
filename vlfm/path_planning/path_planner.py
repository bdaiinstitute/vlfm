# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import io
import sys
from typing import List, Tuple

import numpy as np

from .rrt import RRT, RRTStar


def get_paths(
    agent_pos: Tuple[float, float],
    waypoints: np.ndarray,
    occupancy_map: np.ndarray,
    rand_area: List[int],
    robot_radius: int,
    method: str = "rrt_star",
) -> List[np.ndarray]:
    paths = []

    for i in range(waypoints.shape[0]):
        # Silence the print statements
        text_trap = io.StringIO()
        sys.stdout = text_trap

        if method == "rrt":
            rrt = RRT(
                start=agent_pos,
                goal=waypoints[i, :],
                occupancy_map=occupancy_map,
                rand_area=rand_area,
                robot_radius=robot_radius,
            )

            path = rrt.planning(animation=False)

            # rrt.write_img(path)

        if method == "rrt_star":
            rrt_star = RRTStar(
                start=agent_pos,
                goal=waypoints[i, :],
                occupancy_map=occupancy_map,
                rand_area=rand_area,
                robot_radius=robot_radius,
            )

            path = rrt_star.planning(animation=False)

            # rrt_star.write_img(path)

        sys.stdout = sys.__stdout__

        print("START: ", agent_pos, "GOAL: ", waypoints[i, :], "RAND_AREA: ", rand_area)

        if path is not None:
            print("SUCCESS!")
            paths += [np.flip(np.array(path)[:-1], 0)]

    return paths
