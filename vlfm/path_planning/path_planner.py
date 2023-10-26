# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import io
import sys
from typing import List, Tuple

import numpy as np

from .rrt import RRT, RRTStar

N_PATHS = 4


def get_paths(
    agent_pos: Tuple[float, float],
    waypoints: np.ndarray,
    occupancy_map: np.ndarray,
    rand_area: List[int],
    robot_radius: int,
    method: str = "both",
) -> List[np.ndarray]:
    paths = []

    for i in range(waypoints.shape[0]):
        # Silence the print statements
        text_trap = io.StringIO()
        sys.stdout = text_trap

        if method == "rrt":
            for j in range(N_PATHS):
                rrt = RRT(
                    start=agent_pos,
                    goal=waypoints[i, :],
                    occupancy_map=occupancy_map,
                    rand_area=rand_area,
                    robot_radius=robot_radius,
                )

                path = rrt.planning(animation=False)

                # rrt.write_img(path)

                if path is not None:
                    # print("SUCCESS!")
                    paths += [np.flip(np.array(path)[:-1], 0)]

        if method == "rrt_star":
            for j in range(N_PATHS):
                rrt_star = RRTStar(
                    start=agent_pos,
                    goal=waypoints[i, :],
                    occupancy_map=occupancy_map,
                    rand_area=rand_area,
                    robot_radius=robot_radius,
                )

                path = rrt_star.planning(animation=False)

                # rrt_star.write_img(path)

                if path is not None:
                    # print("SUCCESS!")
                    paths += [np.flip(np.array(path)[:-1], 0)]

        if method == "both":
            for j in range(N_PATHS // 2):
                rrt = RRT(
                    start=agent_pos,
                    goal=waypoints[i, :],
                    occupancy_map=occupancy_map,
                    rand_area=rand_area,
                    robot_radius=robot_radius,
                )

                path = rrt.planning(animation=False)

                # rrt.write_img(path)

                if path is not None:
                    # print("SUCCESS!")
                    paths += [np.flip(np.array(path)[:-1], 0)]

            for j in range(N_PATHS - N_PATHS // 2):
                rrt_star = RRTStar(
                    start=agent_pos,
                    goal=waypoints[i, :],
                    occupancy_map=occupancy_map,
                    rand_area=rand_area,
                    robot_radius=robot_radius,
                )

                path = rrt_star.planning(animation=False)

                # rrt_star.write_img(path)

                if path is not None:
                    # print("SUCCESS!")
                    paths += [np.flip(np.array(path)[:-1], 0)]

        sys.stdout = sys.__stdout__

        # print("START: ", agent_pos, "GOAL: ", waypoints[i, :], "RAND_AREA: ", rand_area)

    return paths
