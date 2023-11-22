# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import io
import sys
from typing import List, Optional, Tuple

import numpy as np

from .rrt import RRT, RRTStar


def get_paths(
    agent_pos: Tuple[float, float],
    waypoints: np.ndarray,
    occupancy_map: np.ndarray,
    rand_area: List[int],
    robot_radius: int,
    method: str = "both",
    one_path: bool = False,
    n_paths: int = 4,
) -> List[np.ndarray]:
    paths = []

    for i in range(waypoints.shape[0]):
        # Silence the print statements
        text_trap = io.StringIO()
        sys.stdout = text_trap

        if one_path:
            path: Optional[np.ndarray] = None
            method_i = 0
            for j in range(50):
                if method == "both":  # switch which method is used
                    method_l = ["rrt", "rrt_star"][method_i]
                    method_i = (method_i + 1) % 2
                else:
                    method_l = method

                if method_l == "rrt":
                    rrt = RRT(
                        start=agent_pos,
                        goal=waypoints[i, :],
                        occupancy_map=occupancy_map,
                        rand_area=rand_area,
                        robot_radius=robot_radius,
                    )

                    path = rrt.planning(animation=False)
                elif method_l == "rrt_star":
                    rrt_star = RRTStar(
                        start=agent_pos,
                        goal=waypoints[i, :],
                        occupancy_map=occupancy_map,
                        rand_area=rand_area,
                        robot_radius=robot_radius,
                    )

                    path = rrt_star.planning(animation=False)
                else:
                    raise Exception(f"Invalid path generation method {method_l}")

                if path is not None:
                    break
            if path is not None:
                paths += [np.flip(np.array(path), 0)]

        else:
            if method == "rrt":
                for j in range(n_paths):
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
                for j in range(n_paths):
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
                for j in range(n_paths // 2):
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

                for j in range(n_paths - n_paths // 2):
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
