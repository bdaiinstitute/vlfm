# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple

# import matplotlib.pyplot as plt
import cv2
import numpy as np

from external.python_robotics.PathPlanning.RRT.rrt import (
    RRT as RRT_PR,
)

# mypy: ignore-errors
from external.python_robotics.PathPlanning.RRTStar.rrt_star import (
    RRTStar as RRTStar_PR,
)

# mypy: ignore-errors
from vlfm.mapping.traj_visualizer import TrajectoryVisualizer


# Re-write collision check to use occupancy map instead of obstacle list
def check_collision_occmap(
    node: RRT_PR.Node, occupancy_map: np.ndarray, robot_radius: int
) -> bool:
    # print("ROBOT RADIUS: ", robot_radius, "OCCUPANCY_MAP SHAPE: ", occupancy_map.shape,
    #     "OCCUPANCY_MAP FILL: ", occupancy_map[:].sum())

    if node is None:
        print("Node is None")
        return False

    for i in range(len(node.path_x)):
        coord = (int(np.rint(node.path_y[i])), int(np.rint(node.path_x[i])))
        # print("NODE LOCATION: ", coord)
        if occupancy_map[coord[0], coord[1]]:
            return False

        # mp = pixel_value_within_radius(occupancy_map, coord, robot_radius, "max")
        # if mp > 0.5:
        #     # print("COLLIDED: ", coord, mp)
        #     return False # collision

    # print("NO COLLISION")

    return True  # safe


# Re-write draw_graph to use obstacle map instead of obstacle list
def visualize(
    img: np.ndarray,
    node_list: List[RRT_PR.Node],
    rnd_node: RRT_PR.Node,
    start: np.ndarray,
    end: np.ndarray,
    robot_radius: int,
    rand_n: int,
    occupancy_map: np.ndarray,
    traj_vis: TrajectoryVisualizer,
):
    if rnd_node is not None:
        node_coord = np.array([-rnd_node.y, -rnd_node.x])

        if occupancy_map[-int(np.rint(node_coord[0])), -int(np.rint(node_coord[1]))]:
            img = traj_vis.draw_circle(
                img, node_coord, color=(225, 0, 200), radius=robot_radius
            )
        else:
            img = traj_vis.draw_circle(
                img, node_coord, color=(0, 255, 0), radius=robot_radius
            )

    if len(node_list) > 0:
        for node in node_list:
            if node.parent:
                img = traj_vis._draw_future_path(
                    img,
                    np.vstack([-np.array(node.path_y), -np.array(node.path_x)]).T,
                    (0, 255, 0),
                )

    return img


def draw_final_path(
    img: np.ndarray, path: List[List[float]], traj_vis: TrajectoryVisualizer
):
    if len(path) > 1:
        path_arr = np.array(path)
        path_arr[:, [0, 1]] = path_arr[:, [1, 0]]
        path_arr[:, 1] *= -1
        path_arr[:, 0] *= -1
        img = traj_vis._draw_future_path(img, path_arr, (0, 0, 255))

    return img


class RRT(RRT_PR):
    def __init__(
        self,
        start: Tuple[float, float],
        goal: np.ndarray,
        occupancy_map: np.ndarray,
        rand_area: List[int],
        robot_radius: int,
    ):
        super().__init__(
            start=start,
            goal=goal,
            obstacle_list=occupancy_map,
            rand_area=rand_area,
            robot_radius=robot_radius,
            max_iter=1000,
            path_resolution=4.0,
            expand_dis=15.0,
            goal_sample_rate=5,
            play_area=None,
        )
        self.occupancy_map = occupancy_map
        self.img = cv2.cvtColor(
            occupancy_map.copy().astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
        )
        self.rand_n = np.random.randint(1000)

        self.traj_vis = TrajectoryVisualizer(np.array([0, 0]), 1.0)

    def draw_graph(self, rnd=None):
        self.img = visualize(
            self.img,
            self.node_list,
            rnd,
            np.array([-self.start.y, -self.start.x]),
            np.array([-self.end.y, -self.end.x]),
            self.robot_radius,
            self.rand_n,
            self.occupancy_map,
            self.traj_vis,
        )

    def write_img(self, final_path: List[float]):
        self.img = self.traj_vis.draw_circle(
            self.img,
            np.array([-self.start.y, -self.start.x]),
            color=(255, 0, 0),
            radius=self.robot_radius,
        )
        self.img = self.traj_vis.draw_circle(
            self.img,
            np.array([-self.end.y, -self.end.x]),
            color=(0, 0, 255),
            radius=self.robot_radius,
        )
        if final_path is not None:
            self.img = draw_final_path(self.img, final_path, self.traj_vis)
        cv2.imwrite(f"rrt_debug/RRTvis_{self.rand_n}.png", self.img)

    @staticmethod
    def check_collision(node, occupancy_map, robot_radius):
        return check_collision_occmap(node, occupancy_map, robot_radius)


class RRTStar(RRTStar_PR):
    def __init__(
        self,
        start: Tuple[float, float],
        goal: np.ndarray,
        occupancy_map: np.ndarray,
        rand_area: List[int],
        robot_radius: int,
    ):
        super().__init__(
            start=start,
            goal=goal,
            obstacle_list=occupancy_map,
            rand_area=rand_area,
            robot_radius=robot_radius,
            max_iter=1000,
            path_resolution=4.0,
            expand_dis=15.0,
            goal_sample_rate=5,
            connect_circle_dist=20,
        )
        self.occupancy_map = occupancy_map
        self.img = cv2.cvtColor(
            occupancy_map.copy().astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR
        )
        self.rand_n = np.random.randint(1000)

        self.traj_vis = TrajectoryVisualizer(np.array([0, 0]), 1.0)

    def draw_graph(self, rnd=None):
        self.img = visualize(
            self.img,
            self.node_list,
            rnd,
            np.array([-self.start.y, -self.start.x]),
            np.array([-self.end.y, -self.end.x]),
            self.robot_radius,
            self.rand_n,
            self.occupancy_map,
            self.traj_vis,
        )

    def write_img(self, final_path: List[float]):
        self.img = self.traj_vis.draw_circle(
            self.img,
            np.array([-self.start.y, -self.start.x]),
            color=(255, 0, 0),
            radius=self.robot_radius,
        )
        self.img = self.traj_vis.draw_circle(
            self.img,
            np.array([-self.end.y, -self.end.x]),
            color=(0, 0, 255),
            radius=self.robot_radius,
        )
        if final_path is not None:
            self.img = draw_final_path(self.img, final_path, self.traj_vis)
        cv2.imwrite(f"rrt_debug/RRTstarvis_{self.rand_n}.png", self.img)

    @staticmethod
    def check_collision(node, occupancy_map, robot_radius):
        return check_collision_occmap(node, occupancy_map, robot_radius)
