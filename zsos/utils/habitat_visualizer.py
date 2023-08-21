from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.maps import MAP_TARGET_POINT_INDICATOR
from habitat.utils.visualizations.utils import overlay_frame
from habitat_baselines.common.tensor_dict import TensorDict

from frontier_exploration.utils.general_utils import xyz_to_habitat
from zsos.utils.geometry_utils import transform_points
from zsos.utils.img_utils import (
    crop_white_border,
    pad_larger_dim,
    pad_to_square,
    resize_images,
    rotate_image,
)
from zsos.utils.visualization import add_text_to_image, pad_images


class HabitatVis:
    def __init__(self):
        self.rgb = []
        self.depth = []
        self.maps = []
        self.cost_maps = []
        self.texts = []
        self.using_cost_map = False
        self.using_annotated_rgb = False
        self.using_annotated_depth = False

    def reset(self):
        self.rgb = []
        self.depth = []
        self.maps = []
        self.cost_maps = []
        self.texts = []
        self.using_annotated_rgb = False
        self.using_annotated_depth = False

    def collect_data(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
    ):
        assert len(infos) == 1, "Only support one environment for now"

        if "annotated_depth" in policy_info[0]:
            depth = policy_info[0]["annotated_depth"]
            self.using_annotated_depth = True
        else:
            depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[0])
        self.depth.append(depth)

        if "annotated_rgb" in policy_info[0]:
            rgb = policy_info[0]["annotated_rgb"]
            self.using_annotated_rgb = True
        else:
            rgb = observations["rgb"][0].cpu().numpy()
        self.rgb.append(rgb)

        # Visualize target point cloud on the map
        color_point_cloud_on_map(infos, policy_info)

        map = maps.colorize_draw_agent_and_fit_to_height(
            infos[0]["top_down_map"], self.depth[0].shape[0]
        )
        self.maps.append(map)
        if "cost_map" in policy_info[0]:
            cost_map = policy_info[0]["cost_map"]
            # Rotate the cost map to match the agent's orientation at the start
            # of the episode
            start_yaw = infos[0]["start_yaw"]
            cost_map = rotate_image(cost_map, start_yaw, border_value=(255, 255, 255))
            # Remove unnecessary white space around the edges
            cost_map = crop_white_border(cost_map)
            # Make the image at least 150 pixels tall or wide
            cost_map = pad_larger_dim(cost_map, 150)
            # Rotate the image if the corresponding map is taller than it is wide
            map = infos[0]["top_down_map"]["map"]
            if map.shape[0] > map.shape[1]:
                cost_map = np.rot90(cost_map, 1)
            # Pad the shorter dimension to be the same size as the longer
            cost_map = pad_to_square(cost_map, extra_pad=50)
            # Pad the image border with some white space
            cost_map = cv2.copyMakeBorder(
                cost_map, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            self.cost_maps.append(cost_map)
            self.using_cost_map = True
        else:
            self.cost_maps.append(np.ones_like(self.maps[0]) * 255)
        text = [
            policy_info[0][text_key]
            for text_key in policy_info[0].get("render_below_images", [])
            if text_key in policy_info[0]
        ]
        self.texts.append(text)

    def flush_frames(self) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the annotated frames are actually one step delayed, pop the first one
        # and add a placeholder frame to the end (gets removed anyway)
        if self.using_annotated_rgb is not None:
            self.rgb.append(self.rgb.pop(0))
        if self.using_annotated_depth is not None:
            self.depth.append(self.depth.pop(0))
        if self.using_cost_map:  # Cost maps are also one step delayed
            self.cost_maps.append(self.cost_maps.pop(0))

        frames = []
        num_frames = len(self.depth) - 1  # last frame is from next episode, remove it
        for i in range(num_frames):
            frame = self._create_frame(
                self.depth[i],
                self.rgb[i],
                self.maps[i],
                self.cost_maps[i],
                self.texts[i],
            )
            frames.append(frame)

        frames = pad_images(frames, pad_from_top=True)
        self.reset()

        return frames

    @staticmethod
    def _create_frame(depth, rgb, map, cost_map, text):
        """Create a frame with depth, rgb, map, cost_map, and text"""
        row_1 = np.hstack([depth, rgb])
        map, cost_map = resize_images([map, cost_map], match_dimension="height")
        row_2 = np.hstack([map, cost_map])
        row_2_height_scaled = int(row_2.shape[0] * (row_1.shape[1] / row_2.shape[1]))
        row_2_scaled = cv2.resize(row_2, (row_1.shape[1], row_2_height_scaled))
        frame = np.vstack([row_1, row_2_scaled])

        # Add text to the top of the frame
        for t in text[::-1]:
            frame = add_text_to_image(frame, t, top=True)

        return frame


def sim_xy_to_grid_xy(
    upper_bound: Tuple[int, int],
    lower_bound: Tuple[int, int],
    grid_resolution: Tuple[int, int],
    sim_xy: np.ndarray,
    remove_duplicates: bool = True,
) -> np.ndarray:
    """Converts simulation coordinates to grid coordinates.

    Args:
        upper_bound (Tuple[int, int]): The upper bound of the grid.
        lower_bound (Tuple[int, int]): The lower bound of the grid.
        grid_resolution (Tuple[int, int]): The resolution of the grid.
        sim_xy (np.ndarray): A numpy array of 2D simulation coordinates.
        remove_duplicates (bool): Whether to remove duplicate grid coordinates.

    Returns:
        np.ndarray: A numpy array of 2D grid coordinates.
    """
    grid_size = np.array(
        [
            abs(upper_bound[1] - lower_bound[1]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        ]
    )
    grid_xy = ((sim_xy - lower_bound[::-1]) / grid_size).astype(int)

    if remove_duplicates:
        grid_xy = np.unique(grid_xy, axis=0)

    return grid_xy


def color_point_cloud_on_map(infos, policy_info):
    if len(policy_info[0]["target_point_cloud"]) == 0:
        return

    upper_bound = infos[0]["top_down_map"]["upper_bound"]
    lower_bound = infos[0]["top_down_map"]["lower_bound"]
    grid_resolution = infos[0]["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos[0]["top_down_map"]["tf_episodic_to_global"]

    cloud_episodic_frame = policy_info[0]["target_point_cloud"][:, :3]
    cloud_global_frame_xyz = transform_points(
        tf_episodic_to_global, cloud_episodic_frame
    )
    cloud_global_frame_habitat = xyz_to_habitat(cloud_global_frame_xyz)
    cloud_global_frame_habitat_xy = cloud_global_frame_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        cloud_global_frame_habitat_xy,
        remove_duplicates=True,
    )

    new_map = infos[0]["top_down_map"]["map"].copy()
    for x, y in grid_xy:
        new_map[x, y] = MAP_TARGET_POINT_INDICATOR

    infos[0]["top_down_map"]["map"] = new_map
