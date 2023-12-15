# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from typing import Any, Dict

import cv2
import numpy as np
from frontier_exploration.utils.general_utils import xyz_to_habitat

from vlfm.utils.geometry_utils import transform_points
from vlfm.utils.habitat_visualizer import sim_xy_to_grid_xy
from vlfm.utils.log_saver import log_episode


def log_episode_stats(episode_id: int, scene_id: str, infos: Dict) -> str:
    """Log episode stats to the console.

    Args:
        episode_id: The episode ID.
        scene_id: The scene ID.
        infos: The info dict from the environment after update with policy info.
    """
    scene = os.path.basename(scene_id).split(".")[0]
    if infos["success"] == 1:
        failure_cause = "did_not_fail"
    else:
        failure_cause = determine_failure_cause(infos)
        print(f"Episode {episode_id} in scene {scene} failed due to '{failure_cause}'.")

    if "ZSOS_LOG_DIR" in os.environ:
        infos_no_map = infos.copy()
        infos_no_map.pop("top_down_map")

        data = {
            "failure_cause": failure_cause,
            **remove_numpy_arrays(infos_no_map),
        }

        log_episode(episode_id, scene, data)

    return failure_cause


def determine_failure_cause(infos: Dict) -> str:
    """Using the info and policy_info dicts, determine the cause of failure.

    Args:
        infos: The info dict from the environment after update with policy info.

    Returns:
        A string describing the cause of failure.
    """
    if infos["target_detected"]:
        if was_false_positive(infos):
            return "false_positive"
        else:
            if infos["stop_called"]:
                return "bad_stop_true_positive"
            else:
                return "timeout_true_positive"
    else:
        if was_target_seen(infos):
            return "false_negative"
        else:
            if infos["traveled_stairs"]:
                cause = "never_saw_target_traveled_stairs"
            else:
                cause = "never_saw_target_did_not_travel_stairs"
            if not infos["top_down_map"]["is_feasible"]:
                return cause + "_likely_infeasible"
            else:
                return cause + "_feasible"


def was_target_seen(infos: Dict[str, Any]) -> bool:
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
    explored_area = infos["top_down_map"]["fog_of_war_mask"]
    # Dilate the target_bboxes_mask by 10 pixels to add a margin of error
    target_bboxes_mask = cv2.dilate(target_bboxes_mask, np.ones((10, 10)))
    target_explored = bool(np.any(np.logical_and(explored_area, target_bboxes_mask)))
    return target_explored


def was_false_positive(infos: Dict[str, Any]) -> bool:
    """Return whether the point goal target is within a bounding box."""
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
    nav_goal_episodic_xy = infos["nav_goal"]
    nav_goal_episodic_xyz = np.array(
        [nav_goal_episodic_xy[0], nav_goal_episodic_xy[1], 0]
    ).reshape(1, 3)

    upper_bound = infos["top_down_map"]["upper_bound"]
    lower_bound = infos["top_down_map"]["lower_bound"]
    grid_resolution = infos["top_down_map"]["grid_resolution"]
    tf_episodic_to_global = infos["top_down_map"]["tf_episodic_to_global"]

    nav_goal_global_xyz = transform_points(tf_episodic_to_global, nav_goal_episodic_xyz)
    nav_goal_global_habitat = xyz_to_habitat(nav_goal_global_xyz)
    nav_goal_global_habitat_xy = nav_goal_global_habitat[:, [2, 0]]

    grid_xy = sim_xy_to_grid_xy(
        upper_bound,
        lower_bound,
        grid_resolution,
        nav_goal_global_habitat_xy,
        remove_duplicates=True,
    )

    try:
        return target_bboxes_mask[grid_xy[0, 0], grid_xy[0, 1]] == 0
    except IndexError:
        # If the point goal is outside the map, assume it is a false positive
        return True


def remove_numpy_arrays(d: Any) -> Dict:
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = remove_numpy_arrays(value)
        elif not isinstance(value, np.ndarray):
            new_dict[key] = value

    return new_dict
