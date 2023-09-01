import json
import os
from typing import Any, Dict

import cv2
import numpy as np

from frontier_exploration.utils.general_utils import xyz_to_habitat
from zsos.utils.geometry_utils import transform_points
from zsos.utils.habitat_visualizer import sim_xy_to_grid_xy


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
        log_dir = os.environ["ZSOS_LOG_DIR"]
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            pass
        base = f"{episode_id}_{scene}.json"
        filename = os.path.join(log_dir, base)

        infos_no_map = infos.copy()
        infos_no_map.pop("top_down_map")

        data = {
            "episode_id": episode_id,
            "scene_id": scene_id,
            "failure_cause": failure_cause,
            **remove_numpy_arrays(infos_no_map),
        }

        # Skip if the filename already exists AND it isn't empty
        if not (os.path.exists(filename) and os.path.getsize(filename) > 0):
            print(f"Logging episode {int(episode_id):04d} to {filename}")
            with open(filename, "w") as f:
                json.dump(data, f, indent=4)

    return failure_cause


def determine_failure_cause(infos: Dict) -> str:
    """Using the info and policy_info dicts, determine the cause of failure.

    Args:
        infos: The info dict from the environment after update with policy info.

    Returns:
        A string describing the cause of failure.
    """
    if not infos["top_down_map"]["is_feasible"]:
        return "infeasible"
    elif infos["target_detected"]:
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
                return "never_saw_target_traveled_stairs"
            else:
                return "never_saw_target"


def was_target_seen(infos: Dict[str, Any]) -> bool:
    target_bboxes_mask = infos["top_down_map"]["target_bboxes_mask"]
    explored_area = infos["top_down_map"]["fog_of_war_mask"]
    # Dilate the target_bboxes_mask by 10 pixels to add a margin of error
    target_bboxes_mask = cv2.dilate(target_bboxes_mask, np.ones((10, 10)))
    target_explored = np.any(np.logical_and(explored_area, target_bboxes_mask))
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

    return target_bboxes_mask[grid_xy[0, 0], grid_xy[0, 1]] == 0


def remove_numpy_arrays(d: Dict) -> Dict:
    if not isinstance(d, dict):
        return d

    new_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = remove_numpy_arrays(value)
        elif not isinstance(value, np.ndarray):
            new_dict[key] = value

    return new_dict
