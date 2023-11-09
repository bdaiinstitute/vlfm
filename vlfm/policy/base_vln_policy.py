# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.obs_transformers.utils import image_resize
from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.utils.geometry_utils import rho_theta

try:
    import habitat
    from habitat_baselines.common.tensor_dict import TensorDict

    from vlfm.policy.base_policy import BasePolicy
except Exception:

    class BasePolicy:  # type: ignore
        pass


class BaseVLNPolicy(BasePolicy):
    _policy_info: Dict[str, Any] = {}
    _stop_action: Tensor = None  # MUST BE SET BY SUBCLASS
    _observations_cache: Dict[str, Any] = {}

    _max_iters: int = 500

    def __init__(
        self,
        pointnav_policy_path: str,
        depth_image_shape: Tuple[int, int],
        pointnav_stop_radius: float,
        use_gt_path: bool,
        options: Any,
        visualize: bool = True,
        compute_frontiers: bool = True,
        min_obstacle_height: float = 0.15,
        max_obstacle_height: float = 0.88,
        agent_radius: float = 0.18,
        obstacle_map_area_threshold: float = 1.5,
        hole_area_thresh: int = 100000,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(args, kwargs)
        # seperate function to allow for easy changing
        self._vl_model = None
        self.args = options
        self._use_gt_path = use_gt_path

        self._pointnav_policy = WrappedPointNavResNetPolicy(pointnav_policy_path)
        self._depth_image_shape = tuple(depth_image_shape)
        self._pointnav_stop_radius = pointnav_stop_radius
        self._visualize = visualize

        self._num_steps = 0
        self._did_reset = False
        self._last_goal = np.zeros(2)
        self._curr_instruction_idx = 0
        self._reached_goal = False

        self._done_initializing = False
        self._called_stop = False
        self._compute_frontiers = compute_frontiers
        if compute_frontiers:
            self._obstacle_map = ObstacleMap(
                min_height=min_obstacle_height,
                max_height=max_obstacle_height,
                area_thresh=obstacle_map_area_threshold,
                agent_radius=agent_radius,
                hole_area_thresh=hole_area_thresh,
            )

        self.gt_path_for_viz: Optional[np.ndarray] = None
        self.gt_path_world_coord: Optional[np.ndarray] = None

        self._instruction: str = ""
        self._instruction_parts: List[str] = []

        self.envs: Optional[habitat.core.vector_env.VectorEnv] = None

    def _reset(self) -> None:
        self._curr_instruction_idx = 0
        self._instruction_parts = []
        self._pointnav_policy.reset()
        self._last_goal = np.zeros(2)
        self._num_steps = 0
        self._done_initializing = False
        self._called_stop = False
        if self._compute_frontiers:
            self._obstacle_map.reset()
        self._did_reset = True
        self._reached_goal = False

        self.envs = None

    def set_gt_path_for_viz(
        self, gt_path_for_viz: np.ndarray, gt_path_world_coord: np.ndarray
    ) -> None:
        self.gt_path_for_viz = gt_path_for_viz
        self.gt_path_world_coord = gt_path_world_coord

    def set_instruction(self, instructions: str) -> None:
        self._instruction = instructions

    def set_envs(self, envs: habitat.core.vector_env.VectorEnv) -> None:
        self.envs = envs

    def act(
        self,
        observations: Dict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor]:  # type: ignore[override]
        """
        Starts the episode by 'initializing' and allowing robot to get its bearings
        (e.g., spinning in place to get a good view of the scene).
        Then, tries to follow the instruction.
        Once it has completed the whole instruction it stops.
        """
        # self._pre_step(observations, instructions, masks)

        self._observations_cache["object_map_rgbd"]

        self._observations_cache["robot_xy"]

        if self._num_steps > self._max_iters - 2:
            print("STOPPING (end iter)")
            return self._stop_action, rnn_hidden_states

        if not self._done_initializing:  # Initialize
            mode = "initialize"
            pointnav_action = self._initialize()
        else:
            mode = "navigate"
            goal, should_stop = self._plan()

            if should_stop:
                print("STOPPING (should_stop)")
                self._called_stop = True
                return self._stop_action, rnn_hidden_states
            if goal is None:
                print("No goal found so choosing random action!")
                pointnav_action = torch.tensor([[np.random.randint(1, 4)]])

            else:
                if self._use_gt_path:
                    pointnav_action = self._gtlocalnav(goal[:2])
                else:
                    pointnav_action = self._pointnav(goal[:2])

        action_numpy = pointnav_action.detach().cpu().numpy()[0]
        if len(action_numpy) == 1:
            action_numpy = action_numpy[0]
        if (mode != "navigate") or (goal is None):
            print(f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy}")
        else:
            xy = np.array2string(
                self._observations_cache["robot_xy"], precision=2, floatmode="fixed"
            )
            g = np.array2string(goal[:2], precision=2, floatmode="fixed")
            print(
                f"Step: {self._num_steps} | Mode: {mode} | Action: {action_numpy} |"
                f" Goal: {g} | Position: {xy}"
            )
        self._policy_info.update(self._get_policy_info())
        self._num_steps += 1

        self._observations_cache = {}
        self._did_reset = False

        return pointnav_action, rnn_hidden_states

    def _pre_step(self, observations: "TensorDict", masks: Tensor) -> None:
        assert masks.shape[1] == 1, "Currently only supporting one env at a time"
        if not self._did_reset and masks[0] == 0:
            self._reset()
            self._instruction_parts = self._parse_instruction(self._instruction)
        try:
            self._cache_observations(observations)
        except IndexError as e:
            print(e)
            print("Reached edge of map, stopping.")
            raise StopIteration
        self._policy_info = {}

    def _initialize(self) -> Tensor:
        raise NotImplementedError

    def _plan(self) -> Tuple[np.ndarray, bool]:
        raise NotImplementedError

    def _parse_instruction(self, instruction: str) -> List[str]:
        raise NotImplementedError

    def _choose_random_nonstop_action(self) -> torch.tensor:
        raise NotImplementedError

    def _get_policy_info(self) -> Dict[str, Any]:
        policy_info = {
            "instruction": self._instruction,
            "gps": str(self._observations_cache["robot_xy"] * np.array([1, -1])),
            "yaw": np.rad2deg(self._observations_cache["robot_heading"]),
            "nav_goal": self._last_goal,
            "stop_called": self._called_stop,
            # don't render these on egocentric images when making videos:
            "render_below_images": [
                "instruction",
            ],
        }

        if not self._visualize:
            return policy_info

        annotated_depth = self._observations_cache["object_map_rgbd"][0][1] * 255
        annotated_depth = cv2.cvtColor(
            annotated_depth.astype(np.uint8), cv2.COLOR_GRAY2RGB
        )

        annotated_rgb = self._observations_cache["object_map_rgbd"][0][0]

        policy_info["annotated_rgb"] = annotated_rgb
        policy_info["annotated_depth"] = annotated_depth

        if self._compute_frontiers:
            policy_info["obstacle_map"] = cv2.cvtColor(
                self._obstacle_map.visualize(gt_traj=self.gt_path_for_viz),
                cv2.COLOR_BGR2RGB,
            )

        if "DEBUG_INFO" in os.environ:
            policy_info["render_below_images"].append("debug")
            policy_info["debug"] = "debug: " + os.environ["DEBUG_INFO"]

        return policy_info

    def _gtlocalnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        assert self.envs is not None, "Trying to use gt paths without setting envs"
        action = self.envs.call(["get_gt_path_action"], [{"goal": goal}])[0]

        if action.item() == self._stop_action.item():
            print("GT path follower tried to stop, choosing random action!")
            self._reached_goal = True
            return self._choose_random_nonstop_action().to(action.device)

        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)

        if rho < self._pointnav_stop_radius:
            self._reached_goal = True

        return action

    def _pointnav(self, goal: np.ndarray, stop: bool = False) -> Tensor:
        """
        Calculates rho and theta from the robot's current position to the goal using the
        gps and heading sensors within the observations and the given goal, then uses
        it to determine the next action to take using the pre-trained pointnav policy.

        Args:
            goal (np.ndarray): The goal to navigate to as (x, y), where x and y are in
                meters.
            stop (bool): Whether to stop if we are close enough to the goal.

        """
        masks = torch.tensor([self._num_steps != 0], dtype=torch.bool, device="cuda")
        if not np.array_equal(goal, self._last_goal):
            if np.linalg.norm(goal - self._last_goal) > 0.1:
                self._pointnav_policy.reset()
                masks = torch.zeros_like(masks)
            self._last_goal = goal
        robot_xy = self._observations_cache["robot_xy"]
        heading = self._observations_cache["robot_heading"]
        rho, theta = rho_theta(robot_xy, heading, goal)
        rho_theta_tensor = torch.tensor(
            [[rho, theta]], device="cuda", dtype=torch.float32
        )
        obs_pointnav = {
            "depth": image_resize(
                self._observations_cache["nav_depth"],
                (self._depth_image_shape[0], self._depth_image_shape[1]),
                channels_last=True,
                interpolation_mode="area",
            ),
            "pointgoal_with_gps_compass": rho_theta_tensor,
        }
        self._policy_info["rho_theta"] = np.array([rho, theta])
        if rho < self._pointnav_stop_radius:
            self._reached_goal = True

            if stop:
                self._called_stop = True
                print("STOPPING (pointnav in stop logic)")
                return self._stop_action

        action = self._pointnav_policy.act(obs_pointnav, masks, deterministic=True)
        if action.item() == self._stop_action.item():
            print("Pointnav tried to stop, choosing random action!")
            return self._choose_random_nonstop_action().to(action.device)
        return action

    def _cache_observations(self, observations: "TensorDict") -> None:
        """Extracts the rgb, depth, and camera transform from the observations.

        Args:
            observations ("TensorDict"): The observations from the current timestep.
        """
        raise NotImplementedError

    def _infer_depth(
        self, rgb: np.ndarray, min_depth: float, max_depth: float
    ) -> np.ndarray:
        """Infers the depth image from the rgb image.

        Args:
            rgb (np.ndarray): The rgb image to infer the depth from.

        Returns:
            np.ndarray: The inferred depth image.
        """
        raise NotImplementedError


@dataclass
class ZSOSConfig:
    name: str = "HabitatVLNPolicy"
    pointnav_policy_path: str = "data/pointnav_weights.pth"
    use_gt_path: bool = False
    options: Any = None
    depth_image_shape: Tuple[int, int] = (224, 224)
    pointnav_stop_radius: float = 0.9
    use_max_confidence: bool = False
    obstacle_map_area_threshold: float = 1.5  # in square meters
    min_obstacle_height: float = 0.61
    max_obstacle_height: float = 0.88
    hole_area_thresh: int = 100000
    agent_radius: float = 0.18

    @classmethod  # type: ignore
    @property
    def kwaarg_names(cls) -> List[str]:
        # This returns all the fields listed above, except the name field
        return [f.name for f in fields(ZSOSConfig) if f.name != "name"]


cs = ConfigStore.instance()
cs.store(group="policy", name="zsos_config_base", node=ZSOSConfig())
