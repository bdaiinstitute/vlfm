from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

try:
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy

    HABITAT_BASELINES_AVAILABLE = True
except ModuleNotFoundError:
    from zsos.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetPolicy,
    )

    HABITAT_BASELINES_AVAILABLE = False

from torch import Tensor


class WrappedPointNavResNetPolicy:
    """
    Wrapper for the PointNavResNetPolicy that allows for easier usage, however it can
    only handle one environment at a time. Automatically updates the hidden state
    and previous action for the policy.
    """

    def __init__(
        self,
        ckpt_path: str,
        device: Union[str, torch.device] = "cuda",
        discrete_actions: bool = True,
    ):
        if isinstance(device, str):
            device = torch.device(device)
        self.policy = load_pointnav_policy(ckpt_path)
        self.policy.to(device)
        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments.
            self.policy.net.num_recurrent_layers,
            512,  # hidden state size
            device=device,
        )
        if discrete_actions:
            num_actions = 1
            action_dtype = torch.long
        else:
            num_actions = 2
            action_dtype = torch.float32
        self.pointnav_prev_actions = torch.zeros(
            1,  # number of environments
            num_actions,
            device=device,
            dtype=action_dtype,
        )
        self.device = device

    def act(
        self,
        observations: Union["TensorDict", Dict],  # noqa: F821
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Determines the best action to take towards the given (rho, theta) based on
        depth vision.

        Args:
            observations (Union["TensorDict", Dict]): A dictionary containing (at least)
                the following:
                    - "depth" (torch.float32): Depth image tensor (N, H, W, 1).
                    - "pointgoal_with_gps_compass" (torch.float32):
                        PointGoalWithGPSCompassSensor tensor representing a rho and
                        theta w.r.t. to the agent's current pose (N, 2).
            masks (torch.bool): Tensor of masks, with a value of 1 for any step after
                the first in an episode; has 0 for first step.
            deterministic (bool): Whether to select a logit action deterministically.

        Returns:
            Tensor (torch.dtype.long): A tensor denoting the action to take:
                (0: STOP, 1: FWD, 2: LEFT, 3: RIGHT).
        """
        # Convert numpy arrays to torch tensors for each dict value
        for k, v in observations.items():
            if isinstance(v, np.ndarray):
                observations[k] = torch.from_numpy(v).to(
                    device=self.device, dtype=torch.float32
                )
                if k == "depth" and len(observations[k].shape) == 3:
                    observations[k] = observations[k].unsqueeze(0)
                elif (
                    k == "pointgoal_with_gps_compass"
                    and len(observations[k].shape) == 1
                ):
                    observations[k] = observations[k].unsqueeze(0)
        pointnav_action = self.policy.act(
            observations,
            self.pointnav_test_recurrent_hidden_states,
            self.pointnav_prev_actions,
            masks,
            deterministic=deterministic,
        )

        if HABITAT_BASELINES_AVAILABLE:
            self.pointnav_prev_actions = pointnav_action.actions.clone()
            self.pointnav_test_recurrent_hidden_states = (
                pointnav_action.rnn_hidden_states
            )
            return pointnav_action.actions
        else:
            self.pointnav_prev_actions = pointnav_action[0].clone()
            self.pointnav_test_recurrent_hidden_states = pointnav_action[1]
            return pointnav_action[0]

    def reset(self) -> None:
        """
        Resets the hidden state and previous action for the policy.
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(
            self.pointnav_test_recurrent_hidden_states
        )
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def rho_theta_from_gps_compass_goal(
    observations: "TensorDict",  # noqa: F821
    goal: np.ndarray,
    device: Union[str, torch.device] = "cuda",  # noqa: F821
) -> Tensor:
    """
    Calculates polar coordinates (rho, theta) relative to the agent's current position
    and heading towards a given goal position using GPS and compass observations given
    in Habitat format from the observations batch.

    Args:
       observations ("TensorDict"): A dictionary containing observations from the agent.
           It should include "gps" and "compass" information.
           - "gps" (Tensor): Tensor of shape (batch_size, 2) representing the
             GPS coordinates of the agent.
           - "compass" (Tensor): Tensor of shape (batch_size, 1) representing
             the compass heading of the agent in radians. It represents how many radians
             the agent must turn to the left (CCW from above) from its initial heading to
             reach its current heading.
       goal (np.ndarray): Array of shape (2,) representing the goal position.
       device (Union[str, torch.device]): The device to use for the tensor.

    Returns:
       Tensor: A tensor of shape (2,) representing the polar coordinates (rho, theta).
           - rho (float): The distance from the agent to the goal.
           - theta (float): The angle, in radians, that the agent must turn (to the
             left, CCW from above) to face the goal.
    """
    gps_numpy = observations["gps"].squeeze(1).cpu().numpy()[0]
    heading = observations["compass"].squeeze(1).cpu().numpy()[0]
    gps_numpy[1] *= -1  # Flip y-axis to match habitat's coordinate system.
    rho, theta = rho_theta(gps_numpy, heading, goal)
    rho_theta_tensor = torch.tensor([rho, theta], device=device, dtype=torch.float32)

    return rho_theta_tensor


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


def load_pointnav_policy(file_path: str) -> PointNavResNetPolicy:
    """
    Loads a PointNavResNetPolicy policy from a .pth file.

    Args:
        file_path (str): The path to the policy file.
    Returns:
        NetPolicy: The policy.
    """
    ckpt_dict = torch.load(file_path, map_location="cpu")
    pointnav_policy = _generate_untrained_policy(ckpt_dict)
    return pointnav_policy


def _generate_untrained_policy(ckpt_dict: Any) -> PointNavResNetPolicy:
    if HABITAT_BASELINES_AVAILABLE:
        obs_space = SpaceDict(
            {
                "depth": spaces.Box(
                    low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32
                ),
                "pointgoal_with_gps_compass": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        action_space = Discrete(4)
        policy = PointNavResNetPolicy.from_config(
            ckpt_dict["config"], obs_space, action_space
        )
        policy.load_state_dict(ckpt_dict["state_dict"])
        return policy

    else:
        policy = PointNavResNetPolicy()
        current_state_dict = policy.state_dict()
        policy.load_state_dict(
            {k: v for k, v in ckpt_dict.items() if k in current_state_dict}
        )
        unused_keys = [k for k in ckpt_dict.keys() if k not in current_state_dict]
        print(
            "The following unused keys were not loaded when loading the pointnav"
            f" policy: {unused_keys}"
        )
        return policy


def wrap_heading(theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Wraps given angle to be between -pi and pi.

    Args:
        theta (float): The angle in radians.
    Returns:
        float: The wrapped angle in radians.
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Load a checkpoint file for PointNavResNetPolicy")
    parser.add_argument("ckpt_path", help="path to checkpoint file")
    args = parser.parse_args()

    policy = load_pointnav_policy(args.ckpt_path)
    print("Loaded model from checkpoint successfully!")
    mask = torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.bool)
    observations = {
        "depth": torch.zeros(1, 224, 224, 1, device=torch.device("cuda")),
        "pointgoal_with_gps_compass": torch.zeros(1, 2, device=torch.device("cuda")),
    }
    policy.to(torch.device("cuda"))
    action = policy.act(
        observations,
        torch.zeros(1, 4, 512, device=torch.device("cuda"), dtype=torch.float32),
        torch.zeros(1, 1, device=torch.device("cuda"), dtype=torch.long),
        mask,
    )
    print("Forward pass successful!")
