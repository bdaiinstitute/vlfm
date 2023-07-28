from typing import Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
from torch import Tensor


class WrappedPointNavResNetPolicy:
    """
    Wrapper for the PointNavResNetPolicy that allows for easier usage, however it can
    only handle one environment at a time. Automatically updates the hidden state
    and previous action for the policy.
    """

    def __init__(self, ckpt_path: str):
        self.policy = load_pointnav_policy(ckpt_path)
        self.policy.to(torch.device("cuda"))
        self.pointnav_test_recurrent_hidden_states = torch.zeros(
            1,  # The number of environments.
            self.policy.net.num_recurrent_layers,
            512,  # hidden state size
            device=torch.device("cuda"),
        )
        self.pointnav_prev_actions = torch.zeros(
            1,  # The number of environments.
            1,  # The number of actions.
            device=torch.device("cuda"),
            dtype=torch.long,
        )

    def get_actions(
        self,
        observations: Union[TensorDict, Dict],
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Determines the best action to take towards the given (rho, theta) based on
        depth vision.

        Args:
            observations (Union[TensorDict, Dict]): A dictionary containing (at least)
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
        pointnav_action = self.policy.act(
            observations,
            self.pointnav_test_recurrent_hidden_states,
            self.pointnav_prev_actions,
            masks,
            deterministic=deterministic,
        )

        self.pointnav_test_recurrent_hidden_states = pointnav_action.rnn_hidden_states
        self.pointnav_prev_actions = pointnav_action.actions.clone()

        return pointnav_action.actions

    def reset(self) -> None:
        """
        Resets the hidden state and previous action for the policy.
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(
            self.pointnav_test_recurrent_hidden_states
        )
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def rho_theta_from_gps_compass_goal(
    observations: TensorDict, goal: np.ndarray
) -> Tensor:
    """
    Calculates polar coordinates (rho, theta) relative to the agent's current position
    and heading towards a given goal position using GPS and compass observations given
    in Habitat format from the observations batch.

    Args:
       observations (TensorDict): A dictionary containing observations from the agent.
           It should include "gps" and "compass" information.
           - "gps" (Tensor): Tensor of shape (batch_size, 2) representing the
             GPS coordinates of the agent.
           - "compass" (Tensor): Tensor of shape (batch_size, 1) representing
             the compass heading of the agent in radians. It represents how many radians
             the agent must turn to the left (CCW from above) from its initial heading to
             reach its current heading.
       goal (np.ndarray): Array of shape (2,) representing the goal position.

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
    rho_theta_tensor = torch.tensor(
        [rho, theta], device=torch.device("cuda"), dtype=torch.float32
    )

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
    obs_space = SpaceDict(
        {
            "depth": spaces.Box(
                low=0.0, high=1.0, shape=(224, 224, 1), dtype=np.float32
            ),
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid: spaces.Box(
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
    print(policy)
