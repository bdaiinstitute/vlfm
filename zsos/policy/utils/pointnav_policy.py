from typing import Dict, Tuple, Union

import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete
from torch import Tensor

try:
    from habitat_baselines.common.tensor_dict import TensorDict
    from habitat_baselines.rl.ddppo.policy import PointNavResNetPolicy
    from habitat_baselines.rl.ppo.policy import PolicyActionData

    class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):
        def act(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
            policy_actions: "PolicyActionData" = super().act(*args, **kwargs)
            return policy_actions.actions, policy_actions.rnn_hidden_states

    HABITAT_BASELINES_AVAILABLE = True
except ModuleNotFoundError:
    from zsos.policy.utils.non_habitat_policy.nh_pointnav_policy import (
        PointNavResNetPolicy,
    )

    class PointNavResNetTensorOutputPolicy(PointNavResNetPolicy):
        """Already outputs a tensor, so no need to convert."""

        pass

    HABITAT_BASELINES_AVAILABLE = False


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
        observations: Union["TensorDict", Dict],
        masks: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """Infers action to take towards the given (rho, theta) based on depth vision.

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
            Tensor: A tensor denoting the action to take.
        """
        # Convert numpy arrays to torch tensors for each dict value
        observations = move_obs_to_device(observations, self.device)
        pointnav_action, rnn_hidden_states = self.policy.act(
            observations,
            self.pointnav_test_recurrent_hidden_states,
            self.pointnav_prev_actions,
            masks,
            deterministic=deterministic,
        )
        self.pointnav_prev_actions = pointnav_action.clone()
        self.pointnav_test_recurrent_hidden_states = rnn_hidden_states
        return pointnav_action

    def reset(self) -> None:
        """
        Resets the hidden state and previous action for the policy.
        """
        self.pointnav_test_recurrent_hidden_states = torch.zeros_like(
            self.pointnav_test_recurrent_hidden_states
        )
        self.pointnav_prev_actions = torch.zeros_like(self.pointnav_prev_actions)


def load_pointnav_policy(file_path: str) -> PointNavResNetTensorOutputPolicy:
    """Loads a PointNavResNetPolicy policy from a .pth file.

    Args:
        file_path (str): The path to the trained weights of the pointnav policy.
    Returns:
        PointNavResNetTensorOutputPolicy: The policy.
    """
    ckpt_dict = torch.load(file_path, map_location="cpu")

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
        pointnav_policy = PointNavResNetTensorOutputPolicy.from_config(
            ckpt_dict["config"], obs_space, action_space
        )
        pointnav_policy.load_state_dict(ckpt_dict["state_dict"])
        return pointnav_policy

    else:
        pointnav_policy = PointNavResNetTensorOutputPolicy()
        current_state_dict = pointnav_policy.state_dict()
        pointnav_policy.load_state_dict(
            {k: v for k, v in ckpt_dict.items() if k in current_state_dict}
        )
        unused_keys = [k for k in ckpt_dict.keys() if k not in current_state_dict]
        print(
            "The following unused keys were not loaded when loading the pointnav"
            f" policy: {unused_keys}"
        )
        return pointnav_policy


def move_obs_to_device(
    observations: Dict[str, Union[Tensor, np.ndarray]],
    device: torch.device,
    unsqueeze: bool = False,
) -> Dict[str, Tensor]:
    """Moves observations to the given device, converts numpy arrays to torch tensors.

    Args:
        observations (Dict[str, Union[Tensor, np.ndarray]]): The observations.
        device (torch.device): The device to move the observations to.
        unsqueeze (bool): Whether to unsqueeze the tensors or not.
    Returns:
        Dict[str, Tensor]: The observations on the given device as torch tensors.
    """
    # Convert numpy arrays to torch tensors for each dict value
    for k, v in observations.items():
        if isinstance(v, np.ndarray):
            tensor_dtype = torch.uint8 if v.dtype == np.uint8 else torch.float32
            observations[k] = torch.from_numpy(v).to(device=device, dtype=tensor_dtype)
            if unsqueeze:
                observations[k] = observations[k].unsqueeze(0)

    return observations


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
