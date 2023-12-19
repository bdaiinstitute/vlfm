# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Generator

import torch
from habitat import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo import Policy
from habitat_baselines.rl.ppo.policy import PolicyActionData


@baseline_registry.register_policy
class BasePolicy(Policy):
    """The bare minimum needed to load a policy for evaluation using ppo_trainer.py"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    @property
    def should_load_agent_state(self) -> bool:
        return False

    @classmethod
    def from_config(cls, *args: Any, **kwargs: Any) -> Any:
        return cls()

    def act(
        self,
        observations: TensorDict,
        rnn_hidden_states: torch.Tensor,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyActionData:
        # Just moves forwards
        num_envs = observations["rgb"].shape[0]
        action = torch.ones(num_envs, 1, dtype=torch.long)
        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    # used in ppo_trainer.py eval:

    def to(self, *args: Any, **kwargs: Any) -> None:
        return

    def eval(self) -> None:
        return

    def parameters(self) -> Generator:
        yield torch.zeros(1)


if __name__ == "__main__":
    # Save a dummy state_dict using torch.save. This is useful for generating a pth file
    # that can be used to load other policies that don't even read from checkpoints,
    # even though habitat requires a checkpoint to be loaded.
    config = get_config("habitat-lab/habitat-baselines/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml")
    dummy_dict = {
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    torch.save(dummy_dict, "dummy_policy.pth")
