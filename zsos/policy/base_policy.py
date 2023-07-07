import torch
from gym import spaces
from habitat import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo import Policy
from habitat_baselines.rl.ppo.policy import PolicyActionData
from omegaconf import DictConfig


@baseline_registry.register_policy
class BasePolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def should_load_agent_state(self):
        return False

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        return cls()

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        num_envs = observations["rgb"].shape[0]
        action = torch.ones(num_envs, 1, dtype=torch.long)
        return PolicyActionData(actions=action, rnn_hidden_states=rnn_hidden_states)

    # used in ppo_trainer.py eval:

    def to(self, *args, **kwargs):
        return

    def eval(self):
        return

    def parameters(self):
        yield torch.zeros(1)


if __name__ == "__main__":
    # Save a dummy state_dict using torch.save
    config = get_config(
        "habitat-lab/habitat-baselines/habitat_baselines/config/pointnav/ppo_pointnav_example.yaml"
    )
    dummy_dict = {
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    torch.save(dummy_dict, "dummy_policy.pth")
