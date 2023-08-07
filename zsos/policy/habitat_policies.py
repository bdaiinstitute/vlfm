from typing import Any

import numpy as np
import torch
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from torch import Tensor

from .itm_policy import ITMPolicy
from .semantic_policy import SemanticPolicy

ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    def act(
        self: SemanticPolicy,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic=False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        obs_dict[ObjectGoalSensor.cls_uuid]: str = ID_TO_NAME[object_id]  # type: ignore
        action = super().act(  # type: ignore
            obs_dict, rnn_hidden_states, prev_actions, masks, deterministic
        )
        return PolicyActionData(actions=action, rnn_hidden_states=torch.zeros(1))

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self.done_initializing = not self.num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class HabitatPolicy(HabitatMixin, ITMPolicy):
    pass


@baseline_registry.register_policy
class FBEPolicy(HabitatMixin, SemanticPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        curr_pos = observations["gps"][0].cpu().numpy() * np.array([1, -1])
        if np.linalg.norm(self.last_goal - curr_pos) < 0.25:
            frontiers = observations["frontier_sensor"][0].cpu().numpy()
            goal = frontiers[0]
        else:
            goal = self.last_goal

        pointnav_action = self._pointnav(
            observations, goal[:2], deterministic=True, stop=False
        )

        return pointnav_action
