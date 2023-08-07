from typing import Any, Dict

import torch
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from habitat_baselines.rl.ppo.policy import PolicyActionData
from torch import Tensor

from .base_objectnav_policy import BaseObjectNavPolicy
from .itm_policy import ITMPolicy

ID_TO_NAME = ["chair", "bed", "potted plant", "toilet", "tv", "couch"]


class TorchActionIDs:
    STOP = torch.tensor([[0]], dtype=torch.long)
    MOVE_FORWARD = torch.tensor([[1]], dtype=torch.long)
    TURN_LEFT = torch.tensor([[2]], dtype=torch.long)
    TURN_RIGHT = torch.tensor([[3]], dtype=torch.long)


class HabitatMixin:
    id_to_padding: Dict[str, float] = {
        "bed": 0.3,
        "couch": 0.15,
    }
    _stop_action: Tensor = TorchActionIDs.STOP
    # ObjectMap parameters
    min_depth: float = 0.5
    max_depth: float = 5.0
    hfov: float = 79.0
    proximity_threshold: float = 1.5

    def act(
        self: BaseObjectNavPolicy,
        observations: TensorDict,
        rnn_hidden_states: Any,
        prev_actions: Any,
        masks: Tensor,
        deterministic=False,
    ) -> PolicyActionData:
        """Converts object ID to string name, returns action as PolicyActionData"""
        object_id: int = observations[ObjectGoalSensor.cls_uuid][0].item()
        obs_dict = observations.to_tree()
        obs_dict[ObjectGoalSensor.cls_uuid]: str = ID_TO_NAME[object_id]
        parent_cls: BaseObjectNavPolicy = super()  # type: ignore
        action, rnn_hidden_states = parent_cls.act(
            obs_dict, rnn_hidden_states, prev_actions, masks, deterministic
        )
        return PolicyActionData(
            actions=action,
            rnn_hidden_states=rnn_hidden_states,
            policy_info=[self.policy_info],
        )

    def _initialize(self) -> Tensor:
        """Turn left 30 degrees 12 times to get a 360 view at the beginning"""
        self.done_initializing = not self.num_steps < 11  # type: ignore
        return TorchActionIDs.TURN_LEFT


@baseline_registry.register_policy
class HabitatITMPolicy(HabitatMixin, ITMPolicy):
    pass
