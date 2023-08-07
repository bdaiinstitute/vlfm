from frontier_exploration.base_explorer import BaseExplorer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from torch import Tensor

from zsos.policy.base_objectnav_policy import BaseObjectNavPolicy


@baseline_registry.register_policy
class OracleFBEPolicy(BaseObjectNavPolicy):
    def _explore(self, observations: TensorDict) -> Tensor:
        pointnav_action = observations[BaseExplorer.cls_uuid]
        return pointnav_action
