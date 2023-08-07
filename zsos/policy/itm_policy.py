import os
from typing import Dict, Union

from torch import Tensor

from zsos.mapping.frontier_map import FrontierMap
from zsos.policy.base_objectnav_policy import BaseObjectNavPolicy
from zsos.vlm.blip2itm import BLIP2ITMClient

try:
    from habitat_baselines.common.tensor_dict import TensorDict
except ModuleNotFoundError:
    pass


class ITMPolicy(BaseObjectNavPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.itm = BLIP2ITMClient()
        self.frontier_map: FrontierMap = FrontierMap()

    def _reset(self):
        super()._reset()
        self.frontier_map.reset()

    def _explore(self, observations: Union[Dict[str, Tensor], "TensorDict"]) -> Tensor:
        frontiers = observations["frontier_sensor"][0].cpu().numpy()
        rgb = observations["rgb"][0].cpu().numpy()
        text = f"Seems like there is a {self.target_object} ahead."
        self.frontier_map.update(frontiers, rgb, text)
        goal, cosine = self.frontier_map.get_best_frontier()
        os.environ["DEBUG_INFO"] = f"Best frontier: {cosine:.3f}"
        print(f"Step: {self.num_steps} Best frontier: {cosine}")
        pointnav_action = self._pointnav(
            observations, goal[:2], deterministic=True, stop=False
        )

        return pointnav_action
