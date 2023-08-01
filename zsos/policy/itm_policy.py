import os
from typing import Tuple

from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensor_dict import TensorDict
from torch import Tensor

from zsos.llm.llm import BaseLLM
from zsos.mapping.frontier_map import FrontierMap
from zsos.policy.semantic_policy import SemanticPolicy
from zsos.vlm.blip2itm import BLIP2ITMClient


@baseline_registry.register_policy
class ITMPolicy(SemanticPolicy):
    llm: BaseLLM = None
    visualize: bool = True
    current_best_object: str = ""
    depth_image_shape: Tuple[int, int] = (244, 224)
    camera_height: float = 0.88
    det_conf_threshold: float = 0.5
    pointnav_stop_radius: float = 0.65

    def __init__(self, *args, **kwargs):
        super().__init__()
        # VL models
        self.itm = BLIP2ITMClient()
        self.frontier_map: FrontierMap = FrontierMap()

    def _reset(self):
        super()._reset()
        self.frontier_map.reset()

    def _explore(self, observations: TensorDict) -> Tensor:
        frontiers = observations["frontier_sensor"][0].cpu().numpy()
        rgb = observations["rgb"][0].cpu().numpy()
        text = f"Seems like there is a {self.target_object} ahead."
        self.frontier_map.update(frontiers, rgb, text)
        goal, cosine = self.frontier_map.get_best_frontier()
        os.environ["DEBUG_INFO"] = f"Best frontier: {cosine}"
        print(f"Step: {self.num_steps} Best frontier: {cosine}")
        pointnav_action = self._pointnav(
            observations, goal[:2], deterministic=True, stop=False
        )

        return pointnav_action
