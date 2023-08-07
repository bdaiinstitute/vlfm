from typing import Dict, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from zsos.policy.semantic_policy import SemanticPolicy
from zsos.policy.utils.pointnav_policy import move_obs_to_device


class BasicObjNavSpotPolicy(SemanticPolicy):
    depth_image_shape: Tuple[int, int] = (212, 240)  # height, width
    det_conf_threshold: float = 0.50
    pointnav_stop_radius: float = 1.2
    visualize: bool = True
    discrete_actions: bool = False
    # Object map parameters:
    min_depth: float = 0.0
    max_depth: float = 7.0
    hfov: float = 60.0
    proximity_threshold: float = 0.5

    initial_panning = np.deg2rad([-60, -30, 0, 30, 60, 0]).tolist()

    def act(
        self,
        observations: Dict[str, np.ndarray],
        masks: Tensor[torch.bool],
        deterministic: bool = False,
        **kwargs,
    ) -> Tensor:
        """Same as parent but ensures observations are Tensors on the right device"""
        rnn_hidden_states, prev_actions = None, None  # we don't use these
        observations = move_obs_to_device(
            observations, self.pointnav_policy.device, unsqueeze=True
        )
        return super().act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

    def _initialize(self) -> Dict[str, Union[np.ndarray, float]]:
        """Pans the gripper camera around to initialize the object map"""
        yaw = self.initial_panning.pop(0)
        self.done_initializing = len(self.initial_panning) == 0
        return {"gripper_camera_pan": np.array([yaw], dtype=np.dtype("float32"))}

    def _explore(self, obs: "TensorDict") -> Tensor:  # noqa: F821
        raise RuntimeError(
            "Not implemented for BasicObjNavSpotPolicy. This error means that the "
            "target object was not detected during initialization."
        )
