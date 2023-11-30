# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Simulator
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R


@registry.register_measure
class GTPathVLN(Measure):
    cls_uuid: str = "gt_path_vln"

    def __init__(
        self, sim: Simulator, config: DictConfig, *args: Any, **kwargs: Any
    ) -> None:
        self._sim = sim
        self._config = config
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any) -> str:
        return GTPathVLN.cls_uuid

    def reset_metric(self, *args: Any, episode: Any, **kwargs: Any) -> None:
        self.update_metric(episode=episode)

    def update_metric(self, *args: Any, episode: Any, **kwargs: Any) -> None:
        start_rot = (R.from_quat(episode.start_rotation)).as_matrix()
        path = np.array(episode.reference_path)
        path[:, 0] -= episode.start_position[0]
        path[:, 1] -= episode.start_position[1]
        path[:, 2] -= episode.start_position[2]
        path = start_rot.T @ path.T
        path[[0, 1, 2], :] = path[[2, 0, 1], :]
        path[[0, 1], :] *= -1
        self._metric = path.T


@dataclass
class GTPathVLNMeasurementConfig(MeasurementConfig):
    type: str = GTPathVLN.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.gt_path_vln",
    group="habitat/task/measurements",
    name="gt_path_vln",
    node=GTPathVLNMeasurementConfig,
)
