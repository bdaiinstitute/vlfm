from dataclasses import dataclass

import numpy as np
from habitat import registry
from habitat.config.default_structured_configs import (
    MeasurementConfig,
)
from habitat.core.embodied_task import Measure
from hydra.core.config_store import ConfigStore


@registry.register_measure
class TraveledStairs(Measure):
    cls_uuid: str = "traveled_stairs"

    def __init__(self, sim, config, *args, **kwargs):
        self._sim = sim
        self._config = config
        self._history = []
        super().__init__(*args, **kwargs)

    @staticmethod
    def _get_uuid(*args, **kwargs):
        return TraveledStairs.cls_uuid

    def reset_metric(self, *args, episode, task, observations, **kwargs):
        self._history = []
        self.update_metric(
            *args, episode=episode, task=task, observations=observations, **kwargs
        )

    def update_metric(self, *args, episode, task, observations, **kwargs):
        curr_z = self._sim.get_agent_state().position[1]
        self._history.append(curr_z)
        # Make self._metric True (1) if peak-to-peak distance is greater than 0.9m
        self._metric = int(np.ptp(self._history) > 0.9)


@dataclass
class TraveledStairsMeasurementConfig(MeasurementConfig):
    type: str = TraveledStairs.__name__


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.traveled_stairs",
    group="habitat/task/measurements",
    name="traveled_stairs",
    node=TraveledStairsMeasurementConfig,
)
