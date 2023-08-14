from typing import Any, Set

import numpy as np


class StateAction:
    def __init__(self, position: np.ndarray, action: Any):
        self.position = position
        self.action = action

    def __eq__(self, other: "StateAction") -> bool:
        return self.__hash__() == other.__hash__()

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.action}"
        return hash(string_repr)


class AcyclicEnforcer:
    history: Set[StateAction] = set()

    def check_cyclic(self, position: np.ndarray, action: Any) -> bool:
        state_action = StateAction(position, action)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, action: Any):
        state_action = StateAction(position, action)
        self.history.add(state_action)
