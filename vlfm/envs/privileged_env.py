from typing import Optional, Tuple

import numpy as np
from habitat import registry
from habitat.core.dataset import Dataset
from habitat.core.environments import GymHabitatEnv
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation as R
from torch import Tensor, tensor


@registry.register_env(name="PrivEnv")
class PrivEnv(GymHabitatEnv):
    def __init__(
        self,
        config: "DictConfig",
        dataset: Optional[Dataset] = None,
        goal_radius: float = 0.4,
    ) -> None:
        super().__init__(config, dataset)
        self.shortest_path_follower = ShortestPathFollower(
            sim=self.env.env.habitat_env.sim,
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def get_gt_path_action(self, goal: np.ndarray) -> Tensor:
        episode = self.env.env.habitat_env.current_episode
        start_rot = (R.from_quat(episode.start_rotation)).as_matrix()
        start_pos = episode.start_position

        goal_w = np.array([-goal[1], start_pos[1], -goal[0]])
        goal_w = start_rot @ goal_w.T
        goal_w[0] += episode.start_position[0]
        goal_w[2] += episode.start_position[2]

        best_action = self.shortest_path_follower.get_next_action(goal_w.T)

        return tensor([[best_action]])

    def get_gt_path(self) -> Tuple[np.ndarray, np.ndarray]:
        episode = self.env.env.habitat_env.current_episode

        start_rot = (R.from_quat(episode.start_rotation)).as_matrix()
        path = np.array(episode.reference_path)
        path[:, 0] -= episode.start_position[0]
        path[:, 1] -= episode.start_position[1]
        path[:, 2] -= episode.start_position[2]
        path = start_rot.T @ path.T
        path[[0, 1, 2], :] = path[[2, 0, 1], :]
        path[[0, 1], :] *= -1

        path_wc = np.array(episode.reference_path)
        path_wc[:, [0, 1, 2]] = path_wc[:, [2, 0, 1]]

        return path.T, path_wc
