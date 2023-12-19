# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np
import torch
from spot_wrapper.spot import Spot

from vlfm.policy.utils.pointnav_policy import WrappedPointNavResNetPolicy
from vlfm.reality.pointnav_env import PointNavEnv
from vlfm.reality.robots.bdsw_robot import BDSWRobot


def run_env(env: PointNavEnv, policy: WrappedPointNavResNetPolicy, goal: np.ndarray) -> None:
    observations = env.reset(goal)
    done = False
    mask = torch.zeros(1, 1, device=policy.device, dtype=torch.bool)
    action = policy.act(observations, mask)
    action_dict = {"rho_theta": action}
    while not done:
        observations, _, done, info = env.step(action_dict)
        action = policy.act(observations, mask, deterministic=True)
        mask = torch.ones_like(mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pointnav_ckpt_path",
        type=str,
        default="pointnav_resnet_18.pth",
        help="Path to the pointnav model checkpoint",
    )
    parser.add_argument(
        "-g",
        "--goal",
        type=str,
        default="3.5,0.0",
        help="Goal location in the form x,y",
    )
    args = parser.parse_args()
    pointnav_ckpt_path = args.pointnav_ckpt_path
    policy = WrappedPointNavResNetPolicy(pointnav_ckpt_path)
    goal = np.array([float(x) for x in args.goal.split(",")])

    spot = Spot("BDSW_env")  # just a name, can be anything
    with spot.get_lease():  # turns the robot on, and off upon any errors or completion
        spot.power_on()
        spot.blocking_stand()
        robot = BDSWRobot(spot)
        env = PointNavEnv(robot)
        run_env(env, policy, goal)
