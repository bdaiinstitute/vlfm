import time

import torch
from spot_wrapper.spot import Spot

from .objectnav_env import NOMINAL_ARM_POSE, ObjectNavEnv
from .policies.basic_objnav_policy import BasicObjNavSpotPolicy
from .robots.bdsw_robot import BDSWRobot


def run_env(env: ObjectNavEnv, policy: BasicObjNavSpotPolicy, goal: str):
    observations = env.reset(goal)
    done = False
    mask = torch.zeros(1, 1, device=policy.pointnav_policy.device, dtype=torch.bool)
    action = policy.act(observations, None, None, mask)
    while not done:
        observations, _, done, info = env.step(action)
        action = policy.act(observations, None, None, mask, deterministic=True)
        mask = torch.ones_like(mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g",
        "--goal",
        type=str,
        default="chair",
        help="Object to search for",
    )
    args = parser.parse_args()
    policy = BasicObjNavSpotPolicy()
    goal = args.goal

    spot = Spot("BDSW_env")  # just a name, can be anything
    with spot.get_lease():  # turns the robot on, and off upon any errors/completion
        spot.power_on()
        spot.blocking_stand()
        robot = BDSWRobot(spot)
        robot.open_gripper()
        robot.set_arm_joints(NOMINAL_ARM_POSE, travel_time=0.75)
        time.sleep(0.75)
        env = ObjectNavEnv(robot)
        run_env(env, policy, goal)
