# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import torch

from vlfm.run import get_config


def save_dummy_policy(filename: str) -> None:
    # Save a dummy state_dict using torch.save
    config = get_config("config/experiments/vlfm_objectnav_hm3d.yaml")
    dummy_dict = {
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    torch.save(dummy_dict, filename)


if __name__ == "__main__":
    save_dummy_policy("data/dummy_policy.pth")
    print("Dummy policy weights saved to data/dummy_policy.pth")
