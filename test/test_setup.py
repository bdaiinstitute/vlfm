import os

import torch
from habitat import get_config


def test_habitat():
    # Save a dummy state_dict using torch.save
    config = get_config("config/experiments/ppo_pointnav_example.yaml")
    dummy_dict = {
        "config": config,
        "extra_state": {"step": 0},
        "state_dict": {},
    }

    filename = "build/dummy_policy.pth"
    torch.save(dummy_dict, filename)

    # Get the file size of the output PDF
    file_size = os.path.getsize(filename)

    # Check the size is greater than 30 KB
    assert file_size > 30 * 1024, "Test failed - failed to create pth"
