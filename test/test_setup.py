import os

import torch
from habitat_baselines.common.baseline_registry import baseline_registry  # noqa

from zsos.run import get_config


def test_load_and_save_config():
    if not os.path.exists("build"):
        os.makedirs("build")

    # Save a dummy state_dict using torch.save
    config = get_config("config/experiments/llm_objectnav_hm3d.yaml")
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
