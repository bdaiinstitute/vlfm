# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os

from habitat_baselines.common.baseline_registry import baseline_registry  # noqa

from vlfm.utils.generate_dummy_policy import save_dummy_policy


def test_load_and_save_config() -> None:
    if not os.path.exists("data"):
        os.makedirs("data")

    filename = "data/dummy_policy.pth"
    save_dummy_policy(filename)

    # Get the file size of the output PDF
    file_size = os.path.getsize(filename)

    # Check the size is greater than 30 KB
    assert file_size > 30 * 1024, "Test failed - failed to create pth"
