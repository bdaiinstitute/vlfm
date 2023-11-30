# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import numpy as np


def get_agent_radius_in_px(pixels_per_meter: float, radius: float = 0.18) -> int:
    return int(np.ceil(radius * pixels_per_meter))
