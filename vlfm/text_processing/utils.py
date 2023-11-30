# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import itertools
from typing import List, Tuple

import numpy as np


def parse_instruction(instruction: str, split_strs: List[str]) -> List[str]:
    parsed_instruct = []

    working_list = [instruction]

    for ss in split_strs:
        working_list_new = []
        for instruct in working_list:
            parsed = instruct.split(ss)
            working_list_new += parsed
        working_list = working_list_new

    for instruct in working_list:
        instruct = instruct.strip()
        if instruct != "":
            parsed_instruct += [instruct]

    return parsed_instruct


def get_dist(a1: np.ndarray, a2: np.ndarray) -> float:
    return np.sqrt(np.sum(np.square(a1 - a2)))


def get_closest_vals(a1: np.ndarray, a2: np.ndarray) -> Tuple[int, int, float]:
    p = list(itertools.product(a1, a2))
    p_idx = min([i for i in range(len(p))], key=lambda t: get_dist(p[t][0], p[t][1]))
    val = get_dist(p[p_idx][0], p[p_idx][1])

    if p_idx >= a2.shape[0]:
        a1_idx = int(np.floor(p_idx / a2.shape[0]))
        a2_idx = p_idx - a1_idx * a2.shape[0]
    else:
        a1_idx = 0
        a2_idx = p_idx

    # Check we found the indexes properly
    assert val == get_dist(a1[a1_idx], a2[a2_idx])

    return a1_idx, a2_idx, val
