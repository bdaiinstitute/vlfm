# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List


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
