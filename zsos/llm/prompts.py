def get_textual_map_prompt(
    target: str,
    textual_map: str,
    object_options: str,
    frontier_options: str,
    curr_position: str = None,
):
    prompt = (
        "You are a robot exploring an unfamiliar home. Your task is to find a "
        f"{target}.\n"
    )

    if curr_position is not None:
        prompt += (
            f"You are currently at the following x-y coordinates: {curr_position}.\n"
        )

    prompt += (
        "This is a list of the names and locations of the objects that you have seen "
        "so far:\n\n"
        f"{textual_map}\n\n"
    )

    if object_options != "":
        prompt += (
            "Here are a list of possible objects that you can go to:\n\n"
            f"{object_options}\n\n"
            "Alternatively, you can navigate to the following frontiers to explore "
            "unexplored areas of the home:\n\n"
        )
        choice = "EITHER one object or frontier"
    else:
        prompt += (
            "You can navigate to the following frontiers to explore unexplored areas "
            "of the home:\n\n"
        )
        choice = "the frontier"

    # prompt += (
    #     f"{frontier_options}\n\n"
    #     "Carefully think about the layout of the objects and their categories, and "
    #     f"then select {choice} that represents the best location "
    #     f"to go to in order to find a {target} as soon as possible.\n"
    #     "Your response must be ONLY ONE integer (ex. '1', '23', etc.).\n"
    # )

    prompt += (
        f"{frontier_options}\n\n"
        "Carefully think about the layout of the objects and their categories, and "
        f"then select {choice} that best represents the location to go to in order to "
        f"find a {target} with the highest likelihood.\n"
        "Your response must be ONLY ONE integer (ex. '1', '23', etc.).\n"
    )

    return prompt


def unnumbered_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def numbered_list(items: list[str], start: int = 1) -> str:
    return "\n".join(f"{i+start}. {item}" for i, item in enumerate(items))
