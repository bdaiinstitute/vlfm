# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.


class SpotCamIds:
    r"""Enumeration of types of cameras."""

    BACK_DEPTH = "back_depth"
    BACK_DEPTH_IN_VISUAL_FRAME = "back_depth_in_visual_frame"
    BACK_FISHEYE = "back_fisheye_image"
    FRONTLEFT_DEPTH = "frontleft_depth"
    FRONTLEFT_DEPTH_IN_VISUAL_FRAME = "frontleft_depth_in_visual_frame"
    FRONTLEFT_FISHEYE = "frontleft_fisheye_image"
    FRONTRIGHT_DEPTH = "frontright_depth"
    FRONTRIGHT_DEPTH_IN_VISUAL_FRAME = "frontright_depth_in_visual_frame"
    FRONTRIGHT_FISHEYE = "frontright_fisheye_image"
    HAND_COLOR = "hand_color_image"
    HAND_COLOR_IN_HAND_DEPTH_FRAME = "hand_color_in_hand_depth_frame"
    HAND_DEPTH = "hand_depth"
    HAND_DEPTH_IN_HAND_COLOR_FRAME = "hand_depth_in_hand_color_frame"
    HAND = "hand_image"
    LEFT_DEPTH = "left_depth"
    LEFT_DEPTH_IN_VISUAL_FRAME = "left_depth_in_visual_frame"
    LEFT_FISHEYE = "left_fisheye_image"
    RIGHT_DEPTH = "right_depth"
    RIGHT_DEPTH_IN_VISUAL_FRAME = "right_depth_in_visual_frame"
    RIGHT_FISHEYE = "right_fisheye_image"


# CamIds that need to be rotated by 270 degrees in order to appear upright
SHOULD_ROTATE = {
    SpotCamIds.FRONTLEFT_DEPTH,
    SpotCamIds.FRONTRIGHT_DEPTH,
    SpotCamIds.HAND_DEPTH,
    SpotCamIds.HAND,
}

# Maps camera ids to the shapes of their images
CAM_ID_TO_SHAPE = {
    SpotCamIds.BACK_DEPTH: (424, 240, 1),
    SpotCamIds.BACK_DEPTH_IN_VISUAL_FRAME: (640, 480, 1),
    SpotCamIds.BACK_FISHEYE: (640, 480, 3),
    SpotCamIds.FRONTLEFT_DEPTH: (424, 240, 1),
    SpotCamIds.FRONTLEFT_DEPTH_IN_VISUAL_FRAME: (640, 480, 1),
    SpotCamIds.FRONTLEFT_FISHEYE: (640, 480, 3),
    SpotCamIds.FRONTRIGHT_DEPTH: (424, 240, 1),
    SpotCamIds.FRONTRIGHT_DEPTH_IN_VISUAL_FRAME: (640, 480, 1),
    SpotCamIds.FRONTRIGHT_FISHEYE: (640, 480, 3),
    SpotCamIds.HAND_COLOR: (640, 480, 3),
    SpotCamIds.HAND_COLOR_IN_HAND_DEPTH_FRAME: (640, 480, 1),
    SpotCamIds.HAND_DEPTH: (224, 171, 1),
    SpotCamIds.HAND_DEPTH_IN_HAND_COLOR_FRAME: (224, 171, 1),
    SpotCamIds.HAND: (224, 171, 3),
    SpotCamIds.LEFT_DEPTH: (424, 240, 1),
    SpotCamIds.LEFT_DEPTH_IN_VISUAL_FRAME: (640, 480, 1),
    SpotCamIds.LEFT_FISHEYE: (640, 480, 3),
    SpotCamIds.RIGHT_DEPTH: (424, 240, 1),
    SpotCamIds.RIGHT_DEPTH_IN_VISUAL_FRAME: (640, 480, 1),
    SpotCamIds.RIGHT_FISHEYE: (640, 480, 3),
}
