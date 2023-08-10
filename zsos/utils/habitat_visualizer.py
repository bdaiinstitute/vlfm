from typing import Any, Dict, List

import cv2
import numpy as np
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import overlay_frame
from habitat_baselines.common.tensor_dict import TensorDict

from zsos.utils.visualization import add_text_to_image, pad_images


class HabitatVis:
    def __init__(self):
        self.rgb = []
        self.depth = []
        self.maps = []
        self.cost_maps = []
        self.texts = []
        self.last_rgb = None

    def reset(self):
        self.rgb = []
        self.depth = []
        self.maps = []
        self.cost_maps = []
        self.texts = []
        self.last_rgb = None

    def collect_data(
        self,
        observations: TensorDict,
        infos: List[Dict[str, Any]],
        policy_info: List[Dict[str, Any]],
    ):
        assert len(infos) == 1, "Only support one environment for now"

        depth = (observations["depth"][0].cpu().numpy() * 255.0).astype(np.uint8)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        depth = overlay_frame(depth, infos[0])
        self.depth.append(depth)
        self.last_rgb = observations["rgb"][0].cpu().numpy()
        self.rgb.append(policy_info[0]["visualized_detections"])
        self.maps.append(
            maps.colorize_draw_agent_and_fit_to_height(
                infos[0]["top_down_map"], self.depth[0].shape[0]
            )
        )
        self.cost_maps.append(
            maps.colorize_draw_agent_and_fit_to_height(
                infos[0]["top_down_map"], self.depth[0].shape[0]
            )
        )
        text = [
            policy_info[0][text_key]
            for text_key in policy_info[0]["render_below_images"]
            if text_key in policy_info[0]
        ]
        self.texts.append(text)

    def flush_frames(self) -> List[np.ndarray]:
        """Flush all frames and return them"""
        # Because the rgb frames are actually on step delayed, pop the first one and
        # add a black frame to the end
        self.rgb.pop(0)
        self.rgb.append(self.last_rgb)

        frames = []
        for i in range(len(self.depth)):
            frame = self._create_frame(
                self.depth[i],
                self.rgb[i],
                self.maps[i],
                self.cost_maps[i],
                self.texts[i],
            )
            frames.append(frame)

        frames = pad_images(frames, pad_from_top=True)
        self.reset()

        return frames

    @staticmethod
    def _create_frame(depth, rgb, map, cost_map, text):
        """Create a frame with depth, rgb, map, cost_map, and text"""
        row_1 = np.concatenate([depth, rgb], axis=1)
        row_2 = np.concatenate([map, cost_map], axis=1)
        # Resize row_2 to match row_1 in width
        row_1_width = row_1.shape[1]
        row_2_new_height = int(row_2.shape[0] * row_1_width / row_2.shape[1])
        row_2 = cv2.resize(row_2, (row_1_width, row_2_new_height))
        frame = np.concatenate([row_1, row_2], axis=0)

        # Add text to the top of the frame
        for t in text[::-1]:
            frame = add_text_to_image(frame, t, top=True)

        return frame
