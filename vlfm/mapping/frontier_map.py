# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import List, Tuple

import numpy as np

from vlfm.vlm.blip2itm import BLIP2ITMClient


class Frontier:
    def __init__(self, xyz: np.ndarray, cosine: float):
        self.xyz = xyz
        self.cosine = cosine


class FrontierMap:
    frontiers: List[Frontier] = []

    def __init__(self, encoding_type: str = "cosine"):
        self.encoder: BLIP2ITMClient = BLIP2ITMClient()

    def reset(self) -> None:
        self.frontiers = []

    def update(self, frontier_locations: List[np.ndarray], curr_image: np.ndarray, text: str) -> None:
        """
        Takes in a list of frontier coordinates and the current image observation from
        the robot. Any stored frontiers that are not present in the given list are
        removed. Any frontiers in the given list that are not already stored are added.
        When these frontiers are added, their cosine field is set to the encoding
        of the given image. The image will only be encoded if a new frontier is added.

        Args:
            frontier_locations (List[np.ndarray]): A list of frontier coordinates.
            curr_image (np.ndarray): The current image observation from the robot.
            text (str): The text to compare the image to.
        """
        # Remove any frontiers that are not in the given list. Use np.array_equal.
        self.frontiers = [
            frontier
            for frontier in self.frontiers
            if any(np.array_equal(frontier.xyz, location) for location in frontier_locations)
        ]

        # Add any frontiers that are not already stored. Set their image field to the
        # given image.
        cosine = None
        for location in frontier_locations:
            if not any(np.array_equal(frontier.xyz, location) for frontier in self.frontiers):
                if cosine is None:
                    cosine = self._encode(curr_image, text)
                self.frontiers.append(Frontier(location, cosine))

    def _encode(self, image: np.ndarray, text: str) -> float:
        """
        Encodes the given image using the encoding type specified in the constructor.

        Args:
            image (np.ndarray): The image to encode.

        Returns:

        """
        return self.encoder.cosine(image, text)

    def sort_waypoints(self) -> Tuple[np.ndarray, List[float]]:
        """
        Returns the frontier with the highest cosine and the value of that cosine.
        """
        # Use np.argsort to get the indices of the sorted cosines
        cosines = [f.cosine for f in self.frontiers]
        waypoints = [f.xyz for f in self.frontiers]
        sorted_inds = np.argsort([-c for c in cosines])  # sort in descending order
        sorted_values = [cosines[i] for i in sorted_inds]
        sorted_frontiers = np.array([waypoints[i] for i in sorted_inds])

        return sorted_frontiers, sorted_values
