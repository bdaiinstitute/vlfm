import cv2
import numpy as np

DEBUG = False


class ValueMap:
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    def __init__(self, fov: float, max_depth: float):
        self.fov = fov
        self.max_depth = max_depth
        self.map = np.ones((100, 100)) * -1.0
        self.pixels_per_meter = 20

    def update_map(
        self, depth: np.ndarray, tf_episodic_to_camera: np.ndarray, value: float
    ):
        """Updates the value map with the given depth image, pose, and value to use.

        Args:
            depth: The depth image to use for updating the map.
            tf_episodic_to_camera: The transformation matrix from the episodic frame to
                the camera frame.
            value: The value to use for updating the map.
        """
        # Get new portion of the map
        self._get_visible_mask(depth)

        # Determine where this mask should be overlaid
        cx, cy = tf_episodic_to_camera[:2, 3] / tf_episodic_to_camera[3, 3]
        # Convert to pixel units
        cx = int(cx * self.pixels_per_meter + self.map.shape[0] / 2)
        cy = int(cy * self.pixels_per_meter + self.map.shape[1] / 2)

    def _get_visible_mask(self, depth: np.ndarray):
        """Using the FOV and depth, return the visible portion of the FOV."""
        # Squash depth image into one row with the max depth value for each column
        depth_row = np.max(depth, axis=0) * self.max_depth

        # Create a linspace of the same length as the depth row from -fov/2 to fov/2
        angles = np.linspace(-self.fov / 2, self.fov / 2, len(depth_row))

        # Assign each value in the row with an x, y coordinate depending on 'angles'
        # and the max depth value for that column
        x = depth_row * np.cos(angles)
        y = depth_row * np.sin(angles)

        # Get blank cone mask
        cone_mask = self._get_blank_cone_mask()

        # Convert the x, y coordinates to pixel coordinates
        x = (x * self.pixels_per_meter + cone_mask.shape[0] / 2).astype(int)
        y = (y * self.pixels_per_meter + cone_mask.shape[1] / 2).astype(int)

        # Create a contour from the x, y coordinates, with the top left and right
        # corners of the image as the first two points
        last_row = cone_mask.shape[0] - 1
        last_col = cone_mask.shape[1] - 1
        start = np.array([[0, last_col]])
        end = np.array([[last_row, last_col]])
        contour = np.concatenate((start, np.stack((y, x), axis=1), end), axis=0)
        contour_cv2 = contour[:, [1, 0]]  # cv2 uses (y, x) instead of (x, y)

        # Draw the contour onto the cone mask, in filled-in black
        visible_mask = cv2.drawContours(cone_mask, [contour_cv2], -1, 0, -1)

        if DEBUG:
            vis = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
            vis[cone_mask == 1] = (255, 255, 255)
            cv2.drawContours(vis, [contour_cv2], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[0], point[1]] = (0, 255, 0)
            cv2.imshow("obstacle mask", vis)
            cv2.waitKey(0)

        return visible_mask

    def _get_blank_cone_mask(self) -> np.ndarray:
        """Generate a FOV cone without any obstacles considered"""
        size = int(self.max_depth * self.pixels_per_meter)
        cone_mask = np.zeros((size * 2 + 1, size * 2 + 1))
        cone_mask = cv2.ellipse(
            cone_mask,
            (size, size),  # center_pixel
            (size, size),  # axes lengths
            0,  # angle circle is rotated
            -np.rad2deg(self.fov) / 2,  # start_angle
            np.rad2deg(self.fov) / 2,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask
