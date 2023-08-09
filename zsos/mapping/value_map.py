import cv2
import numpy as np

from zsos.utils.geometry_utils import extract_yaw, get_rotation_matrix
from zsos.utils.img_utils import place_img_in_img, rotate_image

DEBUG = False


class ValueMap:
    """Generates a map representing how valuable explored regions of the environment
    are with respect to finding and navigating to the target object."""

    _confidence_mask: np.ndarray = None

    def __init__(self, fov: float, max_depth: float):
        size = 700
        self.fov = fov
        self.max_depth = max_depth
        self.map = np.ones((size, size)) * -1.0
        self.pixels_per_meter = 20
        self.episode_pixel_origin = np.array([size // 2, size // 2])
        self.min_confidence = 0.25

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
        curr_data = self._get_visible_mask(depth)

        # Rotate this new data to match the camera's orientation
        curr_data = rotate_image(curr_data, -extract_yaw(tf_episodic_to_camera))

        # Determine where this mask should be overlaid
        cam_x, cam_y = tf_episodic_to_camera[:2, 3] / tf_episodic_to_camera[3, 3]

        # Convert to pixel units
        px = int(cam_x * self.pixels_per_meter) + self.episode_pixel_origin[0]
        py = int(cam_y * self.pixels_per_meter) + self.episode_pixel_origin[1]

        # Overlay the new data onto the map
        blank_map = np.zeros_like(self.map)
        blank_map = place_img_in_img(blank_map, curr_data, (-py, px))
        self.map[blank_map == 1] = value

    def visualize(self) -> np.ndarray:
        """Return an image representation of the map"""
        # Must negate the y values, then rotate 90 degrees counter-clockwise
        # to get the correct orientation
        map_img = np.flipud(self.map)
        return (map_img * 255).astype(np.uint8)

    def _get_visible_mask(self, depth: np.ndarray) -> np.ndarray:
        """Using the FOV and depth, return the visible portion of the FOV.

        Args:
            depth: The depth image to use for determining the visible portion of the
                FOV.

        Returns:
            A mask of the visible portion of the FOV.
        """
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

        # Draw the contour onto the cone mask, in filled-in black
        visible_mask = cv2.drawContours(cone_mask, [contour], -1, 0, -1)

        if DEBUG:
            vis = np.zeros((visible_mask.shape[0], visible_mask.shape[1], 3))
            vis[cone_mask == 1] = (255, 255, 255)
            cv2.drawContours(vis, [contour], -1, (0, 0, 255), -1)
            for point in contour:
                vis[point[1], point[0]] = (0, 255, 0)
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
            -np.rad2deg(self.fov) / 2 + 90,  # start_angle
            np.rad2deg(self.fov) / 2 + 90,  # end_angle
            1,  # color
            -1,  # thickness
        )
        return cone_mask

    def _get_confidence_mask(self) -> np.ndarray:
        """Generate a FOV cone with central values weighted more heavily"""
        if self._confidence_mask is not None:
            return self._confidence_mask
        cone_mask = self._get_blank_cone_mask()
        adjusted_mask = np.zeros_like(cone_mask).astype(np.float32)
        for row in range(adjusted_mask.shape[0]):
            for col in range(adjusted_mask.shape[1]):
                horizontal = abs(row - adjusted_mask.shape[0] // 2)
                vertical = abs(col - adjusted_mask.shape[1] // 2)
                angle = np.arctan2(vertical, horizontal)
                angle = remap(angle, 0, self.fov / 2, 0, np.pi / 2)
                confidence = np.cos(angle) ** 2
                confidence = remap(confidence, 0, 1, self.min_confidence, 1)
                adjusted_mask[row, col] = confidence
        adjusted_mask = adjusted_mask * cone_mask
        self._confidence_mask = adjusted_mask

        return adjusted_mask


def remap(value, from_low, from_high, to_low, to_high):
    """Maps a value from one range to another.

    Parameters:
        value (float): The value to be mapped.
        from_low (float): The lower bound of the input range.
        from_high (float): The upper bound of the input range.
        to_low (float): The lower bound of the output range.
        to_high (float): The upper bound of the output range.

    Returns:
        float: The mapped value.
    """
    return (value - from_low) * (to_high - to_low) / (from_high - from_low) + to_low


if __name__ == "__main__":
    v = ValueMap(
        fov=np.deg2rad(60),
        max_depth=5.0,
    )
    depth = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = v._get_visible_mask(depth)
    cv2.imshow("img", img * 255)
    cv2.waitKey(0)

    num_points = 20

    x = [0, 10, 10, 0]
    y = [0, 0, 10, 10]
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    points = np.stack((x, y), axis=1)

    for pt, angle in zip(points, angles):
        tf = np.eye(4)
        tf[:2, 3] = pt
        tf[:2, :2] = get_rotation_matrix(angle)
        v.update_map(depth, tf, 1)
        img = v.visualize()
        cv2.imshow("img", img)
        key = cv2.waitKey(0)
        if key == ord("q"):
            break
