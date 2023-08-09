import cv2
import numpy as np


def rotate_image(image: np.ndarray, radians: float) -> np.ndarray:
    """Rotate an image by the specified angle in radians.

    Args:
        image (numpy.ndarray): The input image.
        radians (float): The angle of rotation in radians.

    Returns:
        numpy.ndarray: The rotated image.
    """
    height, width = image.shape[0], image.shape[1]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def place_img_in_img(img1: np.ndarray, img2: np.ndarray, xy: tuple) -> np.ndarray:
    """Place img2 in img1 such that img2's center is at the specified coordinates (xy)
    in img1.

    Args:
        img1 (numpy.ndarray): The base image.
        img2 (numpy.ndarray): The image to be placed.
        xy (tuple): The (x, y) coordinates of the center of img2 in img1.

    Returns:
        numpy.ndarray: The updated base image with img2 placed.
    """
    x, y = xy
    x = int(x)
    y = int(y)
    x1, y1 = x - img2.shape[1] // 2, y - img2.shape[0] // 2
    x2, y2 = x1 + img2.shape[1], y1 + img2.shape[0]
    img1[y1:y2, x1:x2] = img2
    return img1


def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap(
        (normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )

    return inferno_colormap
