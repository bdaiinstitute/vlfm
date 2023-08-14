from typing import Tuple, Union

import cv2
import numpy as np


def rotate_image(image: np.ndarray, radians: float, border_value=0) -> np.ndarray:
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
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (width, height), borderValue=border_value
    )

    return rotated_image


def place_img_in_img(
    img1: np.ndarray, img2: np.ndarray, row: int, col: int
) -> np.ndarray:
    """Place img2 in img1 such that img2's center is at the specified coordinates (xy)
    in img1.

    Args:
        img1 (numpy.ndarray): The base image.
        img2 (numpy.ndarray): The image to be placed.


    Returns:
        numpy.ndarray: The updated base image with img2 placed.
    """
    assert (
        0 <= row < img1.shape[0] and 0 <= col < img1.shape[1]
    ), "Pixel location is outside the image."
    top = row - img2.shape[0] // 2
    left = col - img2.shape[1] // 2
    bottom = top + img2.shape[0]
    right = left + img2.shape[1]

    img1_top = max(0, top)
    img1_left = max(0, left)
    img1_bottom = min(img1.shape[0], bottom)
    img1_right = min(img1.shape[1], right)

    img2_top = max(0, -top)
    img2_left = max(0, -left)
    img2_bottom = img2_top + (img1_bottom - img1_top)
    img2_right = img2_left + (img1_right - img1_left)

    img1[img1_top:img1_bottom, img1_left:img1_right] = img2[
        img2_top:img2_bottom, img2_left:img2_right
    ]

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
    peak_to_peak = np.max(image) - np.min(image)
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - np.min(image)) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap(
        (normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )

    return inferno_colormap


def resize_images(images, match_dimension="height"):
    """
    Resize images to match either their heights or their widths.

    Args:
        images (List[np.ndarray]): List of NumPy images.
        match_dimension (str): Specify 'height' to match heights, or 'width' to match
            widths.

    Returns:
        List[np.ndarray]: List of resized images.
    """
    if len(images) < 2:
        raise ValueError("At least two images are required for resizing.")

    if match_dimension == "height":
        max_height = max(img.shape[0] for img in images)
        resized_images = [
            cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height))
            for img in images
        ]
    elif match_dimension == "width":
        max_width = max(img.shape[1] for img in images)
        resized_images = [
            cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1])))
            for img in images
        ]
    else:
        raise ValueError("Invalid 'match_dimension' argument. Use 'height' or 'width'.")

    return resized_images


def crop_white_border(image: np.ndarray) -> np.ndarray:
    """Crop the image to the bounding box of non-white pixels.

    Args:
        image (np.ndarray): The input image (BGR format).

    Returns:
        np.ndarray: The cropped image. If the image is entirely white, the original
            image is returned.
    """
    # Convert the image to grayscale for easier processing
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the bounding box of non-white pixels
    non_white_pixels = np.argwhere(gray_image != 255)

    if len(non_white_pixels) == 0:
        return image  # Return the original image if it's entirely white

    min_row, min_col = np.min(non_white_pixels, axis=0)
    max_row, max_col = np.max(non_white_pixels, axis=0)

    # Crop the image to the bounding box
    cropped_image = image[min_row : max_row + 1, min_col : max_col + 1, :]

    return cropped_image


def pad_to_square(
    img: np.ndarray,
    padding_color: Tuple[int, int, int] = (255, 255, 255),
    extra_pad: int = 0,
) -> np.ndarray:
    """
    Pad an image to make it square by adding padding to the left and right sides
    if its height is larger than its width, or adding padding to the top and bottom
    if its width is larger.

    Args:
        img (numpy.ndarray): The input image.
        padding_color (Tuple[int, int, int], optional): The padding color in (R, G, B)
            format. Defaults to (255, 255, 255).

    Returns:
        numpy.ndarray: The squared and padded image.
    """
    height, width, _ = img.shape
    larger_side = max(height, width)
    square_size = larger_side + extra_pad
    padded_img = np.ones((square_size, square_size, 3), dtype=np.uint8) * np.array(
        padding_color, dtype=np.uint8
    )
    padded_img = place_img_in_img(padded_img, img, square_size // 2, square_size // 2)

    return padded_img


def pad_larger_dim(image: np.ndarray, target_dimension: int) -> np.ndarray:
    """Pads an image to the specified target dimension by adding whitespace borders.

    Args:
        image (np.ndarray): The input image as a NumPy array with shape (height, width,
            channels).
        target_dimension (int): The desired target dimension for the larger dimension
            (height or width).

    Returns:
        np.ndarray: The padded image as a NumPy array with shape (new_height, new_width,
            channels).
    """
    height, width, _ = image.shape
    larger_dimension = max(height, width)

    if larger_dimension < target_dimension:
        pad_amount = target_dimension - larger_dimension
        first_pad_amount = pad_amount // 2
        second_pad_amount = pad_amount - first_pad_amount

        if height > width:
            top_pad = np.ones((first_pad_amount, width, 3), dtype=np.uint8) * 255
            bottom_pad = np.ones((second_pad_amount, width, 3), dtype=np.uint8) * 255
            padded_image = np.vstack((top_pad, image, bottom_pad))
        else:
            left_pad = np.ones((height, first_pad_amount, 3), dtype=np.uint8) * 255
            right_pad = np.ones((height, second_pad_amount, 3), dtype=np.uint8) * 255
            padded_image = np.hstack((left_pad, image, right_pad))
    else:
        padded_image = image

    return padded_image


def pixel_value_within_radius(
    image: np.ndarray,
    pixel_location: Tuple[int, int],
    radius: int,
    reduction: str = "median",
) -> Union[float, int]:
    """Returns the maximum pixel value within a given radius of a specified pixel
    location in the given image.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.
        pixel_location (Tuple[int, int]): The location of the pixel as a tuple (row,
            column).
        radius (int): The radius within which to find the maximum pixel value.
        reduction (str, optional): The method to use to reduce the cropped image to a
            single value. Defaults to "median".

    Returns:
        Union[float, int]: The maximum pixel value within the given radius of the pixel
            location.
    """
    # Ensure that the pixel location is within the image
    assert (
        0 <= pixel_location[0] < image.shape[0]
        and 0 <= pixel_location[1] < image.shape[1]
    ), "Pixel location is outside the image."

    top_left_x = max(0, pixel_location[0] - radius)
    top_left_y = max(0, pixel_location[1] - radius)
    bottom_right_x = min(image.shape[0], pixel_location[0] + radius + 1)
    bottom_right_y = min(image.shape[1], pixel_location[1] + radius + 1)
    cropped_image = image[top_left_x:bottom_right_x, top_left_y:bottom_right_y]

    # Draw a circular mask for the cropped image
    circle_mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    circle_mask = cv2.circle(
        circle_mask,
        (radius, radius),
        radius,
        color=255,
        thickness=-1,
    )
    if reduction == "mean":
        return np.mean(cropped_image[circle_mask > 0])  # type: ignore
    elif reduction == "max":
        return np.max(cropped_image[circle_mask > 0])
    elif reduction == "median":
        return np.median(cropped_image[circle_mask > 0])  # type: ignore
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")
