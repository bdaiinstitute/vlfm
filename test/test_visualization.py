import os

import cv2

from vlfm.utils.visualization import generate_text_image


def test_visualization():
    if not os.path.exists("build"):
        os.makedirs("build")

    width = 400
    text = (
        "This is a long text that needs to be drawn on an image with a specified "
        "width. The text should wrap around if it exceeds the given width."
    )

    result_image = generate_text_image(width, text)

    # Save the image to a file
    output_filename = "build/output_image.png"
    cv2.imwrite(output_filename, result_image)

    # Assert that the file exists
    assert os.path.exists(output_filename), "Output image file not found!"
