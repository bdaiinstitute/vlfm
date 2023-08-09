import cv2
import numpy as np


def rotate_image(image, radians):
    height, width = image.shape[0], image.shape[1]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, np.degrees(radians), 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    return rotated_image


def place_img_in_img(img1, img2, xy):
    """Place img2 in img1 such that img2's center is at xy in img1"""
    x, y = xy
    x = int(x)
    y = int(y)
    x1, y1 = x - img2.shape[1] // 2, y - img2.shape[0] // 2
    x2, y2 = x1 + img2.shape[1], y1 + img2.shape[0]
    img1[y1:y2, x1:x2] = img2
    return img1
