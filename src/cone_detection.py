"""
Cone Detection via HSV Color Segmentation + Contour Analysis.

Detects red/orange traffic cones for obstacle avoidance. Uses color-based
segmentation followed by geometric contour filtering.
"""

import cv2
import numpy as np


def cone_detect(img, element, hsv_threshold):
    """
    Detect cone objects in image using HSV thresholding.

    Args:
        img: BGR input image.
        element: Key into hsv_threshold dict (e.g. 'cone').
        hsv_threshold: Dict of HSV lower/upper bounds per element type.

    Returns:
        element_area: Pixel area of the largest cone contour found.
    """
    blur_frame = cv2.medianBlur(img, 7)
    hsv_image = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    binary_img = cv2.inRange(hsv_image,
                             hsv_threshold[element]['L'],
                             hsv_threshold[element]['H'])

    erode_kernel = (3, 3)
    dilate_kernel = (7, 7)
    erode_img = cv2.erode(binary_img, erode_kernel)
    dilate_img = cv2.dilate(erode_img, dilate_kernel)

    contours, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    element_area = 0
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        contours_img = cv2.drawContours(img, c, -1, (0, 255, 0), 3)
        element_area = cv2.contourArea(c)
        cv2.imshow("yolo", contours_img)

    return element_area


def get_cone_position(img, hsv_threshold_cone):
    """
    Get the (x, y) center of the detected cone for obstacle avoidance planning.

    Args:
        img: BGR input image.
        hsv_threshold_cone: HSV bounds for cone color.

    Returns:
        (center_x, center_y) or None if no cone detected.
    """
    blur_frame = cv2.medianBlur(img, 7)
    hsv_image = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    binary_img = cv2.inRange(hsv_image,
                             hsv_threshold_cone['L'],
                             hsv_threshold_cone['H'])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_img = cv2.erode(binary_img, kernel)
    dilate_img = cv2.dilate(erode_img, kernel)

    contours, _ = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return (cx, cy)
    return None
