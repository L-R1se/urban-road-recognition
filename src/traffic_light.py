"""
Traffic Light & Element Color Detection via HSV Color Space.

Uses OpenCV to recognize red, yellow, and green traffic lights by analyzing
contour area within predefined HSV thresholds.
"""

import cv2
import numpy as np

# HSV thresholds for various elements
HSV_Threshold = {
    'red': {
        'L': np.array([0, 100, 100]),
        'H': np.array([10, 255, 255])
    },
    'green': {
        'L': np.array([40, 100, 100]),
        'H': np.array([80, 255, 255])
    },
    'yellow': {
        'L': np.array([20, 100, 100]),
        'H': np.array([30, 255, 255])
    },
    'red_line': {
        'L': np.array([0, 80, 80]),
        'H': np.array([10, 255, 255])
    },
}


def element_preprocess(img, element, hsv_thresh=None):
    """
    Detect a colored element by HSV thresholding + contour analysis.

    Args:
        img: BGR input image.
        element: Key into HSV_Threshold dict (e.g. 'red', 'green').
        hsv_thresh: Optional custom HSV threshold dict.

    Returns:
        element_area: Area in pixels of the largest detected contour.
    """
    if hsv_thresh is None:
        hsv_thresh = HSV_Threshold

    blur_frame = cv2.medianBlur(img, 7)
    hsv_image = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)
    binary_img = cv2.inRange(hsv_image,
                             hsv_thresh[element]['L'],
                             hsv_thresh[element]['H'])

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
