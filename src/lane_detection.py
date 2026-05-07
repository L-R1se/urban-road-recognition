"""
Lane Line Detection Pipeline.

Uses HSV color space conversion, morphological operations, and Hough Transform
to detect left/right lane lines and compute steering error for the PID controller.
"""

import cv2
import numpy as np


def line_preprocess(origin_img, H_min, S_min, V_min, H_max, S_max, V_max):
    """HSV-based lane line extraction with erosion + dilation."""
    hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([H_min, S_min, V_min])
    higher_color = np.array([H_max, S_max, V_max])
    binary_img = cv2.inRange(hsv, lower_color, higher_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_img = cv2.erode(binary_img, kernel, iterations=1)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=2)
    return dilate_img


def line_preprocess_2(origin_img, H_min_2, S_min_2, V_min_2, H_max_2, S_max_2, V_max_2):
    """Secondary HSV extraction for special elements (red lines, crosswalks, etc.)."""
    hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([H_min_2, S_min_2, V_min_2])
    higher_color = np.array([H_max_2, S_max_2, V_max_2])
    binary_img = cv2.inRange(hsv, lower_color, higher_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode_img = cv2.erode(binary_img, kernel, iterations=1)
    dilate_img = cv2.dilate(erode_img, kernel, iterations=2)
    return dilate_img


def least_squares_fit(point_list, ymin, ymax):
    """Fit a first-degree polynomial to points using least squares."""
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    return [(xmin, ymin), (xmax, ymax)], fit


def find_points(lines):
    """Find the y-range (ymin, ymax) of a set of line segments."""
    line_y_max = 0
    line_y_min = 999
    for line in lines:
        for x1, y1, x2, y2 in line:
            line_y_max = max(line_y_max, y1, y2)
            line_y_min = min(line_y_min, y1, y2)
    return line_y_max, line_y_min


def clean_lines(lines, threshold):
    """Iteratively remove lines whose slope deviates too far from the mean."""
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def HoughLines(line_roi, rho, theta, threshold,
               min_line_length, max_line_gap, slope_min, slope_max,
               clean_right_flag, change_flag, change_flag_2):
    """
    Detect lane lines using Hough Transform and classify into left/right.

    Returns:
        left_results, right_results: endpoint pairs for each lane
        left_fit, right_fit: polynomial fit coefficients
    """
    lines = cv2.HoughLinesP(line_roi, rho, theta, threshold,
                            minLineLength=min_line_length,
                            maxLineGap=max_line_gap)
    left_results, right_results, left_fit, right_fit = [], [], [], []

    if lines is not None:
        left_lines, right_lines = [], []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if y1 != y2:
                    k = float(x1 - x2) / (y1 - y2)
                    if slope_max > abs(k) > slope_min:
                        if k > 0:
                            if (k > 0.2 and not clean_right_flag) or \
                               (k > 0.2 and not change_flag and not change_flag_2):
                                continue
                            if (x1 > x2 and x1 < 120) or (x2 > x1 and x2 < 120):
                                continue
                            right_lines.append(line)
                        elif k < 0:
                            if k > -1.4 and not change_flag and not change_flag_2:
                                continue
                            if (x1 > x2 and x2 > 120) or (x2 > x1 and x1 > 120):
                                continue
                            left_lines.append(line)
                    else:
                        continue

        if len(left_lines) > 0:
            clean_lines(left_lines, 0.1)
            left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
            left_points += [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
            ymax, ymin = find_points(left_lines)
            left_results, left_fit = least_squares_fit(left_points, ymin, ymax)

        if len(right_lines) > 0:
            clean_lines(right_lines, 0.1)
            right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
            right_points += [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
            ymax, ymin = find_points(right_lines)
            right_results, right_fit = least_squares_fit(right_points, ymin, ymax)

    return left_results, right_results, left_fit, right_fit


def how_to_turn(shape, left_results, right_results, left_fit, right_fit,
                start_cnt, road_width, EN,
                turn_flag, turn_left_flag, turn_right_flag,
                side_walk_flag, speed_limit_flag, speed_flag,
                green_light_flag, change_flag_3, change_flag_4,
                danger_flag, danger_flag_2, danger_flag_3):
    """
    Compute steering error (rho) from lane line positions.

    Rho > 0 → turn right, Rho < 0 → turn left.
    """
    height, width = shape[0], shape[1]
    middle_line = width / 2

    # Both lanes detected
    if len(left_results) > 0 and len(right_results) > 0:
        if start_cnt < 5 and EN == 1:
            start_cnt += 1
            road_width += (right_results[1][0] + right_results[0][0]) / 2 - \
                          (left_results[1][0] + left_results[0][0]) / 2
            if start_cnt == 5:
                road_width /= 5
        X = (left_fit[0] * height / 2 + left_fit[1] +
             right_fit[0] * height / 2 + right_fit[1]) / 2
        if abs(left_results[1][1] - right_results[1][1]) < 20:
            X1 = (left_results[1][0] + right_results[1][0]) / 2
            X = (X + X1) / 2
        middle_line = X
        rho = -(middle_line - width / 2)

    # Only right lane — turn left
    elif len(left_results) == 0 and len(right_results) > 0:
        if start_cnt != 5:
            road_width = 165 if (not turn_flag and side_walk_flag) else 140
        X_top = right_results[0][0]
        X_bottom = (height - right_results[0][1]) * \
                   (right_results[0][0] - right_results[1][0]) / \
                   (right_results[0][1] - right_results[1][1]) + right_results[0][0]
        X = (X_top + X_bottom) / 2
        middle_line = X - road_width / 2
        rho = width / 2 - middle_line
        rho = abs(rho)
        if not change_flag_3 and change_flag_4:
            rho = 45

    # Only left lane — turn right
    elif len(right_results) == 0 and len(left_results) > 0:
        if start_cnt != 5:
            road_width = 165 if (not turn_flag and side_walk_flag) else 140
        X_top = left_results[0][0]
        X_bottom = (height - left_results[0][1]) * \
                   (left_results[0][0] - left_results[1][0]) / \
                   (left_results[0][1] - left_results[1][1]) + left_results[0][0]
        X = (X_top + X_bottom) / 2
        middle_line = X + road_width / 2
        rho = width / 2 - middle_line
        rho = abs(rho)
        rho = -rho
    else:
        rho = 0

    # Override rho for special states
    if not turn_left_flag and turn_flag:
        rho = 100
    elif not turn_right_flag and turn_flag:
        rho = -100
    if not danger_flag and danger_flag_2:
        rho = 100
    if not danger_flag and danger_flag_3 and not danger_flag_2:
        rho = 0

    return rho, start_cnt, road_width


def draw_lines(line_img, left_results, right_results, left_k, right_k):
    """Draw detected lane lines on image for visualization."""
    if len(left_results) != 0:
        cv2.line(line_img, left_results[0], left_results[1], (0, 0, 255), 10)
        cv2.putText(line_img, f"left_k:{left_k[0]:.2f}", (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if len(right_results) != 0:
        cv2.line(line_img, right_results[0], right_results[1], (0, 255, 0), 10)
        cv2.putText(line_img, f"right_k:{right_k[0]:.2f}", (120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("line_img", line_img)


def add_diagonal_line(binary_img):
    """Add a diagonal line from first white pixel (L→R, B→T) to bottom-right."""
    height, width = binary_img.shape[:2]
    first_white = None
    for x in range(width):
        for y in range(height - 1, -1, -1):
            if binary_img[y, x] == 255:
                first_white = (x, y)
                break
        if first_white is not None:
            break
    if first_white is not None:
        bottom_right = (width - 1, height - 1)
        cv2.line(binary_img, first_white, bottom_right, 255, 2)
    return binary_img


def tuple_transform(left_results, right_results):
    """Add offset to lane line endpoints for masked region adjustment."""
    A_x, A_y = left_results[1]
    B_x, B_y = left_results[0]
    C_x, C_y = right_results[0]
    D_x, D_y = right_results[1]
    offset = 20
    return (A_x + offset, A_y), (B_x + int(offset / 2), B_y), (C_x, C_y), (D_x, D_y)
