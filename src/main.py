"""
Main ROS Node — Urban Road Recognition Smart Car.

Entry point that orchestrates:
1. Camera image capture (USB camera @ 320×240)
2. Lane line detection via OpenCV
3. YOLOv5 object detection via TensorRT
4. Traffic light color recognition
5. Steering & speed control via ROS → serial → STM32

Usage:
    rosrun opencv_detection main.py
"""

import time
import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist

from lane_detection import (
    line_preprocess, line_preprocess_2, HoughLines,
    how_to_turn, draw_lines
)
from object_detection import YoLov5TRT, get_max_area, draw_rerectangle
from traffic_light import element_preprocess
from servo_control import Servo_PD
from speed_control import speed_control, switch_callback, EN

# ---- Categories for YOLOv5 (7 classes) ----
CATEGORIES = [
    "danger",       # 0 — Cone / obstacle
    "side_walk",    # 1 — Crosswalk
    "speed",        # 2 — Speed limit lifted
    "speed_limit",  # 3 — Speed limit
    "turn_left",    # 4 — Left turn
    "turn_right",   # 5 — Right turn
    "lane_change",  # 6 — Lane change
]

# ---- Paths (adjust for your Jetson Nano setup) ----
PLUGIN_LIBRARY = "/home/jetson/Desktop/ai_control-FULL-3.0/src/opencv_detection/scripts/shuju/libmyplugins.so"
ENGINE_FILE_PATH = "/home/jetson/Desktop/ai_control-FULL-3.0/src/opencv_detection/scripts/shuju/best.engine"

# ---- HSV thresholds for lane lines (tuned per track) ----
H_MIN, S_MIN, V_MIN = 0, 0, 150
H_MAX, S_MAX, V_MAX = 180, 50, 255
# Secondary thresholds for red/right-side lines
H_MIN_2, S_MIN_2, V_MIN_2 = 0, 80, 80
H_MAX_2, S_MAX_2, V_MAX_2 = 10, 255, 255


def main():
    rospy.init_node("opencv_new_main")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
    twist = Twist()

    # Initialize YOLOv5 TensorRT engine
    yolo = YoLov5TRT(ENGINE_FILE_PATH)

    # PD controller for steering
    servo = Servo_PD(servo_p=0.013, servo_d=0.065)

    # Camera capture
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    rospy.loginfo("Urban Road Recognition system started.")

    start_time = time.time()

    # ---- Per-frame state ----
    road_width = 0
    start_cnt = 0
    line_flag = True
    speed_x = 0.35

    while not rospy.is_shutdown():
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)

        # ---- FPS calculation ----
        end_time = start_time
        start_time = time.time()
        fps = 1.0 / (start_time - end_time) if (start_time - end_time) > 0 else 0

        # ---- Dynamic ROI mask ----
        from speed_control import (
            clean_right_flag, turn_flag, side_walk_flag,
            turn_left_flag, turn_flag_2, turn_right_flag,
            change_flag
        )
        if not clean_right_flag:
            mask_points = np.array([[(0, 70), (118, 0), (240, 0), (240, 70)]])
        elif not turn_flag and side_walk_flag:
            if not turn_left_flag and not turn_flag_2:
                mask_points = np.array([[(0, 70), (0, 0), (130, 0), (130, 70)]])
            elif not turn_right_flag and not turn_flag_2:
                mask_points = np.array([[(110, 70), (110, 0), (240, 0), (240, 70)]])
            elif not turn_left_flag and turn_flag_2:
                mask_points = np.array([[(0, 70), (0, 60), (120, 0), (240, 0), (240, 70)]])
            elif not turn_right_flag and turn_flag_2:
                mask_points = np.array([[(0, 70), (0, 0), (120, 0), (240, 60), (240, 70)]])
            else:
                mask_points = np.array([[(0, 70), (0, 50), (50, 0), (190, 0), (240, 50), (240, 70)]])
        else:
            mask_points = np.array([[(0, 70), (0, 50), (50, 0), (190, 0), (240, 50), (240, 70)]])
        mask = np.zeros((70, 240), dtype=np.uint8)
        cv2.fillPoly(mask, mask_points, 255)

        # ---- Lane line detection ----
        if line_flag:
            line_img = cv2.resize(img, (240, 180), interpolation=cv2.INTER_AREA)

            # Hough params vary by state
            if not change_flag and change_flag_4:
                min_line_length, max_line_gap = 18, 30
            elif turn_flag:
                min_line_length, max_line_gap = 13, 20
            else:
                min_line_length, max_line_gap = 30, 20

            line_img = line_img[90:160, :, :]  # ROI crop
            line_roi = line_preprocess(line_img, H_MIN, S_MIN, V_MIN,
                                       H_MAX, S_MAX, V_MAX)

            if not change_flag:  # Merge red line detection during lane change
                from speed_control import change_flag
                line_roi_2 = line_preprocess_2(line_img,
                                               H_MIN_2, S_MIN_2, V_MIN_2,
                                               H_MAX_2, S_MAX_2, V_MAX_2)
                line_roi = cv2.bitwise_or(line_roi, line_roi_2)

            # Slope limits vary by state
            if (not turn_flag_2 and side_walk_flag) or not change_flag:
                slope_max = 5.0
            else:
                slope_max = 3.0
            slope_min = 0.1

            line_roi = cv2.bitwise_and(line_roi, mask)

            left_results, right_results, left_fit, right_fit = HoughLines(
                line_roi, rho=1, theta=np.pi / 180, threshold=20,
                min_line_length=min_line_length, max_line_gap=max_line_gap,
                slope_min=slope_min, slope_max=slope_max,
                clean_right_flag=clean_right_flag, change_flag=change_flag,
                change_flag_2=change_flag_2
            )

            error, start_cnt, road_width = how_to_turn(
                line_img.shape, left_results, right_results, left_fit, right_fit,
                start_cnt, road_width, EN,
                turn_flag, turn_left_flag, turn_right_flag,
                side_walk_flag, True, True,  # speed_limit_flag, speed_flag
                True, False, False,  # green_light_flag, change_flag_3, change_flag_4
                True, True, True  # danger_flag, danger_flag_2, danger_flag_3
            )

            # Adaptive PD gains
            if not change_flag and not change_flag_2:
                Servo_Kp, Servo_Turn_Kp, Kd = 0.014, 0.021, 0.065
            else:
                Servo_Kp, Servo_Turn_Kp, Kd = 0.013, 0.016, 0.065

            if abs(error) > 60:
                Kp = Servo_Turn_Kp
            elif 60 >= abs(error) > 30:
                Kp = (Servo_Turn_Kp + Servo_Kp) * 0.5
            else:
                Kp = Servo_Kp

            from speed_control import speed as current_speed
            if current_speed == 0.0:
                error = 0

            output = servo.calc_servo_pd(error, Kp, Kd)
            twist.angular.z = output

            draw_lines(line_img, left_results, right_results, left_fit, right_fit)

        # ---- YOLOv5 object detection ----
        cls_id, area = 5, 0
        from speed_control import yolo_flag, traffic_flag, traffic_flag_2

        if yolo_flag:
            frame = img[:180, :320, :]
            frame = cv2.resize(frame, (192, 144), cv2.INTER_AREA)
            result_boxes, result_scores, result_classid, use_time = yolo.infer(frame)

            if len(result_classid) != 0:
                cls_id, area, max_box, score = get_max_area(
                    result_boxes, result_classid, result_scores
                )
                if area > 100 and score >= 0.9:
                    draw_rerectangle(frame, max_box, CATEGORIES[cls_id], score, area)

        # ---- Traffic light detection ----
        from speed_control import (
            red_light_flag, green_light_flag,
            red_light_area, green_light_area
        )
        if traffic_flag and not traffic_flag_2:
            element_img = img[:180, :320, :]
            element_img = cv2.resize(element_img, (192, 144), cv2.INTER_AREA)
            red_light_area = element_preprocess(element_img, 'red')
            if not red_light_flag and green_light_flag:
                green_light_area = element_preprocess(element_img, 'green')

        # ---- Speed control ----
        twist.linear.x = speed_control(cls_id, area, speed_x)

        pub.publish(twist)
        print(f"FPS: {fps:.1f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            twist.linear.x = 0.0
            twist.angular.z = 0
            pub.publish(twist)
            break

    rospy.loginfo("Exiting...")
    yolo.destroy()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
