"""
Finite State Machine for Speed Control.

Implements the full race-track FSM: adjusts vehicle speed based on the current
state (lane following, turning, crosswalk, speed limit, traffic light, etc.)
and detected sign/obstacle events.

State transitions are driven by YOLOv5 detection results (cls_id, area) and
timing logic for controlled maneuvers.
"""

import time

# Global state flags (referenced from main module)
# These are module-level globals shared across the control loop
side_walk_flag = True
speed_limit_flag = True
speed_flag = True
turn_left_flag = True
turn_right_flag = True
change_flag = True
change_flag_2 = False
change_flag_3 = True
change_flag_4 = True
turn_flag = True
turn_flag_2 = True
red_light_flag = True
green_light_flag = True
danger_flag = True
danger_flag_2 = True
danger_flag_3 = True
clean_right_flag = True
clean_flag = True
traffic_flag = True
traffic_flag_2 = False
yolo_flag = True
speed_4_flag = False
delayed_stop_flag = False

# Timing state
stop_time = [-1, -1]
turn_time = [0, 0]
change_time = [0, 0]
danger_time = [0, 0]
clean_time = [0, 0]
side_time = [0, 0]
traffic_time = [0, 0]
delayed_stop_time = 0


def wait(seconds):
    """Busy-wait for the given number of seconds."""
    start_time = time.time()
    while time.time() - start_time < seconds:
        pass


def speed_control(cls_id, area, speed_x=0.35):
    """
    Determine vehicle speed based on current FSM state and detection input.

    This is the core decision logic: it reads detection results (cls_id, area)
    and updates global state flags to transition through the race course.

    Args:
        cls_id: Detected object class ID (0=danger, 1=side_walk, 2=speed,
                3=speed_limit, 4=turn_left, 5=turn_right, 6=lane_change)
        area: Detection bounding box area (pixels).
        speed_x: Default cruise speed.

    Returns:
        speed: Target linear speed for the vehicle.
    """
    global side_walk_flag, speed_limit_flag, speed_flag
    global turn_left_flag, turn_right_flag
    global change_flag, change_flag_2, change_flag_3, change_flag_4
    global turn_flag, turn_flag_2
    global red_light_flag, red_light_area, green_light_flag, green_light_area
    global danger_flag, danger_flag_2, danger_flag_3
    global speed_4_flag, yolo_flag, traffic_flag, traffic_flag_2
    global clean_right_flag, clean_time, clean_flag
    global delayed_stop_flag, delayed_stop_time
    global stop_time, turn_time, change_time, danger_time, side_time, traffic_time
    global speed

    red_light_area = 0
    green_light_area = 0

    # ---- State transition triggers based on detection ----
    if turn_left_flag and turn_right_flag and int(cls_id) == 4 and area > 520:  # Left turn
        turn_left_flag = False
        turn_time[0] = time.time()
    elif turn_right_flag and turn_left_flag and int(cls_id) == 5 and area > 520:  # Right turn
        turn_right_flag = False
        turn_time[0] = time.time()
    elif side_walk_flag and int(cls_id) == 1 and not turn_flag and area >= 850:  # Crosswalk after turn
        side_walk_flag = False
        yolo_flag = False
        stop_time[0] = time.time()
        side_time[0] = time.time()
    elif speed_limit_flag and int(cls_id) == 3 and not side_walk_flag and area > 800:  # Speed limit sign
        speed_limit_flag = False
        clean_right_flag = False
        speed_4_flag = True
        clean_flag = False
        yolo_flag = False
        clean_time[0] = time.time()
    elif speed_flag and int(cls_id) == 2 and not speed_limit_flag and area > 1000:  # Speed limit lifted
        speed_flag = False
        yolo_flag = False
    elif traffic_flag and not speed_flag:
        traffic_flag_2 = False

    # ---- Traffic light logic ----
    if not speed_flag and red_light_flag and \
       (red_light_area > green_light_area + 100 and red_light_area > 450):
        red_light_flag = False
        delayed_stop_time = time.time() + 0.45
        traffic_time[0] = time.time()
    elif not red_light_flag and green_light_flag and \
         (green_light_area > red_light_area + 100 and green_light_area > 400):
        green_light_flag = False
        traffic_flag = False
        yolo_flag = True
    elif traffic_time[1] - traffic_time[0] > 6:  # Timeout fallback
        traffic_flag = False
        yolo_flag = True
    elif not traffic_flag and change_flag and int(cls_id) == 6 and area > 400:  # Lane change
        change_flag = False
        change_time[0] = time.time()
    elif not change_flag and danger_flag and int(cls_id) == 0 and area > 330:  # Danger/cone
        danger_flag = False
        yolo_flag = False
        danger_time[0] = time.time()

    # ---- Speed assignment based on current state ----
    speed = speed_x  # default

    if stop_time[0] != -1:
        speed = 0.0
        stop_time[1] = time.time()

    if stop_time[0] == -1 or stop_time[1] - stop_time[0] > 2:
        if speed_4_flag:  # Under speed limit
            speed = 0.28
            if not speed_flag:
                speed_4_flag = False
        elif turn_left_flag and turn_right_flag and turn_flag:  # Initial speed
            speed = 0.3
        elif turn_flag:  # Turning phase
            speed = 0.3
        elif not turn_flag_2 and side_walk_flag:
            speed = 0.3
        elif not turn_flag and turn_flag_2 and side_walk_flag:  # Turn end → crosswalk
            speed = 0.45
        elif not side_walk_flag and speed_limit_flag:  # Crosswalk → speed limit
            speed = 0.45
        elif not speed_flag and red_light_flag:  # Speed limit end → traffic light
            speed = 0.38
        elif traffic_flag and not red_light_flag and green_light_flag and \
                time.time() >= delayed_stop_time:  # Stop at red light
            speed = 0.0
            traffic_time[1] = time.time()
        elif not traffic_flag and change_flag:  # Post-stop → reduce speed for signs
            speed = 0.15
        elif not change_flag and change_flag_4:
            speed = 0.3
        elif not change_flag_4 and danger_flag:
            speed = 0.3
        elif not danger_flag and danger_flag_2:
            speed = 0.2
        elif not danger_flag and not danger_flag_2:
            speed = 0.4
        else:
            speed = speed_x

    stop_time = [-1, -1]

    # ---- Timing-based state transitions ----
    if turn_time[0] != 0 and (not turn_left_flag or not turn_right_flag):
        turn_time[1] = time.time()
        if turn_time[1] - turn_time[0] > 1.1:
            turn_flag = False
        if turn_time[1] - turn_time[0] > 3.8:
            turn_flag_2 = False

    if change_time[0] != 0 and not change_flag:
        change_time[1] = time.time()
        if change_time[1] - change_time[0] > 1.7:
            change_flag_2 = True
        if change_time[1] - change_time[0] > 3.62:
            change_flag_3 = False
        if change_time[1] - change_time[0] > 4.5:
            change_flag_4 = False

    if danger_time[0] != 0 and not danger_flag:
        danger_time[1] = time.time()
        if danger_time[1] - danger_time[0] > 0.4:
            danger_flag_2 = False
        if danger_time[1] - danger_time[0] > 1.5:
            danger_flag_3 = False
    danger_time = [0, 0]

    if clean_time[0] != 0 and not clean_flag:
        clean_time[1] = time.time()
        if clean_time[1] - clean_time[0] > 2:
            clean_right_flag = True
        if clean_time[1] - clean_time[0] > 6:
            yolo_flag = True
    clean_time = [0, 0]

    if side_time[0] != 0 and not side_walk_flag:
        side_time[1] = time.time()
        if side_time[1] - side_time[0] > 6:
            yolo_flag = True
    side_time = [0, 0]

    return speed


def switch_callback(msg):
    """ROS topic callback to enable/disable the system."""
    global EN
    EN = msg.data
