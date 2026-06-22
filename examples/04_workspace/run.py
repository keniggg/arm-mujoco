#!/usr/bin/env python3
"""Interactive grasping demo 鈥?drag the cube, press SPACE, watch the arm find & fetch it.

Features:
  - Cube is draggable (Ctrl + drag in MuJoCo viewer) anywhere in the workspace
  - A small target box sits inside the reachable region
  - Press SPACE to trigger the autonomous sequence:
      1. Camera scanning 鈥?find the red cube via OpenCV colour detection
      2. Dynamic IK planning 鈥?compute approach / grasp / lift / place targets
      3. Smooth arm motion with cosine interpolation (non-blocking)
      4. Force-controlled gripper close
      5. Place the cube into the box
  - Real-time gripper tactile-skin display, eye-in-hand camera, joint-state panel
  - Demo complete 鈫?manual control via MuJoCo sliders restored
"""

from __future__ import annotations

import ctypes
import math
import os
import random
import sys
import time
from dataclasses import dataclass
import numpy as np

# 鈹€鈹€ Windows high-resolution timer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import mujoco.viewer

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from common.model_loader import (
    RGB_CAMERA_NAME, BALL_POS,
    WRIST_CAMERA_LOCAL_POS, WRIST_CAMERA_FORWARD_LOCAL,
)
from common.ik_solver import (
    solve_gripper_center_ik, set_joint_positions, IKResult,
)
from common.motion import build_gripper_limits, command_gripper
from common.force_sensor import TactileReading, TactileSkinSensor, TactileSkinDisplay
from common.camera import RGBDCameraWindow

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Constants
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
TARGET_DISPLAY_HZ = 50
PHYSICS_SUBSTEPS = 5
FRAME_DT = 1.0 / TARGET_DISPLAY_HZ
START_TRIGGER_THRESHOLD = 0.95

D405C_DEPTH_WIDTH = 640
D405C_DEPTH_HEIGHT = 360
D405C_DEPTH_FPS = 90
D405C_RGB_WIDTH = 1280
D405C_RGB_HEIGHT = 720
D405C_RGB_FPS = 90
D405C_DEPTH_FOVX_DEG = 87.0
D405C_DEPTH_FOVY_DEG = 58.0
D405C_MIN_RANGE_M = 0.07
D405C_MAX_RANGE_M = 0.50
D405C_MIN_TARGET_M = 0.001
D405C_DEPTH_ACCURACY_AT_50CM = 0.02

SKIN_EVERY_N = 1       # 50 Hz loop; sensor response spec is <=25 ms.
JOINT_EVERY_N = 4      # 12.5 Hz
CAMERA_EVERY_N = 1     # D405C supports 90 Hz; this sim loop is capped at 50 Hz.
SCAN_RGBD_CAPTURE_EVERY_N = 1  # Opportunistic scan detection while moving.
SCAN_MOVING_CAPTURE_PROGRESS = 0.58

SPEED_NORMAL = 1.4
SPEED_SLOW = 0.55  # very slow for precise grasp positioning
SPEED_SCAN = 1.25
SPEED_LIFT = 0.32
SPEED_CARRY = 0.30
SPEED_PLACE = 0.72
SPEED_LOCAL_REPLAN = 1.15

WORKSPACE_MAX_SPHERES = 250
SCAN_POS_TOL = 0.040
SCAN_ORI_TOL = 0.55
SCAN_ORI_WEIGHT = 0.12
WORKSPACE_R_MIN = 0.12
WORKSPACE_R_MAX = 0.48
WORKSPACE_XY_LIMIT = 0.50
# Invisible planning point at the real midpoint between the two finger tips.
# The visible green marker from the source model is hidden; this body has no
# collision and is only used so IK targets the actual jaw center.
PINCH_CENTER_LOCAL_POS = (0.0002, 0.0, 0.13098)

# Colour ranges for red cube detection (HSV) 鈥?lenient to catch varying lighting
RED_LOWER_1 = (0, 60, 50)
RED_UPPER_1 = (18, 255, 255)
RED_LOWER_2 = (158, 60, 50)
RED_UPPER_2 = (180, 255, 255)
RGB_RED_DOMINANCE_MIN = 18
RGB_RED_RATIO_MIN = 1.12
BOX_BLUE_HSV_LOWER = (90, 45, 45)
BOX_BLUE_HSV_UPPER = (125, 255, 255)
BOX_BLUE_MIN = 85
BOX_BLUE_RED_MARGIN = 35
BOX_GREEN_RED_MARGIN = 15
BOX_BLUE_GREEN_MARGIN = -10
MIN_BOX_AREA_PX = 80
MIN_BOX_DEPTH_SAMPLES = 30
BOX_ESTIMATE_MAX_VIEWS = 5
BOX_ESTIMATE_STABILITY_M = 0.018
BOX_ACCEPT_MIN_AREA_PX = 5000
BOX_ACCEPT_MIN_XY_EXTENT_M = 0.120
BOX_ACCEPT_MIN_Z_EXTENT_M = 0.055
BOX_STRONG_SINGLE_VIEW_AREA_PX = 80000
BOX_STRONG_SINGLE_VIEW_MIN_XY_EXTENT_M = 0.125
BOX_STRONG_SINGLE_VIEW_MIN_Z_EXTENT_M = 0.060
BOX_DETECTION_Z_MIN = 0.014
BOX_DETECTION_Z_MAX = 0.160
BOX_FOOTPRINT_FULL_EXTENT_RATIO = 0.78
BOX_FOOTPRINT_VALID_MARGIN = 0.018
BOX_HINT_MAX_DEVIATION_M = 0.030

MIN_BALL_AREA_PX = 12       # minimum contour area (small cube at distance 鈮?15-30 px)
CUBE_HALF_SIZE = 0.020  # 40 mm cube side 鈥?good grip depth in 50 mm gripper
PAD_HALF_HEIGHT = 0.003
CUBE_REST_Z = CUBE_HALF_SIZE + PAD_HALF_HEIGHT
CAMERA_LOCAL_OFFSET_FROM_PINCH = (
    np.asarray(WRIST_CAMERA_LOCAL_POS, dtype=np.float64) -
    np.asarray(PINCH_CENTER_LOCAL_POS, dtype=np.float64)
)
CAMERA_TO_GRIPPER_CENTER = float(np.linalg.norm(CAMERA_LOCAL_OFFSET_FROM_PINCH))
CAMERA_FORWARD_LOCAL = np.asarray(WRIST_CAMERA_FORWARD_LOCAL, dtype=np.float64)
SCAN_CAMERA_DISTANCE = 0.24
MIN_DEPTH_SAMPLES = 8
MIN_RGBD_DEPTH_M = D405C_MIN_RANGE_M
MAX_RGBD_DEPTH_M = D405C_MAX_RANGE_M
VISION_Z_MIN = -0.03
VISION_Z_MAX = 0.45
GRASP_ORI_WEIGHT = 0.36
GRASP_ORI_TOL = 0.08
LIFT_ORI_TOL = 0.12
PLACE_ORI_TOL = 0.22
APPROACH_IK_POS_TOL = 0.010
GRASP_IK_POS_TOL = 0.005
GRASP_WORLD_AXIS_WEIGHT = 180.0
GRASP_TRANSPORT_AXIS_WEIGHT = 18.0
GRASP_TRANSPORT_TARGET_ALIGNMENT = 0.55
GRASP_OPENING_HINT_WEIGHT = 220.0
GRASP_OPENING_HINT_ALIGNMENT = 0.95
GRASP_DIAGONAL_OPENING_WEIGHT = 700.0
RGBD_MAX_ESTIMATE_SPREAD_M = 0.055
RGBD_MAX_PLANE_RESIDUAL_M = 0.010
RGBD_MIN_PLANE_INLIERS = 35
GLOBAL_RGBD_MIN_VIEWS = 2
GLOBAL_RGBD_MAX_VIEWS = 5
GLOBAL_RGBD_STABILITY_M = 0.015
GLOBAL_RGBD_STRONG_SINGLE_VIEW_INLIERS = 80
GLOBAL_RGBD_STRONG_SINGLE_VIEW_RESIDUAL_M = 0.0025
GLOBAL_RGBD_STRONG_SINGLE_VIEW_SPREAD_M = 0.012
RGBD_ANCHORED_CUBE_MAX_ERR_M = 0.035
PREGRASP_ANCHORED_CUBE_MAX_ERR_M = 0.025
GRIP_CONTACT_FORCE = 9.0
GRIP_CONFIRM_FORCE = 5.8
GRIP_LOCK_MIN_FORCE = 1.5
GRIP_MAX_FORCE = 115.0
GRIP_ACTUATOR_RELAX_DEADBAND = 3.0
GRIP_OVERFORCE_RELAX_CORRECTION = 0.00008
SKIN_FORCE_MIN_RECOGNITION_N = 0.1
SKIN_FORCE_RANGE_MIN_N = 0.0
SKIN_FORCE_RANGE_MAX_N = 20.0
SKIN_FORCE_RESOLUTION_N = 0.1
SKIN_RESPONSE_TIME_S = 0.025
SKIN_FILTER_ALPHA = 0.8
SKIN_SENSOR_SPACING_MIN_M = 0.0002
SKIN_OVERLOAD_MULTIPLIER = 2.5
SKIN_SUPPLY_VOLTAGE_V = 5.0
SKIN_COMM_PROTOCOL = "RS485"
SKIN_TOUCH_FORCE = SKIN_FORCE_MIN_RECOGNITION_N
SKIN_CONFIRM_FORCE = 5.0
SKIN_HOLD_MIN_FORCE = 16.0
SKIN_HOLD_TARGET_FORCE = 18.0
SKIN_HOLD_MAX_FORCE = 20.0
SKIN_PRELIFT_READY_FORCE = 16.0
SKIN_BALANCE_MAX_RATIO = 0.80
TACTILE_SECURE_OVERRIDE_MIN_FORCE = 6.0
TACTILE_SECURE_LOG_EVERY_N = 30
VISUAL_GRIP_CAPTURE_EVERY_N = 4
VISUAL_GRIP_LOG_EVERY_N = 18
VISUAL_GRIP_FILTER_ALPHA = 0.55
VISUAL_GRIP_ERR_SOFT_M = 0.004
VISUAL_GRIP_ERR_HARD_M = 0.016
VISUAL_GRIP_DROP_SOFT_M = 0.0025
VISUAL_GRIP_DROP_HARD_M = 0.0095
VISUAL_GRIP_RATE_SOFT_MPS = 0.003
VISUAL_GRIP_RATE_HARD_MPS = 0.020
VISUAL_GRIP_MAX_REL_XY_M = 0.028
VISUAL_GRIP_MAX_REL_Z_M = 0.026
VISUAL_GRIP_MIN_FORCE_BOOST = 6.0
VISUAL_GRIP_TARGET_FORCE_BOOST = 8.0
VISUAL_GRIP_MAX_FORCE_BOOST = 2.0
VISUAL_GRIP_POSITION_CORRECTION_GAIN = 0.095
VISUAL_GRIP_POSITION_CORRECTION_MAX = 0.0035
VISUAL_GRIP_EMERGENCY_DROP_M = 0.0070
VISUAL_GRIP_EMERGENCY_SLIP_M = 0.0110
VISUAL_GRIP_EMERGENCY_CORRECTION = 0.0028
VISUAL_TRANSPORT_ABORT_SLIP_M = 0.026
VISUAL_TRANSPORT_ABORT_DROP_M = 0.016
VISUAL_TRANSPORT_SEVERE_SLIP_M = 0.038
VISUAL_TRANSPORT_SEVERE_DROP_M = 0.028
VISUAL_TRANSPORT_ABORT_FRAMES = 3
GRIP_HOLD_PRELOAD = 0.0240
GRIP_HOLD_CORRECTION = 0.00165
GRIP_HOLD_FAST_CORRECTION = 0.00450
GRIP_HOLD_RELAX_CORRECTION = 0.000025
GRIP_FORCE_DEADBAND = 1.1
GRIP_CLOSE_FRAMES = 220
GRIP_CONTACT_HOLD_FRAMES = 7
GRIP_CLOSE_TIMEOUT_FRAMES = GRIP_CLOSE_FRAMES + 70
GRIP_HOLD_FRAMES = 150
GRIP_PRELIFT_MIN_FRAMES = 28
GRIP_PRELIFT_READY_FRAMES = 8
GRIP_TRANSPORT_HOLD_FRAMES = 12000
LIFT_MIN_FRAMES = 125
CARRY_MID_MIN_FRAMES = 70
PLACE_ABOVE_MIN_FRAMES = 130
PLACE_DROP_MIN_FRAMES = 80
GRASP_LIFT_HEIGHT = 0.052
PLACE_ABOVE_HEIGHT = 0.062
CARRY_CRUISE_HEIGHT = 0.030
CARRY_FINAL_RAISE_XY_M = 0.055
CARRY_STEP_MAX_XY_M = 0.034
CARRY_HIGH_STEP_MAX_XY_M = 0.060
CARRY_STEP_FINAL_DIRECT_XY_M = 0.120
CARRY_STEP_MIN_FRAMES = 70
TRANSPORT_STALL_PROGRESS_EPS_M = 0.006
TRANSPORT_STALL_MAX_STEPS = 3
PLACE_DIRECT_DROP_XY_MARGIN_M = 0.008
BOX_ENTRY_TARGET_FRACTION = 0.72
BOX_ENTRY_CENTER_BLEND = 0.40
TRANSPORT_REGRIP_MAX_COUNT = 3
TRANSPORT_REGRIP_MAX_ERR = 0.018
TRANSPORT_REGRIP_MAX_SPEED = 0.140
TRANSPORT_REGRIP_STABLE_SPEED = 0.050
TRANSPORT_REGRIP_FRAMES = 140
TRANSPORT_REGRIP_STABLE_FRAMES = 18
PLACE_RELEASE_CLEARANCE = -0.0008
PLACE_RELEASE_XY_MARGIN = 0.014
PLACE_RELEASE_Z_TOL = 0.003
PLACE_SETTLE_HOLD_FRAMES = 45
PLACE_RELEASE_OPEN_FRAMES = 95
MAX_PLACE_CORRECTIONS = 2
PLACE_REQUIRE_BOX_BOTTOM_CONTACT = True
PLACE_HELD_RECOVERY_MAX_ERR = 0.060
PLACE_HELD_RECOVERY_MIN_LIFT = 0.008
PLACE_RESCAN_RETURN_MIN_FRAMES = 90
MAX_PLACE_SUPPORT_RESCANS = 1
GRASP_TARGET_FACE_BIAS = 0.000
GRASP_TARGET_APPROACH_BIAS = 0.000
CARRY_MIN_LIFT = 0.012
CARRY_MAX_ERR = 0.028
TACTILE_SECURE_OVERRIDE_MAX_ERR = CARRY_MAX_ERR * 1.35
GRIP_LOCK_MAX_ERR = 0.024
GRIP_LOCK_MAX_OPEN_AXIS_ERR = CUBE_HALF_SIZE * 0.65
GRIP_LOCK_MAX_FACE_AXIS_ERR = CUBE_HALF_SIZE * 0.75
GRIP_LOCK_MAX_APPROACH_AXIS_ERR = CUBE_HALF_SIZE * 0.62
GRIP_STICTION_MAX_ERR = 0.030
GRIP_STICTION_STEP_M = 0.0024
GRIP_STICTION_GAIN = 0.45
CONTACT_CONFIRM_MAX_DIST = 0.0008
MIN_STABLE_FINGER_CONTACTS = 2
TARGET_FINGER_CONTACTS = 4
MIN_CONTACT_DIVERSITY = 0.008
MAX_CONTACT_PAIR_SKEW = 0.034
MAX_CONTACT_CENTER_ERR = 0.026
PREFERRED_CONTACT_CENTER_ERR = 0.010
PREFERRED_CONTACT_PAIR_SKEW = 0.018
PREFERRED_GRASP_Z_OFFSET = -0.004
GRASP_HEIGHT_SCORE_WEIGHT = 240.0
GRASP_APPROACH_ERR_SCORE_WEIGHT = 90.0
INITIAL_GRASP_Z_OFFSET = PREFERRED_GRASP_Z_OFFSET
GRASP_Z_OFFSETS = [
    -0.006,
    INITIAL_GRASP_Z_OFFSET,
    -0.002,
]
PRE_CLOSE_MAX_XY_ERR = 0.010
PRE_CLOSE_MAX_Z_ERR = 0.014
PRE_CLOSE_MAX_OPEN_AXIS_ERR = CUBE_HALF_SIZE * 0.45
PRE_CLOSE_MAX_FACE_AXIS_ERR = CUBE_HALF_SIZE * 0.70
PRE_CLOSE_MAX_CENTER_ERR = 0.022
PRE_CLOSE_MAX_ORI_ERR = 0.14
EARLY_CONTACT_MAX_PENETRATION = 0.0018
MAX_LOCAL_REPLAN = 2
PHYSICAL_EVAL_TOP_K = 24
LOCAL_PHYSICAL_EVAL_TOP_K = 6
REJECTED_GRASP_FRAME_PENALTY = 45.0
VISION_REPLAN_DELTA = 0.018
LOCAL_RGBD_KEEP_PLAN_DELTA = 0.008
MAX_PREGRASP_REPLANS = 2
PREGRASP_REAPPROACH_TOL = 0.020
GRIPPER_OPEN_TOL = 0.0045
GRIPPER_OPEN_WAIT_FRAMES = 90
APPROACH_CAPTURE_PROGRESS = 0.88
DESCEND_CLOSE_PROGRESS = 0.92
PRE_CLOSE_STABLE_FRAMES = 6
DESCEND_MID_CLEARANCE = 0.055
DESCEND_MID_MIN_FRAMES = 65
LOCAL_RETRACT_MIN_FRAMES = 55
LOCAL_REAPPROACH_MIN_FRAMES = 50
ARM_DONE_QPOS_TOL = 0.018
ARM_DONE_QVEL_TOL = 0.08
ARM_DONE_MAX_SETTLE_FRAMES = 45
PRELIFT_LOST_GRACE_FRAMES = 30
CUBE_IDLE_FREEZE_SPEED = 0.010
CUBE_IDLE_FREEZE_ANG_SPEED = 0.050
CUBE_REST_Z_TOL = 0.006
CUBE_PRECONTACT_REST_Z_TOL = 0.012
CUBE_STATIC_ANCHOR_MAX_Z = CUBE_REST_Z + 0.045
BOX_VERIFY_FRAMES = 45
HELD_BOX_SCAN_Z_PLANE = CUBE_REST_Z + 0.070
HELD_BOX_SCAN_MIN_CUBE_Z = CUBE_REST_Z + 0.105
HIGH_CUBE_RECOVERY_OPEN_FRAMES = 45
HIGH_CUBE_SETTLE_SPEED = 0.035
HIGH_CUBE_SETTLE_ANG_SPEED = 0.12
HIGH_CUBE_SETTLE_FRAMES = 8
HIGH_CUBE_SETTLE_TIMEOUT_FRAMES = 220
REQUIRE_BOX_BEFORE_GRASP = True
ALLOW_PARTIAL_BOX_HINT_PLACEMENT = False

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# XML injection 鈥?target box
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
BOX_POS = np.array([0.42, -0.15, 0.06], dtype=np.float64)
BOX_SIZE = np.array([0.07, 0.07, 0.04], dtype=np.float64)  # half-extents
BOX_WALL = 0.005  # wall thickness
BOX_CENTER_Z_PRIOR = CUBE_REST_Z + BOX_SIZE[2] - BOX_WALL + 0.002
PLACE_INSIDE_Z_TOL = 0.008


def inject_control_actuators(xml_content: str) -> str:
    """Add dummy actuators that appear as sliders in the MuJoCo viewer.

    These produce zero force (gainprm=0) so they don't affect physics.
    The sliders are used purely as UI controls for:
      - start_demo:  slide to 1 to trigger the autonomous sequence
      - cube_x/y/z:  reposition the target cube (only during idle)
    """
    if 'name="start_demo"' in xml_content:
        return xml_content

    ctrl_xml = f"""
    <general name="start_demo" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="0 1" dyntype="none" gainprm="0" biasprm="0 0 0"/>
    <general name="cube_x" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="-{WORKSPACE_XY_LIMIT:.2f} {WORKSPACE_XY_LIMIT:.2f}" dyntype="none" gainprm="0" biasprm="0 0 0"/>
    <general name="cube_y" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="-{WORKSPACE_XY_LIMIT:.2f} {WORKSPACE_XY_LIMIT:.2f}" dyntype="none" gainprm="0" biasprm="0 0 0"/>
    <general name="cube_z" joint="Joint1" biastype="affine" gaintype="fixed"
             ctrlrange="{CUBE_REST_Z:.3f} 0.35" dyntype="none" gainprm="0" biasprm="0 0 0"/>
  """
    # Insert into the existing <actuator> section, right before </actuator>
    act_end = xml_content.find("</actuator>")
    if act_end >= 0:
        xml_content = xml_content[:act_end] + ctrl_xml + xml_content[act_end:]
    return xml_content


def inject_target_box(xml_content: str) -> str:
    if 'name="target_box"' in xml_content:
        return xml_content

    bx, by, bz = BOX_POS
    sx, sy, sz = BOX_SIZE
    t = BOX_WALL
    rgba = (0.3, 0.6, 1.0, 0.7)  # blueish semi-transparent

    box_body = f"""
        <body name="target_box" pos="{bx} {by} {bz}">
            <geom name="box_bottom" type="box" size="{sx} {sy} {t/2}"
                  pos="0 0 {-sz}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_front" type="box" size="{sx} {t/2} {sz}"
                  pos="0 {sy} 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_back" type="box" size="{sx} {t/2} {sz}"
                  pos="0 {-sy} 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_left" type="box" size="{t/2} {sy} {sz}"
                  pos="{-sx} 0 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
            <geom name="box_right" type="box" size="{t/2} {sy} {sz}"
                  pos="{sx} 0 0" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"/>
        </body>"""
    xml_content = xml_content.replace("</worldbody>", box_body + "\n  </worldbody>")
    return xml_content


def configure_finger_mesh_collision(xml_content: str) -> str:
    """Use Link7/Link8 as the real gripper contact bodies.

    Keep the visible gripper geometry unchanged.  Link7/Link8 stay as the
    physical fingertip meshes.  The electric-skin sites are only sensing
    volumes; they are not extra pads, support blocks, or collision geometry.
    """
    left_marker = '<geom type="mesh" rgba="0.592157 0.666667 0.682353 1" mesh="Link7" />'
    right_marker = '<geom type="mesh" rgba="0.592157 0.666667 0.682353 1" mesh="Link8" />'
    center_marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118" contype="0" conaffinity="0" group="1" density="0" rgba="0 1 0 1" />'
    hidden_center_marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118" contype="0" conaffinity="0" group="1" density="0" rgba="0 1 0 0" />'
    left_collision = (
        '<geom name="left_finger_collision" type="mesh" '
        'rgba="0.592157 0.666667 0.682353 1" mesh="Link7" '
        'condim="6" friction="260.0 100.0 36.0" '
        'solimp="0.97 0.995 0.0005" solref="0.004 1" />\n'
        '                    <site name="left_inner_skin_site" type="ellipsoid" '
        'pos="-0.005 -0.0005 0.002" size="0.020 0.022 0.010" '
        'rgba="0 0 0 0" group="5" />'
    )
    right_collision = (
        '<geom name="right_finger_collision" type="mesh" '
        'rgba="0.592157 0.666667 0.682353 1" mesh="Link8" '
        'condim="6" friction="260.0 100.0 36.0" '
        'solimp="0.97 0.995 0.0005" solref="0.004 1" />\n'
        '                    <site name="right_inner_skin_site" type="ellipsoid" '
        'pos="-0.005 0.0005 0.002" size="0.020 0.022 0.010" '
        'rgba="0 0 0 0" group="5" />'
    )
    if 'name="left_finger_collision"' not in xml_content:
        xml_content = xml_content.replace(left_marker, left_collision, 1)
    if 'name="right_finger_collision"' not in xml_content:
        xml_content = xml_content.replace(right_marker, right_collision, 1)
    if 'name="gripper_pinch_center_body"' not in xml_content:
        px, py, pz = PINCH_CENTER_LOCAL_POS
        pinch_body = (
            f'\n                  <body name="gripper_pinch_center_body" '
            f'pos="{px:.5f} {py:.5f} {pz:.5f}"/>'
        )
        xml_content = xml_content.replace(center_marker, hidden_center_marker + pinch_body, 1)
    else:
        xml_content = xml_content.replace(center_marker, hidden_center_marker, 1)
    return xml_content


def inject_finger_skin_sensors(xml_content: str) -> str:
    """Attach touch sensors to the inner pads of the two gripper fingers."""
    if 'name="left_inner_skin_touch"' in xml_content:
        return xml_content
    sensor_lines = (
        '    <touch name="left_inner_skin_touch" site="left_inner_skin_site"/>\n'
        '    <touch name="right_inner_skin_touch" site="right_inner_skin_site"/>\n'
    )
    if "<sensor>" in xml_content:
        sensor_end = xml_content.find("</sensor>")
        if sensor_end >= 0:
            return (
                xml_content[:sensor_end] +
                sensor_lines +
                xml_content[sensor_end:]
            )
    sensor_xml = f"\n  <sensor>\n{sensor_lines}  </sensor>"
    return xml_content.replace("</mujoco>", sensor_xml + "\n</mujoco>", 1)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Cube detector 鈥?camera-based red-cube localisation
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class BallDetector:
    """Find the red cube in an RGB-D image and estimate its 3-D position."""

    def __init__(self, model, camera_id: int, ball_body_id: int):
        self._model = model
        self._camera_id = camera_id
        self._ball_body_id = ball_body_id
        self._last_detection_uv: tuple[int, int] | None = None
        self._last_detection_candidates_uv: list[tuple[float, float]] = []
        self._last_detection_mask: np.ndarray | None = None
        self._last_detection_spread_px = 0.0
        self._last_estimate_spread_m = 0.0
        self._last_plane_residual_m = float("inf")
        self._last_plane_inliers = 0
        self._last_estimated_xyz: np.ndarray | None = None
        self._last_axis_uv: np.ndarray | None = None
        self._last_axis_confidence = 0.0

    def _clear_detection(self) -> None:
        self._last_detection_uv = None
        self._last_detection_candidates_uv = []
        self._last_detection_mask = None
        self._last_detection_spread_px = 0.0
        self._last_estimate_spread_m = 0.0
        self._last_plane_residual_m = float("inf")
        self._last_plane_inliers = 0
        self._last_estimated_xyz = None
        self._last_axis_uv = None
        self._last_axis_confidence = 0.0

    def detect(self, rgb: np.ndarray, data) -> tuple[tuple[int, int] | None, int]:
        """Return ((cx, cy), radius_px) or (None, 0) if not found."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = mask1 | mask2
        rgb_i = rgb.astype(np.int16)
        red = rgb_i[:, :, 0]
        green = rgb_i[:, :, 1]
        blue = rgb_i[:, :, 2]
        max_gb = np.maximum(green, blue)
        rgb_red = (
            (red >= 45) &
            ((red - max_gb) >= RGB_RED_DOMINANCE_MIN) &
            (red >= RGB_RED_RATIO_MIN * np.maximum(max_gb, 1))
        )
        mask |= (rgb_red.astype(np.uint8) * 255)
        # Morphological clean-up: remove small noise, merge nearby blobs
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._clear_detection()
            return None, 0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_BALL_AREA_PX:
            self._clear_detection()
            return None, 0
        detection_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(detection_mask, [largest], -1, 255, thickness=cv2.FILLED)
        self._last_detection_mask = detection_mask
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        pts = largest.reshape(-1, 2).astype(np.float64)
        candidates: list[np.ndarray] = [np.array([cx, cy], dtype=np.float64)]
        moments = cv2.moments(largest)
        if abs(float(moments.get("m00", 0.0))) > 1e-9:
            candidates.append(np.array([
                moments["m10"] / moments["m00"],
                moments["m01"] / moments["m00"],
            ], dtype=np.float64))
        rect = cv2.minAreaRect(largest)
        candidates.append(np.array(rect[0], dtype=np.float64))
        x, y, w, h = cv2.boundingRect(largest)
        candidates.append(np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float64))
        if len(pts) > 0:
            candidates.append(np.mean(pts, axis=0))

        candidate_arr = np.vstack(candidates)
        robust_uv = np.median(candidate_arr, axis=0)
        spread = np.linalg.norm(candidate_arr - robust_uv, axis=1)
        self._last_detection_spread_px = float(np.max(spread)) if len(spread) else 0.0
        self._last_detection_candidates_uv = [
            (float(p[0]), float(p[1])) for p in candidate_arr
        ]
        self._last_detection_uv = (
            int(round(float(robust_uv[0]))),
            int(round(float(robust_uv[1]))),
        )
        if len(pts) >= 5:
            centered = pts - np.mean(pts, axis=0)
            cov = centered.T @ centered / max(len(pts) - 1, 1)
            vals, vecs = np.linalg.eigh(cov)
            order = np.argsort(vals)
            major = vecs[:, order[-1]]
            if major[0] < 0:
                major = -major
            denom = max(float(vals[order[-1]]), 1e-9)
            self._last_axis_confidence = float(
                np.clip((vals[order[-1]] - vals[order[-2]]) / denom, 0.0, 1.0)
            )
            self._last_axis_uv = major
            if self._last_axis_confidence < 0.02:
                rect = cv2.minAreaRect(largest)
                box = cv2.boxPoints(rect).astype(np.float64)
                edges = np.roll(box, -1, axis=0) - box
                lengths = np.linalg.norm(edges, axis=1)
                edge = edges[int(np.argmax(lengths))]
                norm = float(np.linalg.norm(edge))
                if norm > 1e-6:
                    edge /= norm
                    if edge[0] < 0:
                        edge = -edge
                    self._last_axis_uv = edge
                    self._last_axis_confidence = 0.05
        else:
            self._last_axis_uv = None
            self._last_axis_confidence = 0.0
        return self._last_detection_uv, int(radius)

    def _back_project_pixels(self, xs: np.ndarray, ys: np.ndarray,
                             depths: np.ndarray, data,
                             img_w: int, img_h: int) -> np.ndarray:
        """Back-project metric depth pixels into world coordinates."""
        cam_pos = data.cam_xpos[self._camera_id].copy()
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)

        fx_px = (img_w / 2.0) / math.tan(math.radians(D405C_DEPTH_FOVX_DEG) / 2.0)
        fy_px = (img_h / 2.0) / math.tan(math.radians(D405C_DEPTH_FOVY_DEG) / 2.0)

        depths = depths.astype(np.float64)
        x_cam = (xs.astype(np.float64) - img_w / 2.0) * depths / fx_px
        y_cam = (img_h / 2.0 - ys.astype(np.float64)) * depths / fy_px
        z_cam = -depths
        pts_cam = np.column_stack((x_cam, y_cam, z_cam))
        return cam_pos + pts_cam @ cam_mat.T

    def _depth_near_uv(self, depth: np.ndarray, uv: tuple[float, float],
                       radius_px: int = 3) -> float | None:
        cx = int(round(float(uv[0])))
        cy = int(round(float(uv[1])))
        h, w = depth.shape
        x0, x1 = max(0, cx - radius_px), min(w, cx + radius_px + 1)
        y0, y1 = max(0, cy - radius_px), min(h, cy + radius_px + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        patch = depth[y0:y1, x0:x1]
        valid = (
            np.isfinite(patch) &
            (patch >= MIN_RGBD_DEPTH_M) &
            (patch <= MAX_RGBD_DEPTH_M)
        )

        if self._last_detection_mask is not None:
            mask_patch = self._last_detection_mask[y0:y1, x0:x1] > 0
            masked_valid = valid & mask_patch
            if np.count_nonzero(masked_valid) >= 2:
                valid = masked_valid

        vals = patch[valid]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def _masked_depth_cloud(self, depth: np.ndarray, data,
                            img_w: int, img_h: int) -> np.ndarray | None:
        if self._last_detection_mask is None:
            return None
        ys, xs = np.nonzero(self._last_detection_mask > 0)
        if xs.size == 0:
            return None
        if xs.size > 2500:
            idx = np.linspace(0, xs.size - 1, 2500).astype(np.int64)
            xs = xs[idx]
            ys = ys[idx]
        samples = depth[ys, xs]
        valid = (
            np.isfinite(samples) &
            (samples >= MIN_RGBD_DEPTH_M) &
            (samples <= MAX_RGBD_DEPTH_M)
        )
        if np.count_nonzero(valid) < MIN_DEPTH_SAMPLES:
            return None
        return self._back_project_pixels(
            xs[valid], ys[valid], samples[valid], data, img_w, img_h,
        )

    def _visible_face_center_estimate(self, cloud: np.ndarray,
                                      data) -> np.ndarray | None:
        """Fit the visible cube face and step inward by half the cube size."""
        if cloud.shape[0] < RGBD_MIN_PLANE_INLIERS:
            return None
        cam_pos = data.cam_xpos[self._camera_id].copy()
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)
        view_dir = _unit(
            cam_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
        def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            center = np.median(points, axis=0)
            centered = points - center
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = _unit(vh[-1], -view_dir)
            if float(np.dot(normal, cam_pos - center)) < 0.0:
                normal = -normal
            residuals = np.abs(centered @ normal)
            return center, normal, residuals

        center, normal, residuals = fit_plane(cloud)
        full_residual = float(np.median(residuals)) if residuals.size else float("inf")
        if full_residual <= RGBD_MAX_PLANE_RESIDUAL_M:
            residual_gate = max(0.0045, float(np.percentile(residuals, 80)) + 0.0015)
            inliers = cloud[residuals <= residual_gate]
            if inliers.shape[0] >= RGBD_MIN_PLANE_INLIERS:
                center, normal, residuals = fit_plane(inliers)
                face_pts = inliers
            else:
                face_pts = cloud
        else:
            depth_coord = (cloud - cam_pos) @ view_dir
            near_cut = np.percentile(depth_coord, 42)
            face_pts = cloud[depth_coord <= near_cut]
            if face_pts.shape[0] < RGBD_MIN_PLANE_INLIERS:
                face_pts = cloud
            center, normal, residuals = fit_plane(face_pts)

        residual_gate = max(0.0045, float(np.percentile(residuals, 65)) + 0.0015)
        inliers = face_pts[residuals <= residual_gate]
        if inliers.shape[0] >= RGBD_MIN_PLANE_INLIERS:
            center, normal, residuals = fit_plane(inliers)
            face_pts = inliers
        residual = float(np.median(residuals)) if residuals.size else float("inf")
        self._last_plane_residual_m = residual
        self._last_plane_inliers = int(face_pts.shape[0])
        if (self._last_plane_inliers < RGBD_MIN_PLANE_INLIERS or
                residual > RGBD_MAX_PLANE_RESIDUAL_M):
            return None
        return center - normal * CUBE_HALF_SIZE

    def _visible_surface_to_cube_center(self, surface_xyz: np.ndarray,
                                        data) -> np.ndarray:
        """Move from the visible RGB-D surface toward the cube center."""
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)
        view_dir = cam_mat @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        norm = float(np.linalg.norm(view_dir))
        if norm < 1e-9:
            return surface_xyz.copy()
        view_dir /= norm
        max_axis = max(float(np.max(np.abs(view_dir))), 1e-6)
        surface_to_center = CUBE_HALF_SIZE / max_axis
        surface_to_center = float(
            np.clip(surface_to_center, CUBE_HALF_SIZE, CUBE_HALF_SIZE * 1.75)
        )
        return surface_xyz + view_dir * surface_to_center

    def estimate_3d_from_depth(self, center_uv: tuple[int, int],
                               depth_img: np.ndarray, data,
                               img_w: int, img_h: int) -> np.ndarray | None:
        """Estimate cube center from aligned depth, without a table-height prior."""
        depth = np.asarray(depth_img, dtype=np.float64)
        if depth.ndim != 2:
            return None
        img_h, img_w = depth.shape

        candidates = list(self._last_detection_candidates_uv)
        if not candidates:
            candidates = [(float(center_uv[0]), float(center_uv[1]))]

        estimates: list[np.ndarray] = []
        plane_estimate: np.ndarray | None = None
        for uv in candidates:
            z_depth = self._depth_near_uv(depth, uv)
            if z_depth is None:
                continue
            surface = self._back_project_pixels(
                np.array([float(uv[0])], dtype=np.float64),
                np.array([float(uv[1])], dtype=np.float64),
                np.array([z_depth], dtype=np.float64),
                data, img_w, img_h,
            )[0]
            estimates.append(self._visible_surface_to_cube_center(surface, data))

        cloud = self._masked_depth_cloud(depth, data, img_w, img_h)
        if cloud is not None:
            plane_estimate = self._visible_face_center_estimate(cloud, data)
            if plane_estimate is not None:
                estimates.append(plane_estimate)

            surface_med = np.median(cloud, axis=0)
            estimates.append(self._visible_surface_to_cube_center(surface_med, data))

            z_low, z_high = np.percentile(cloud[:, 2], [5, 95])
            if z_high > z_low:
                extent_est = self._visible_surface_to_cube_center(surface_med, data)
                if z_high - z_low >= CUBE_HALF_SIZE * 0.70:
                    extent_est[2] = 0.5 * (z_low + z_high)
                estimates.append(extent_est)

        if not estimates:
            return None

        arr = np.vstack(estimates)
        if plane_estimate is not None:
            robust = plane_estimate
            err = np.linalg.norm(arr - robust, axis=1)
        else:
            xyz_med = np.median(arr, axis=0)
            err = np.linalg.norm(arr - xyz_med, axis=1)
            keep = err <= max(0.030, float(np.median(err)) + 0.015)
            if np.any(keep):
                arr = arr[keep]
                err = err[keep]
            robust = np.median(arr, axis=0)
        self._last_estimate_spread_m = float(np.max(err)) if len(err) else 0.0
        self._last_estimated_xyz = robust
        return robust

    @property
    def last_estimated_xyz(self) -> np.ndarray | None:
        return self._last_estimated_xyz

    @property
    def last_detection_spread_px(self) -> float:
        return self._last_detection_spread_px

    @property
    def last_estimate_spread_m(self) -> float:
        return self._last_estimate_spread_m

    @property
    def last_plane_residual_m(self) -> float:
        return self._last_plane_residual_m

    @property
    def last_plane_inliers(self) -> int:
        return self._last_plane_inliers

    def last_opening_hints_world(self, data) -> list[np.ndarray]:
        """Return RGB-derived horizontal gripper-opening hints, if reliable."""
        if self._last_axis_uv is None or self._last_axis_confidence < 0.02:
            return []
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)
        # Image +u is camera +X, image +v is camera -Y.
        axis_cam = np.array([
            self._last_axis_uv[0],
            -self._last_axis_uv[1],
            0.0,
        ], dtype=np.float64)
        axis_world = cam_mat @ axis_cam
        axis_world[2] = 0.0
        norm = float(np.linalg.norm(axis_world))
        if norm < 1e-6:
            return []
        edge = axis_world / norm
        normal = np.array([-edge[1], edge[0], 0.0], dtype=np.float64)
        return [normal, -normal, edge, -edge]


class TargetBoxDetector:
    """Find the blue placement box in RGB-D and estimate its opening centre."""

    def __init__(self, model, camera_id: int):
        self._model = model
        self._camera_id = camera_id
        self.last_estimated_xyz: np.ndarray | None = None
        self.last_raw_center_xyz: np.ndarray | None = None
        self.last_area_px = 0.0
        self.last_points = 0
        self.last_extent_m = np.zeros(3, dtype=np.float64)
        self.last_center_correction_m = 0.0
        self.last_footprint_inside_ratio = 0.0

    def _clear_detection(self) -> None:
        self.last_estimated_xyz = None
        self.last_raw_center_xyz = None
        self.last_area_px = 0.0
        self.last_points = 0
        self.last_extent_m = np.zeros(3, dtype=np.float64)
        self.last_center_correction_m = 0.0
        self.last_footprint_inside_ratio = 0.0

    def _back_project_pixels(self, xs: np.ndarray, ys: np.ndarray,
                             depths: np.ndarray, data,
                             img_w: int, img_h: int) -> np.ndarray:
        cam_pos = data.cam_xpos[self._camera_id].copy()
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)
        fx_px = (img_w / 2.0) / math.tan(math.radians(D405C_DEPTH_FOVX_DEG) / 2.0)
        fy_px = (img_h / 2.0) / math.tan(math.radians(D405C_DEPTH_FOVY_DEG) / 2.0)

        depths = depths.astype(np.float64)
        x_cam = (xs.astype(np.float64) - img_w / 2.0) * depths / fx_px
        y_cam = (img_h / 2.0 - ys.astype(np.float64)) * depths / fy_px
        z_cam = -depths
        pts_cam = np.column_stack((x_cam, y_cam, z_cam))
        return cam_pos + pts_cam @ cam_mat.T

    def _complete_axis_center(
        self,
        lo: float,
        hi: float,
        mean: float,
        cam_coord: float,
        half_extent: float,
    ) -> float:
        full_extent = 2.0 * float(half_extent)
        observed_extent = float(hi - lo)
        if observed_extent >= full_extent * BOX_FOOTPRINT_FULL_EXTENT_RATIO:
            return 0.5 * (float(lo) + float(hi))
        if cam_coord >= mean:
            return float(hi) - float(half_extent)
        return float(lo) + float(half_extent)

    def _complete_box_center_from_cloud(
        self,
        cloud: np.ndarray,
        data,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        raw_center = 0.5 * (lo + hi)
        cam_pos = data.cam_xpos[self._camera_id].copy()
        mean_xy = np.median(cloud[:, :2], axis=0)
        center = raw_center.copy()
        center[0] = self._complete_axis_center(
            lo[0], hi[0], mean_xy[0], cam_pos[0], BOX_SIZE[0])
        center[1] = self._complete_axis_center(
            lo[1], hi[1], mean_xy[1], cam_pos[1], BOX_SIZE[1])
        center[2] = raw_center[2]

        footprint_lo = center[:2] - BOX_SIZE[:2] - BOX_FOOTPRINT_VALID_MARGIN
        footprint_hi = center[:2] + BOX_SIZE[:2] + BOX_FOOTPRINT_VALID_MARGIN
        inside = np.all(
            (cloud[:, :2] >= footprint_lo) &
            (cloud[:, :2] <= footprint_hi),
            axis=1,
        )
        inside_ratio = float(np.count_nonzero(inside)) / max(float(cloud.shape[0]), 1.0)
        correction = float(np.linalg.norm(center[:2] - raw_center[:2]))
        return center, correction, inside_ratio

    def estimate_3d_from_rgbd(self, rgb: np.ndarray, depth_img: np.ndarray,
                              data, img_w: int, img_h: int) -> np.ndarray | None:
        """Estimate box centre from the observed blue wall/bottom cloud.

        The scene still injects the box at a simulator position, but planning
        below consumes only this camera estimate.
        """
        self._clear_detection()
        if self._camera_id < 0 or not CV2_AVAILABLE:
            return None
        depth = np.asarray(depth_img, dtype=np.float64)
        if depth.ndim != 2:
            return None
        img_h, img_w = depth.shape

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, BOX_BLUE_HSV_LOWER, BOX_BLUE_HSV_UPPER)
        rgb_i = rgb.astype(np.int16)
        red = rgb_i[:, :, 0]
        green = rgb_i[:, :, 1]
        blue = rgb_i[:, :, 2]
        rgb_blue = (
            (blue >= BOX_BLUE_MIN) &
            (blue >= red + BOX_BLUE_RED_MARGIN) &
            (green >= red + BOX_GREEN_RED_MARGIN) &
            (blue >= green + BOX_BLUE_GREEN_MARGIN)
        )
        mask |= (rgb_blue.astype(np.uint8) * 255)

        valid_depth = (
            np.isfinite(depth) &
            (depth >= MIN_RGBD_DEPTH_M) &
            (depth <= MAX_RGBD_DEPTH_M)
        )
        mask &= (valid_depth.astype(np.uint8) * 255)
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -float("inf")
        best_center: np.ndarray | None = None
        best_raw_center: np.ndarray | None = None
        best_area = 0.0
        best_points = 0
        best_extent = np.zeros(3, dtype=np.float64)
        best_correction = 0.0
        best_inside_ratio = 0.0
        expected_xy = BOX_SIZE[:2] * 2.0
        min_xy_extent = BOX_SIZE[:2] * 0.65
        min_z_extent = BOX_SIZE[2] * 0.35

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MIN_BOX_AREA_PX:
                continue
            contour_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
            ys, xs = np.nonzero((contour_mask > 0) & valid_depth)
            if xs.size < MIN_BOX_DEPTH_SAMPLES:
                continue
            if xs.size > 5000:
                idx = np.linspace(0, xs.size - 1, 5000).astype(np.int64)
                xs = xs[idx]
                ys = ys[idx]
            cloud = self._back_project_pixels(
                xs, ys, depth[ys, xs], data, img_w, img_h)
            z_ok = (
                (cloud[:, 2] >= BOX_DETECTION_Z_MIN) &
                (cloud[:, 2] <= BOX_DETECTION_Z_MAX)
            )
            xy_ok = np.linalg.norm(cloud[:, :2], axis=1) <= (WORKSPACE_R_MAX + 0.20)
            cloud = cloud[z_ok & xy_ok]
            if cloud.shape[0] < MIN_BOX_DEPTH_SAMPLES:
                continue

            lo = np.percentile(cloud, 3, axis=0)
            hi = np.percentile(cloud, 97, axis=0)
            extent = hi - lo
            if (extent[0] < min_xy_extent[0] or
                    extent[1] < min_xy_extent[1] or
                    extent[2] < min_z_extent):
                continue
            raw_center = 0.5 * (lo + hi)
            center, correction, inside_ratio = self._complete_box_center_from_cloud(
                cloud, data, lo, hi)
            if inside_ratio < 0.86:
                continue
            center[2] = float(np.clip(
                center[2], CUBE_REST_Z, BOX_DETECTION_Z_MAX))
            extent_penalty = float(np.linalg.norm(np.maximum(expected_xy - extent[:2], 0.0)))
            score = (
                area +
                0.12 * cloud.shape[0] -
                900.0 * extent_penalty +
                1200.0 * inside_ratio -
                80.0 * correction
            )
            if score > best_score:
                best_score = score
                best_center = center
                best_raw_center = raw_center
                best_area = area
                best_points = int(cloud.shape[0])
                best_extent = extent
                best_correction = correction
                best_inside_ratio = inside_ratio

        if best_center is None:
            return None
        self.last_estimated_xyz = best_center.copy()
        self.last_raw_center_xyz = (
            None if best_raw_center is None else best_raw_center.copy()
        )
        self.last_area_px = best_area
        self.last_points = best_points
        self.last_extent_m = best_extent.copy()
        self.last_center_correction_m = best_correction
        self.last_footprint_inside_ratio = best_inside_ratio
        return best_center


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Non-blocking smooth arm controller
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class SmoothArmController:
    """Cosine-interpolated joint-space motion  (non-blocking).

    After interpolation finishes the controller enters a *settling* phase:
    it keeps writing the final target for SETTLE_FRAMES more frames so
    the PD actuators have time to physically reach the target despite
    gravity / dynamics.
    """

    SETTLE_FRAMES = 8  # keep phase transitions responsive after each segment

    def __init__(self, model, data, arm_joints: list[int],
                 joint_to_actuator: dict[int, int]):
        self._model = model
        self._data = data
        self._joints = arm_joints
        self._j2a = joint_to_actuator
        cur = np.array([data.qpos[model.jnt_qposadr[j]] for j in arm_joints],
                       dtype=np.float64)
        self._start = cur.copy()
        self._target = cur.copy()
        self._progress = 1.0
        self._total_frames = 1
        self._done = True
        self._settle_left = 0
        self._post_settle_frames = 0

    @property
    def done(self) -> bool:
        """True only after interpolation + settling have both finished."""
        return (
            self._done and
            self._settle_left <= 0 and
            (
                self.physically_settled() or
                self._post_settle_frames >= ARM_DONE_MAX_SETTLE_FRAMES
            )
        )

    def near_done(self, fraction: float = 0.92) -> bool:
        """True when a new target can be blended in without a visible stop."""
        return self._done or self._progress >= float(np.clip(fraction, 0.0, 1.0))

    def physically_settled(self) -> bool:
        q_err, max_qvel = self.settle_metrics()
        return q_err <= ARM_DONE_QPOS_TOL and max_qvel <= ARM_DONE_QVEL_TOL

    def settle_metrics(self) -> tuple[float, float]:
        q_err = float(np.max(np.abs(self.current() - self._target)))
        qvels = []
        for jid in self._joints:
            dof = self._model.jnt_dofadr[jid]
            qvels.append(abs(float(self._data.qvel[dof])))
        max_qvel = max(qvels) if qvels else 0.0
        return q_err, max_qvel

    def current(self) -> np.ndarray:
        return np.array([self._data.qpos[self._model.jnt_qposadr[j]]
                         for j in self._joints], dtype=np.float64)

    def set_target(self, angles: np.ndarray, speed: float = 1.0,
                   min_frames: int | None = None) -> None:
        cur = self.current()
        diffs = angles - cur
        max_diff = float(np.max(np.abs(diffs)))
        divisor = 0.008 * PHYSICS_SUBSTEPS
        min_f = max(80 // PHYSICS_SUBSTEPS, 10)
        if min_frames is not None:
            min_f = max(min_f, int(min_frames))
        raw = int(max_diff / (speed * divisor)) + 1
        self._total_frames = max(raw, min_f)
        self._total_frames = min(self._total_frames, max(200, min_f))
        self._start = cur.copy()
        self._target = angles.copy()
        self._progress = 0.0
        self._done = False
        self._settle_left = 0
        self._post_settle_frames = 0

    def step(self) -> None:
        if self._done:
            if self._settle_left > 0:
                self._settle_left -= 1
                self._post_settle_frames = 0
            else:
                self._post_settle_frames += 1
            # Always hold the final target so the PD actuators can converge
            self._write_angles(self._target)
            return
        self._progress += 1.0 / self._total_frames
        if self._progress >= 1.0:
            self._progress = 1.0
            self._done = True
            self._settle_left = self.SETTLE_FRAMES
        t = 0.5 - 0.5 * math.cos(self._progress * math.pi)
        angles = self._start + (self._target - self._start) * t
        self._write_angles(angles)

    def _write_angles(self, angles: np.ndarray) -> None:
        for idx, jid in enumerate(self._joints):
            act = self._j2a.get(jid)
            if act is not None:
                self._data.ctrl[act] = float(angles[idx])


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Non-blocking gripper controller
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class SmoothGripperController:
    """Rate-limited gripper with a late force hold  (non-blocking)."""

    def __init__(self, model, data, limits: list[dict],
                 joint_to_actuator: dict[int, int]):
        self._model = model
        self._data = data
        self._limits = limits
        self._j2a = joint_to_actuator
        self._progress = 1.0
        self._total_frames = 1
        self._mode: str | None = None
        self._done = True
        self._contact = False
        self._last_ctrl: dict[int, float] = {}
        self._start_ctrl: dict[int, float] = {}
        self._target_ctrl: dict[int, float] = {}
        self._hold_left = 0
        self._force_ema = 0.0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def contact_triggered(self) -> bool:
        return self._contact

    @property
    def holding(self) -> bool:
        return self._mode == "hold" and not self._done

    def open(self, duration_frames: int = 25) -> None:
        self._begin_motion("open", duration_frames)

    def close(self, duration_frames: int = 100) -> None:
        self._begin_motion("close", duration_frames)

    def _begin_motion(self, mode: str, duration_frames: int) -> None:
        self._mode = mode
        self._progress = 0.0
        self._total_frames = max(duration_frames, 1)
        self._done = False
        self._contact = False
        self._last_ctrl = {}
        self._start_ctrl = {}
        self._target_ctrl = {}
        self._hold_left = 0
        self._force_ema = 0.0
        key = "open" if mode == "open" else "closed"
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is None:
                continue
            lo = min(item["open"], item["closed"])
            hi = max(item["open"], item["closed"])
            start = float(self._data.ctrl[act])
            if not np.isfinite(start):
                start = float(self._data.qpos[self._model.jnt_qposadr[item["joint"]]])
            start = float(np.clip(start, lo, hi))
            target = float(item[key])
            self._start_ctrl[act] = start
            self._target_ctrl[act] = target
            self._last_ctrl[act] = start

    def hold(self, duration_frames: int = GRIP_HOLD_FRAMES) -> None:
        self._mode = "hold"
        self._progress = 1.0
        self._total_frames = 1
        self._done = False
        self._contact = True
        self._hold_left = max(duration_frames, 1)
        self._start_ctrl = {}
        self._target_ctrl = {}
        self._force_ema = self._max_actuator_force()
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is not None:
                self._last_ctrl[act] = self._toward_closed(
                    item, float(self._data.ctrl[act]), GRIP_HOLD_PRELOAD)

    def step(self, force_threshold: float = GRIP_CONTACT_FORCE,
             max_force: float = GRIP_MAX_FORCE) -> None:
        if self._done:
            self._write_final()
            return
        measured_force = self._max_actuator_force()
        if self._mode == "hold":
            self._force_ema = 0.85 * self._force_ema + 0.15 * measured_force
            low_force = max(GRIP_LOCK_MIN_FORCE, force_threshold - GRIP_FORCE_DEADBAND)
            if self._force_ema > max_force + GRIP_ACTUATOR_RELAX_DEADBAND:
                self._relax_hold(GRIP_OVERFORCE_RELAX_CORRECTION)
            elif self._force_ema < low_force and self._hold_left % 4 == 0:
                self._tighten_hold(GRIP_HOLD_CORRECTION)
            self._write_final()
            self._hold_left -= 1
            if self._hold_left <= 0:
                self._done = True
            return
        self._progress += 1.0 / self._total_frames
        reached_end = False
        if self._progress >= 1.0:
            self._progress = 1.0
            self._done = True
            reached_end = True
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is None:
                continue
            start = self._start_ctrl.get(act, float(self._data.ctrl[act]))
            target = self._target_ctrl.get(
                act, item["closed"] if self._mode == "close" else item["open"])
            val = float(start + (target - start) * self._progress)
            self._data.ctrl[act] = val
            self._last_ctrl[act] = val
        if reached_end:
            return

    def _max_actuator_force(self) -> float:
        if not hasattr(self._data, "actuator_force"):
            return 0.0
        max_force = 0.0
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is not None:
                max_force = max(max_force, abs(float(self._data.actuator_force[act])))
        return max_force

    def _toward_closed(self, item: dict, value: float, amount: float) -> float:
        direction = 1.0 if item["closed"] > item["open"] else -1.0
        candidate = value + direction * amount
        lo = min(item["open"], item["closed"])
        hi = max(item["open"], item["closed"])
        return float(np.clip(candidate, lo, hi))

    def _toward_open(self, item: dict, value: float, amount: float) -> float:
        direction = 1.0 if item["open"] > item["closed"] else -1.0
        candidate = value + direction * amount
        lo = min(item["open"], item["closed"])
        hi = max(item["open"], item["closed"])
        return float(np.clip(candidate, lo, hi))

    def maintain_tactile_force(
        self,
        reading: TactileReading,
        min_force: float = SKIN_HOLD_MIN_FORCE,
        target_force: float = SKIN_HOLD_TARGET_FORCE,
        max_force: float = SKIN_HOLD_MAX_FORCE,
        correction_scale: float = 1.0,
    ) -> None:
        if self._mode != "hold" or self._done:
            return
        correction_scale = float(np.clip(correction_scale, 0.05, 1.0))
        left = max(0.0, float(reading.left_force))
        right = max(0.0, float(reading.right_force))
        low_side = min(left, right)
        high_side = max(left, right)
        avg_force = 0.5 * (left + right)
        imbalance = abs(left - right)
        if reading.balance > 0.45 and low_side < target_force:
            weak_side = "left" if left < right else "right"
            strong_side = "right" if weak_side == "left" else "left"
            amount = min(
                GRIP_HOLD_FAST_CORRECTION * correction_scale,
                GRIP_HOLD_CORRECTION *
                correction_scale *
                (1.2 + imbalance / max(target_force, 1e-6)),
            )
            self._nudge_hold_side(weak_side, amount, close=True)
            if high_side > max_force * 0.92:
                self._nudge_hold_side(
                    strong_side,
                    GRIP_HOLD_RELAX_CORRECTION * 3.0 * correction_scale,
                    close=False,
                )
            if low_side >= min_force * 0.65:
                return
        if low_side < min_force:
            deficit = min_force - low_side
            amount = min(
                GRIP_HOLD_FAST_CORRECTION * correction_scale,
                GRIP_HOLD_CORRECTION * correction_scale *
                (1.0 + deficit / max(min_force, 1e-6)),
            )
            self._tighten_hold(amount)
        elif avg_force < target_force:
            self._tighten_hold(GRIP_HOLD_CORRECTION * correction_scale)
        elif avg_force > max_force and reading.balance < 0.45:
            self._relax_hold(GRIP_HOLD_RELAX_CORRECTION * correction_scale)
        if reading.balance > SKIN_BALANCE_MAX_RATIO and low_side < target_force:
            self._tighten_hold(GRIP_HOLD_CORRECTION * correction_scale)

    def _nudge_hold_side(self, side: str, amount: float, *,
                         close: bool) -> None:
        side = side.lower()
        for item in self._limits:
            joint_name = (
                mujoco.mj_id2name(
                    self._model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    item["joint"],
                ) or ""
            ).lower()
            if side not in joint_name:
                continue
            act = self._j2a.get(item["joint"])
            if act is None:
                continue
            val = self._last_ctrl.get(act, self._data.ctrl[act])
            if close:
                self._last_ctrl[act] = self._toward_closed(
                    item, float(val), amount)
            else:
                self._last_ctrl[act] = self._toward_open(
                    item, float(val), amount)

    def compensate_visual_slip(self, weight: float, slip_m: float,
                               drop_m: float) -> None:
        if self._mode != "hold" or self._done:
            return
        weight = float(np.clip(weight, 0.0, 1.0))
        if weight <= 0.05:
            return
        excess = max(
            0.0,
            float(slip_m) - VISUAL_GRIP_ERR_SOFT_M,
            float(drop_m) - VISUAL_GRIP_DROP_SOFT_M,
        )
        amount = min(
            VISUAL_GRIP_POSITION_CORRECTION_MAX,
            GRIP_HOLD_CORRECTION * weight +
            VISUAL_GRIP_POSITION_CORRECTION_GAIN * excess,
        )
        if (float(slip_m) >= VISUAL_GRIP_EMERGENCY_SLIP_M or
                float(drop_m) >= VISUAL_GRIP_EMERGENCY_DROP_M):
            amount = max(amount, VISUAL_GRIP_EMERGENCY_CORRECTION)
        self._tighten_hold(amount)

    def _tighten_hold(self, amount: float) -> None:
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is None:
                continue
            val = self._last_ctrl.get(act, self._data.ctrl[act])
            self._last_ctrl[act] = self._toward_closed(item, float(val), amount)

    def _relax_hold(self, amount: float) -> None:
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is None:
                continue
            val = self._last_ctrl.get(act, self._data.ctrl[act])
            self._last_ctrl[act] = self._toward_open(item, float(val), amount)

    def _write_final(self) -> None:
        if self._last_ctrl:
            for act, val in self._last_ctrl.items():
                self._data.ctrl[act] = val
        else:
            key = "open" if self._mode == "open" else "closed"
            for item in self._limits:
                act = self._j2a.get(item["joint"])
                if act is not None:
                    self._data.ctrl[act] = item[key]


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Workspace sampling
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def compute_workspace(
    model, data, arm_joints: list[int],
    left_body_id: int, right_body_id: int,
    resolution: float = 0.04,
) -> list[np.ndarray]:
    xs = np.arange(-WORKSPACE_R_MAX, WORKSPACE_R_MAX + 0.001, resolution)
    ys = np.arange(-WORKSPACE_R_MAX, WORKSPACE_R_MAX + 0.001, resolution)
    zs = np.arange(0.02, 0.42, resolution)
    xy_grid = [
        (x, y)
        for x in xs
        for y in ys
        if workspace_contains_xy(np.array([x, y], dtype=np.float64), margin=0.0)
    ]
    total = len(xy_grid) * len(zs)
    print(f"Sampling workspace: {len(xy_grid)} xy x {len(zs)} z = {total} pts "
          f"(res={resolution}m) ...")
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos[:]
    scratch.qvel[:] = 0.0
    home = np.zeros(len(arm_joints), dtype=np.float64)
    set_joint_positions(model, scratch, arm_joints, home)
    mujoco.mj_forward(model, scratch)
    reachable: list[np.ndarray] = []
    cnt = 0
    last_pct = -1
    for x, y in xy_grid:
        for z in zs:
            cnt += 1
            pct = cnt * 100 // total
            if pct > last_pct:
                last_pct = pct
                if pct % 10 == 0:
                    print(f"  {pct}% ...")
            result = solve_gripper_center_ik(
                model, scratch, np.array([x, y, z], dtype=np.float64),
                left_body_id, right_body_id, arm_joints,
            )
            if result.success:
                reachable.append(np.array([x, y, z], dtype=np.float64))
                set_joint_positions(model, scratch, arm_joints, result.angles)
                mujoco.mj_forward(model, scratch)
    print(f"Workspace done: {len(reachable)} reachable ({len(reachable)*100/total:.1f}%)")
    if len(reachable) > WORKSPACE_MAX_SPHERES:
        rng = random.Random(42)
        reachable = rng.sample(reachable, WORKSPACE_MAX_SPHERES)
        print(f"  鈫? subsampled to {len(reachable)} for smooth rendering")
    return reachable


def render_workspace_spheres(viewer, reachable: list[np.ndarray]) -> None:
    if not reachable:
        return
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        r = 0.007
        rgba = (0.0, 0.85, 0.25, 0.38)
        for pt in reachable:
            g = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1
            if g >= viewer.user_scn.maxgeom:
                break
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[g],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([r, 0, 0]), pt,
                np.eye(3, 1).flatten(), rgba,
            )


_CUBOID_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_YELLOW_RGBA = np.array([1.0, 0.86, 0.02, 0.92], dtype=np.float32)
_RED_AXIS_RGBA = np.array([1.0, 0.12, 0.08, 0.78], dtype=np.float32)
_GREEN_AXIS_RGBA = np.array([0.10, 0.90, 0.18, 0.78], dtype=np.float32)
_BLUE_AXIS_RGBA = np.array([0.15, 0.40, 1.0, 0.78], dtype=np.float32)


def cuboid_corners(center: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float64)
    h = np.asarray(half_extents, dtype=np.float64)
    return np.array([
        [c[0] - h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] - h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] - h[2]],
        [c[0] - h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] + h[0], c[1] - h[1], c[2] + h[2]],
        [c[0] + h[0], c[1] + h[1], c[2] + h[2]],
        [c[0] - h[0], c[1] + h[1], c[2] + h[2]],
    ], dtype=np.float64)


def _next_user_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    idx = scene.ngeom
    scene.ngeom += 1
    return scene.geoms[idx]


def _add_scene_capsule(scene, start: np.ndarray, end: np.ndarray,
                       rgba: np.ndarray, radius: float = 0.0018) -> None:
    geom = _next_user_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.array([radius, 0.0, 0.0], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )
    geom.rgba[:] = rgba


def _add_scene_dashed_line(scene, start: np.ndarray, end: np.ndarray,
                           rgba: np.ndarray = _YELLOW_RGBA,
                           segments: int = 12) -> None:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    for idx in range(segments):
        if idx % 2:
            continue
        t0 = idx / segments
        t1 = (idx + 0.68) / segments
        p0 = start + (end - start) * t0
        p1 = start + (end - start) * min(t1, 1.0)
        _add_scene_capsule(scene, p0, p1, rgba)


def _add_scene_label(scene, text: str, pos: np.ndarray,
                     rgba: np.ndarray = _YELLOW_RGBA) -> None:
    geom = _next_user_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LABEL,
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    geom.label = text[:120]


def _add_scene_cuboid(scene, center: np.ndarray, half_extents: np.ndarray,
                      label: str) -> None:
    corners = cuboid_corners(center, half_extents)
    for a, b in _CUBOID_EDGES:
        _add_scene_dashed_line(scene, corners[a], corners[b])
    label_pos = np.asarray(center, dtype=np.float64).copy()
    label_pos += np.array([
        half_extents[0] + 0.018,
        half_extents[1] + 0.012,
        half_extents[2] + 0.028,
    ], dtype=np.float64)
    dims_mm = np.asarray(half_extents, dtype=np.float64) * 2000.0
    text = (
        f"{label} base/world xyz=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})m "
        f"LWH=({dims_mm[0]:.0f},{dims_mm[1]:.0f},{dims_mm[2]:.0f})mm"
    )
    _add_scene_label(scene, text, label_pos)


def _add_scene_base_frame_hint(scene) -> None:
    origin = np.array([0.0, 0.0, 0.035], dtype=np.float64)
    _add_scene_capsule(scene, origin, origin + np.array([0.065, 0.0, 0.0]), _RED_AXIS_RGBA)
    _add_scene_capsule(scene, origin, origin + np.array([0.0, 0.065, 0.0]), _GREEN_AXIS_RGBA)
    _add_scene_capsule(scene, origin, origin + np.array([0.0, 0.0, 0.065]), _BLUE_AXIS_RGBA)
    _add_scene_label(
        scene,
        "base/world frame origin: robot base center",
        origin + np.array([0.012, -0.030, 0.078], dtype=np.float64),
    )


def render_scene_annotations(viewer, reachable: list[np.ndarray],
                             cube_xyz: np.ndarray | None,
                             box_xyz: np.ndarray | None) -> None:
    with viewer.lock():
        viewer.user_scn.ngeom = 0
        if reachable:
            r = 0.007
            rgba = np.array([0.0, 0.85, 0.25, 0.30], dtype=np.float32)
            for pt in reachable[:WORKSPACE_MAX_SPHERES]:
                geom = _next_user_geom(viewer.user_scn)
                if geom is None:
                    break
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([r, 0.0, 0.0], dtype=np.float64),
                    np.asarray(pt, dtype=np.float64),
                    np.eye(3, dtype=np.float64).reshape(-1),
                    rgba,
                )
        if cube_xyz is not None or box_xyz is not None:
            _add_scene_base_frame_hint(viewer.user_scn)
        if cube_xyz is not None:
            _add_scene_cuboid(
                viewer.user_scn,
                np.asarray(cube_xyz, dtype=np.float64),
                np.array([CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE],
                         dtype=np.float64),
                "red cube",
            )
        if box_xyz is not None:
            _add_scene_cuboid(
                viewer.user_scn,
                np.asarray(box_xyz, dtype=np.float64),
                BOX_SIZE,
                "blue box",
            )


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Joint-state panel  (OpenCV)
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class JointStatePanel:
    def __init__(self) -> None:
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV required.")
        self._w, self._h = 340, 260
        self._canvas = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        cv2.namedWindow("Joint States", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Joint States", self._w, self._h)
        cv2.moveWindow("Joint States", 880, 40)

    def update(self, model, data,
               arm_joints: list[int], gripper_joints: list[int],
               status_text: str = "") -> None:
        self._canvas[:] = (25, 25, 30)
        y = 20
        if status_text:
            cv2.putText(self._canvas, status_text, (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            y += 22
        cv2.putText(self._canvas, "--- ARM JOINTS ---", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        for jid in arm_joints:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"J{jid}"
            rad = float(data.qpos[model.jnt_qposadr[jid]])
            deg = float(np.degrees(rad))
            cv2.putText(self._canvas,
                        f"  {name:12s} {deg:+8.2f} deg  ({rad:+.4f} rad)",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (0, 220, 255), 1, cv2.LINE_AA)
            y += 18
        y += 4
        cv2.putText(self._canvas, "--- FINGER JOINTS ---", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        y += 18
        for jid in gripper_joints:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"F{jid}"
            rad = float(data.qpos[model.jnt_qposadr[jid]])
            deg = float(np.degrees(rad))
            cv2.putText(self._canvas,
                        f"  {name:12s} {deg:+8.2f} deg  ({rad:+.4f} rad)",
                        (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                        (100, 255, 100), 1, cv2.LINE_AA)
            y += 18

    def show(self) -> None:
        cv2.imshow("Joint States", self._canvas)

    def close(self) -> None:
        if CV2_AVAILABLE:
            cv2.destroyWindow("Joint States")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Scanning poses for cube search
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def _legacy_generate_scan_targets(ball_z: float = 0.05) -> list[np.ndarray]:
    """Return a grid of 3-D points covering the workspace.

    The camera looks down from ~18 cm above each grid point, giving a
    wide-area search.  9 positions cover X 鈭?[0.18, 0.38], Y 鈭?[-0.20, 0.20].
    """
    targets: list[np.ndarray] = []
    z_look = ball_z + 0.18
    for x in [0.38, 0.28, 0.18]:
        for y in [-0.20, 0.0, 0.20]:
            targets.append(np.array([x, y, z_look], dtype=np.float64))
    return targets


@dataclass(frozen=True)
class ScanTarget:
    aim: np.ndarray
    gripper: np.ndarray
    xmat: np.ndarray


def _unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        return vec / norm
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return fallback.astype(np.float64, copy=True)


def _radial_xy(xy: np.ndarray) -> np.ndarray:
    return _unit(np.array([xy[0], xy[1], 0.0], dtype=np.float64),
                 np.array([1.0, 0.0, 0.0], dtype=np.float64))


def _tangent_xy(radial: np.ndarray) -> np.ndarray:
    return _unit(np.array([-radial[1], radial[0], 0.0], dtype=np.float64),
                 np.array([0.0, 1.0, 0.0], dtype=np.float64))


def _nearest_world_opening_axis(hint: np.ndarray) -> np.ndarray:
    hint = np.asarray(hint, dtype=np.float64)
    if hint.size < 2:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(hint[0])) >= abs(float(hint[1])):
        sign = 1.0 if float(hint[0]) >= 0.0 else -1.0
        return np.array([sign, 0.0, 0.0], dtype=np.float64)
    sign = 1.0 if float(hint[1]) >= 0.0 else -1.0
    return np.array([0.0, sign, 0.0], dtype=np.float64)


def workspace_project_xy(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).copy()
    xy[0] = float(np.clip(xy[0], -WORKSPACE_XY_LIMIT, WORKSPACE_XY_LIMIT))
    xy[1] = float(np.clip(xy[1], -WORKSPACE_XY_LIMIT, WORKSPACE_XY_LIMIT))
    r = float(np.linalg.norm(xy))
    if r < 1e-9:
        xy[:] = [WORKSPACE_R_MIN, 0.0]
        return xy
    if r < WORKSPACE_R_MIN:
        xy *= WORKSPACE_R_MIN / r
    elif r > WORKSPACE_R_MAX:
        xy *= WORKSPACE_R_MAX / r
    return xy


def workspace_contains_xy(xy: np.ndarray, margin: float = 0.03) -> bool:
    r = float(np.linalg.norm(np.asarray(xy, dtype=np.float64)))
    return (WORKSPACE_R_MIN - margin) <= r <= (WORKSPACE_R_MAX + margin)


def make_tool_xmat(z_axis: np.ndarray, opening_hint: np.ndarray) -> np.ndarray:
    """Build a right-handed Link6 target orientation."""
    z_axis = _unit(z_axis, np.array([0.0, 0.0, -1.0], dtype=np.float64))
    y_axis = opening_hint - np.dot(opening_hint, z_axis) * z_axis
    y_axis = _unit(y_axis, _tangent_xy(z_axis))
    x_axis = _unit(np.cross(y_axis, z_axis),
                   np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_axis = _unit(np.cross(z_axis, x_axis),
                   np.array([0.0, 1.0, 0.0], dtype=np.float64))
    return np.column_stack((x_axis, y_axis, z_axis))


def camera_forward_world(xmat: np.ndarray) -> np.ndarray:
    return _unit(
        np.asarray(xmat, dtype=np.float64) @ CAMERA_FORWARD_LOCAL,
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )


def gripper_target_for_camera_aim(
    aim: np.ndarray,
    xmat: np.ndarray,
    distance: float = SCAN_CAMERA_DISTANCE,
) -> np.ndarray:
    """Return pinch-center target whose side-mounted D405C optical ray hits aim."""
    xmat = np.asarray(xmat, dtype=np.float64)
    cam_offset_world = xmat @ CAMERA_LOCAL_OFFSET_FROM_PINCH
    return (
        np.asarray(aim, dtype=np.float64) -
        float(distance) * camera_forward_world(xmat) -
        cam_offset_world
    )


def contact_patch_span(points: list[np.ndarray], xmat: np.ndarray | None) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.vstack(points)
    if xmat is None:
        face_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        face_axis = _unit(xmat[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
        approach_axis = _unit(xmat[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
    face_span = float(np.ptp(pts @ face_axis))
    approach_span = float(np.ptp(pts @ approach_axis))
    return max(face_span, approach_span)


def contact_patch_is_large(contact_count: int, patch_span: float) -> bool:
    return (
        contact_count >= TARGET_FINGER_CONTACTS or
        (contact_count >= MIN_STABLE_FINGER_CONTACTS and
         patch_span >= MIN_CONTACT_DIVERSITY)
    )


def contact_alignment_metrics(left_points: list[np.ndarray],
                              right_points: list[np.ndarray],
                              cube_xyz: np.ndarray,
                              xmat: np.ndarray | None) -> tuple[float, float]:
    if not left_points or not right_points:
        return float("inf"), float("inf")
    left_center = np.mean(np.vstack(left_points), axis=0)
    right_center = np.mean(np.vstack(right_points), axis=0)
    if xmat is None:
        opening_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        opening_axis = _unit(xmat[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))
    pair_delta = left_center - right_center
    pair_skew = float(np.linalg.norm(
        pair_delta - np.dot(pair_delta, opening_axis) * opening_axis
    ))
    contact_center = 0.5 * (left_center + right_center)
    center_err = float(np.linalg.norm(contact_center - cube_xyz))
    return pair_skew, center_err


def orientation_error_norm(current_xmat: np.ndarray, target_xmat: np.ndarray) -> float:
    current = np.asarray(current_xmat, dtype=np.float64).reshape(3, 3)
    target = np.asarray(target_xmat, dtype=np.float64).reshape(3, 3)
    err = (
        np.cross(current[:, 0], target[:, 0]) +
        np.cross(current[:, 1], target[:, 1]) +
        np.cross(current[:, 2], target[:, 2])
    )
    return float(np.linalg.norm(err))


def generate_grasp_orientations(
    ball_xyz: np.ndarray,
    opening_hints: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Generate vertical top-down grasps that clamp opposite cube faces."""
    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    base_hints: list[np.ndarray] = []
    base_hints.extend([
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    ])
    if opening_hints:
        # The red square contour is perspective-dependent in the wrist camera.
        # Use it only after snapping to world X/Y.  Diagonal opening axes clamp
        # two cube edges instead of two opposite faces and are deliberately
        # excluded from the first grasp.
        for hint in opening_hints:
            base_hints.append(_nearest_world_opening_axis(hint))

    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    for hint in base_hints:
        xmat = make_tool_xmat(z_axis, hint)
        key = tuple(np.round(xmat.reshape(-1), 4))
        if key not in seen:
            seen.add(key)
            candidates.append(xmat)
    return candidates


def grasp_transport_alignment(
    xmat: np.ndarray,
    ball_xyz: np.ndarray,
    box_xyz: np.ndarray | None,
) -> float:
    """How well the jaw normal resists the carry direction in XY."""
    if box_xyz is None:
        return 1.0
    transport_xy = np.asarray(box_xyz[:2], dtype=np.float64) - np.asarray(
        ball_xyz[:2], dtype=np.float64)
    norm = float(np.linalg.norm(transport_xy))
    if norm <= 1e-6:
        return 1.0
    transport_dir = transport_xy / norm
    y_axis = np.asarray(xmat[:, 1], dtype=np.float64)
    return abs(float(np.dot(y_axis[:2], transport_dir)))


def grasp_stability_score(
    xmat: np.ndarray,
    ball_xyz: np.ndarray,
    box_xyz: np.ndarray | None = None,
) -> float:
    """Prefer a real parallel-jaw top grasp.

    The previous score preferred a large radial tilt.  That can still lift in
    MuJoCo, but it reaches the cube by sweeping a finger through the side face,
    which looks like penetration.  A cube resting on a table should be grasped
    by moving the open jaws down around it, then closing horizontally.
    """
    radial = _radial_xy(ball_xyz[:2])
    tangent = _tangent_xy(radial)
    z_axis = xmat[:, 2]
    y_axis = xmat[:, 1]
    horizontal_tilt = float(np.linalg.norm(z_axis[:2]))
    downward = max(0.0, -float(z_axis[2]))
    radial_opening = abs(float(np.dot(y_axis, radial)))
    tangent_opening_signed = float(np.dot(y_axis, tangent))
    tangent_opening = abs(tangent_opening_signed)
    world_axis_alignment = max(abs(float(np.dot(y_axis, np.array([1.0, 0.0, 0.0])))),
                               abs(float(np.dot(y_axis, np.array([0.0, 1.0, 0.0])))))
    diagonal_penalty = GRASP_DIAGONAL_OPENING_WEIGHT * max(
        0.0, 0.995 - world_axis_alignment)
    transport_penalty = 0.0
    if box_xyz is not None:
        transport_alignment = grasp_transport_alignment(xmat, ball_xyz, box_xyz)
        # Prefer a jaw normal with enough component along the carry direction
        # so the thin electronic-skin surfaces can preload the cube before
        # lateral motion, while the physical preview still rejects bad poses.
        transport_penalty = (
            GRASP_TRANSPORT_AXIS_WEIGHT *
            max(0.0, GRASP_TRANSPORT_TARGET_ALIGNMENT - transport_alignment)
        )
    return (
        80.0 * horizontal_tilt +
        20.0 * max(0.0, 0.999 - downward) +
        GRASP_WORLD_AXIS_WEIGHT * max(0.0, 0.98 - world_axis_alignment) +
        diagonal_penalty +
        transport_penalty +
        0.15 * radial_opening +
        0.10 * max(0.0, -tangent_opening_signed) +
        0.08 * max(0.0, 0.75 - tangent_opening)
    )


def grasp_opening_hint_score(
    xmat: np.ndarray,
    opening_hints: list[np.ndarray] | None,
) -> float:
    """Prefer the snapped RGB-D face normal when it is available."""
    if not opening_hints:
        return 0.0
    y_axis = _unit(xmat[:, 1], np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_xy = _unit(y_axis[:2], np.array([1.0, 0.0], dtype=np.float64))
    best = 0.0
    for hint in opening_hints:
        hint_axis = _nearest_world_opening_axis(hint)
        hint_xy = _unit(hint_axis[:2], np.array([1.0, 0.0], dtype=np.float64))
        best = max(best, abs(float(np.dot(y_xy, hint_xy))))
    return GRASP_OPENING_HINT_WEIGHT * max(
        0.0,
        GRASP_OPENING_HINT_ALIGNMENT - best,
    )


def expand_face_opening_hints(
    opening_hints: list[np.ndarray] | None,
) -> list[np.ndarray]:
    """Keep face-centered grasps while allowing the orthogonal cube-face pair."""
    if not opening_hints:
        return []
    expanded: list[np.ndarray] = []
    seen: set[tuple[float, float, float]] = set()

    def add(axis: np.ndarray) -> None:
        axis = _nearest_world_opening_axis(axis)
        key = tuple(float(v) for v in axis)
        if key not in seen:
            seen.add(key)
            expanded.append(axis)

    for hint in opening_hints:
        axis = _nearest_world_opening_axis(hint)
        add(axis)
        add(-axis)
        perp = np.array([-axis[1], axis[0], 0.0], dtype=np.float64)
        add(perp)
        add(-perp)
    return expanded


def carry_height_for_xy(target_xy: np.ndarray, box_xyz: np.ndarray) -> float:
    """Keep most transport low; raise only near the open box."""
    target_xy = np.asarray(target_xy, dtype=np.float64)
    box_xyz = np.asarray(box_xyz, dtype=np.float64)
    dist_to_box = float(np.linalg.norm(target_xy[:2] - box_xyz[:2]))
    if dist_to_box <= CARRY_FINAL_RAISE_XY_M:
        return float(box_xyz[2] + PLACE_ABOVE_HEIGHT)
    return float(max(CUBE_REST_Z + CARRY_CRUISE_HEIGHT, box_xyz[2] + 0.025))


def carry_midpoint_for_grasp(cube_xyz: np.ndarray,
                             box_xyz: np.ndarray,
                             xmat: np.ndarray | None) -> np.ndarray:
    """Stage transport without combining a large lift and a large lateral move."""
    cube_xyz = np.asarray(cube_xyz, dtype=np.float64)
    box_xyz = np.asarray(box_xyz, dtype=np.float64)
    if xmat is None:
        xy = 0.5 * (cube_xyz[:2] + box_xyz[:2])
    else:
        opening_axis = _unit(
            np.asarray(xmat, dtype=np.float64)[:, 1],
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        opening_axis[2] = 0.0
        opening_axis = _unit(
            opening_axis,
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        delta = box_xyz[:2] - cube_xyz[:2]
        normal_part = float(np.dot(delta, opening_axis[:2])) * opening_axis[:2]
        tangent_part = delta - normal_part
        xy = cube_xyz[:2] + 0.60 * normal_part + 0.55 * tangent_part
        if float(np.linalg.norm(xy - cube_xyz[:2])) < 0.025:
            xy = cube_xyz[:2] + 0.35 * delta
        first_step = xy - cube_xyz[:2]
        first_step_len = float(np.linalg.norm(first_step))
        if first_step_len > CARRY_STEP_MAX_XY_M:
            xy = cube_xyz[:2] + first_step / first_step_len * CARRY_STEP_MAX_XY_M
    carry_z = carry_height_for_xy(xy, box_xyz)
    return np.array([
        xy[0],
        xy[1],
        carry_z,
    ], dtype=np.float64)


def generate_scan_targets(
    ball_z: float = CUBE_REST_Z,
    hint_xy: np.ndarray | None = None,
) -> list[ScanTarget]:
    """Return camera poses whose optical axis intersects workspace cells.

    The first pass is a polar scan around the robot base.  IK decides which
    sectors are actually reachable, so the search is not hard-coded to the
    positive-X rectangle.
    """
    z_plane = max(float(ball_z), CUBE_REST_Z)
    grid: list[np.ndarray] = []
    radii = [0.16, 0.22, 0.28, 0.34, 0.40, 0.46]
    angles = list(range(0, 360, 20))
    for ri, radius in enumerate(radii):
        angle_iter = angles if ri % 2 == 0 else list(reversed(angles))
        for deg in angle_iter:
            theta = math.radians(deg)
            grid.append(np.array(
                [radius * math.cos(theta), radius * math.sin(theta)],
                dtype=np.float64,
            ))

    if hint_xy is not None:
        hint = workspace_project_xy(hint_xy)
        grid.insert(0, hint)

    targets: list[ScanTarget] = []
    seen: set[tuple[float, ...]] = set()

    def add_target(aim: np.ndarray, xmat: np.ndarray) -> None:
        gripper = gripper_target_for_camera_aim(aim, xmat)
        key = tuple(np.round(np.concatenate((aim, gripper, xmat.reshape(-1))), 4))
        if key not in seen:
            seen.add(key)
            targets.append(ScanTarget(aim=aim, gripper=gripper, xmat=xmat))

    # Full-workspace polar pass.  The D405C is side-mounted on Link6, so yaw is
    # chosen per workspace cell and the scan target uses the real camera
    # extrinsic instead of assuming the optical axis equals the tool z-axis.
    for xy in grid:
        aim = np.array([xy[0], xy[1], z_plane], dtype=np.float64)
        radial = _radial_xy(xy)
        tangent = _tangent_xy(radial)
        for opening in (tangent, -tangent):
            add_target(aim, make_tool_xmat(
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                opening,
            ))

    # Secondary angled pass for edge views, kept after the raster pass so a
    # stale hint cannot trap the search in a small local loop.
    for xy in grid:
        aim = np.array([xy[0], xy[1], z_plane], dtype=np.float64)
        radial = _radial_xy(xy)
        tangent = _tangent_xy(radial)
        for tilt_deg in [18.0, 34.0, 50.0]:
            tilt = math.radians(tilt_deg)
            z_axis = _unit(
                math.sin(tilt) * radial + np.array([0.0, 0.0, -math.cos(tilt)]),
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
            )
            add_target(aim, make_tool_xmat(z_axis, tangent))
    return targets


def generate_box_scan_targets(
    hint_xy: np.ndarray | None = None,
    z_plane: float = CUBE_REST_Z,
) -> list[ScanTarget]:
    """Prioritize views likely to see a complete blue box opening."""
    targets: list[ScanTarget] = []
    seen: set[tuple[float, ...]] = set()

    def add_target(aim_xy: np.ndarray, opening_hint: np.ndarray) -> None:
        aim_xy = workspace_project_xy(np.asarray(aim_xy, dtype=np.float64))
        aim = np.array([aim_xy[0], aim_xy[1], max(float(z_plane), CUBE_REST_Z)],
                       dtype=np.float64)
        xmat = make_tool_xmat(np.array([0.0, 0.0, -1.0], dtype=np.float64),
                              opening_hint)
        gripper = gripper_target_for_camera_aim(aim, xmat)
        key = tuple(np.round(np.concatenate((aim, gripper, xmat.reshape(-1))), 4))
        if key not in seen:
            seen.add(key)
            targets.append(ScanTarget(aim=aim, gripper=gripper, xmat=xmat))

    def add_views_around(center_xy: np.ndarray) -> None:
        center_xy = workspace_project_xy(np.asarray(center_xy, dtype=np.float64))
        offsets = [
            np.array([0.0, 0.0], dtype=np.float64),
            np.array([-0.10, 0.03], dtype=np.float64),
            np.array([-0.105, 0.045], dtype=np.float64),
            np.array([-0.12, 0.045], dtype=np.float64),
            np.array([-0.10, 0.06], dtype=np.float64),
            np.array([-0.12, 0.06], dtype=np.float64),
            np.array([-0.10, -0.03], dtype=np.float64),
            np.array([-0.14, 0.0], dtype=np.float64),
            np.array([-0.14, 0.045], dtype=np.float64),
            np.array([0.0, 0.10], dtype=np.float64),
            np.array([0.0, -0.10], dtype=np.float64),
            np.array([0.10, 0.0], dtype=np.float64),
            np.array([-0.07, 0.09], dtype=np.float64),
            np.array([-0.07, -0.09], dtype=np.float64),
        ]
        for off in offsets:
            aim_xy = workspace_project_xy(center_xy + off)
            radial = _radial_xy(aim_xy)
            tangent = _tangent_xy(radial)
            for opening in (tangent, -tangent):
                add_target(aim_xy, opening)

    if hint_xy is not None:
        add_views_around(hint_xy)

    # Coarse fallback.  It is still fully autonomous, but starts with broad
    # useful box views instead of waiting through the full cube polar scan.
    for radius in (0.28, 0.38, 0.46, 0.20):
        for deg in (-20, -45, 0, 25, -70, 55, 90, -110, 140, 180):
            theta = math.radians(deg)
            xy = np.array([radius * math.cos(theta),
                           radius * math.sin(theta)], dtype=np.float64)
            radial = _radial_xy(xy)
            tangent = _tangent_xy(radial)
            for opening in (tangent, -tangent):
                add_target(xy, opening)
    return targets


def draw_detection_overlay(rgb: np.ndarray, found: bool,
                           xyz: np.ndarray | None = None) -> np.ndarray:
    """Draw 鉁?(found) or 鉁?(not found) + position on an RGB image."""
    h, w = rgb.shape[:2]
    overlay = rgb.copy()
    if found and xyz is not None:
        cv2.putText(overlay, "V", (w - 50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, f"X:{xyz[0]:.3f}", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Y:{xyz[1]:.3f}", (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        cv2.drawMarker(overlay, (w - 35, 25), (0, 0, 255),
                       cv2.MARKER_TILTED_CROSS, 20, 2, cv2.LINE_AA)
    return overlay


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Main
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
def project_world_to_rgbd_pixel(data, camera_id: int, xyz: np.ndarray,
                                img_w: int, img_h: int) -> tuple[int, int] | None:
    if camera_id < 0:
        return None
    cam_pos = data.cam_xpos[camera_id].copy()
    cam_mat = data.cam_xmat[camera_id].reshape(3, 3)
    local = cam_mat.T @ (np.asarray(xyz, dtype=np.float64) - cam_pos)
    depth = -float(local[2])
    if depth <= MIN_RGBD_DEPTH_M or depth >= MAX_RGBD_DEPTH_M:
        return None
    fx = (img_w / 2.0) / math.tan(math.radians(D405C_DEPTH_FOVX_DEG) / 2.0)
    fy = (img_h / 2.0) / math.tan(math.radians(D405C_DEPTH_FOVY_DEG) / 2.0)
    u = int(round(img_w / 2.0 + float(local[0]) * fx / depth))
    v = int(round(img_h / 2.0 - float(local[1]) * fy / depth))
    if u < -img_w or u > img_w * 2 or v < -img_h or v > img_h * 2:
        return None
    return u, v


def draw_image_dashed_line(img: np.ndarray,
                           p0: tuple[int, int] | None,
                           p1: tuple[int, int] | None,
                           color_rgb: tuple[int, int, int] = (255, 255, 0),
                           segments: int = 12) -> None:
    if p0 is None or p1 is None:
        return
    a = np.array(p0, dtype=np.float64)
    b = np.array(p1, dtype=np.float64)
    for idx in range(segments):
        if idx % 2:
            continue
        q0 = a + (b - a) * (idx / segments)
        q1 = a + (b - a) * min((idx + 0.68) / segments, 1.0)
        cv2.line(
            img,
            tuple(np.round(q0).astype(int)),
            tuple(np.round(q1).astype(int)),
            color_rgb,
            2,
            cv2.LINE_AA,
        )


def draw_projected_cuboid_overlay(rgb: np.ndarray, data, camera_id: int,
                                  center: np.ndarray,
                                  half_extents: np.ndarray,
                                  label: str) -> np.ndarray:
    if not CV2_AVAILABLE:
        return rgb
    h, w = rgb.shape[:2]
    center = np.asarray(center, dtype=np.float64)
    half_extents = np.asarray(half_extents, dtype=np.float64)
    corners = cuboid_corners(center, half_extents)
    pixels = [
        project_world_to_rgbd_pixel(data, camera_id, corner, w, h)
        for corner in corners
    ]
    for a, b in _CUBOID_EDGES:
        draw_image_dashed_line(rgb, pixels[a], pixels[b])

    label_anchor = center + np.array([
        half_extents[0] + 0.018,
        half_extents[1] + 0.012,
        half_extents[2] + 0.022,
    ], dtype=np.float64)
    label_uv = project_world_to_rgbd_pixel(data, camera_id, label_anchor, w, h)
    if label_uv is None:
        label_uv = project_world_to_rgbd_pixel(data, camera_id, center, w, h)
    if label_uv is not None:
        u = int(np.clip(label_uv[0], 6, max(6, w - 220)))
        v = int(np.clip(label_uv[1], 34, max(34, h - 46)))
        dims_mm = half_extents * 2000.0
        lines = [
            f"{label} base/world",
            f"xyz=({center[0]:.3f},{center[1]:.3f},{center[2]:.3f})m",
            f"LWH=({dims_mm[0]:.0f},{dims_mm[1]:.0f},{dims_mm[2]:.0f})mm",
        ]
        for row, text in enumerate(lines):
            cv2.putText(
                rgb,
                text,
                (u, v + row * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return rgb


def main() -> None:
    print("=" * 62)
    print("Interactive Grasping Demo")
    print("- Use MuJoCo sliders to pose the arm freely before starting")
    print("- Use 'cube_x / cube_y / cube_z' sliders to move the target cube")
    print("- Slide 'start_demo' to 1  鈫? autonomous detection & grasping")
    print("=" * 62)

    # 鈹€鈹€ Model loading with box injection 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    import synriard
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)
    model_path = synriard.get_model_path(
        "Alicia_D", version="v5_6", variant="gripper_50mm", model_format="mjcf",
    )
    with open(model_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    xml_content = inject_target_box(xml_content)

    # Re-use model_loader injections, but bypass load_and_inject to add box
    from common.model_loader import (
        inject_options, inject_overview_camera,
        inject_wrist_camera, inject_actuators,
        SIM_HZ,
    )
    xml_content = inject_options(xml_content)
    xml_content = inject_overview_camera(xml_content)
    # Large high-friction pad covering the 360-degree search workspace.
    if 'name="friction_pad"' not in xml_content:
        pad_xml = f"""
        <body name="friction_pad" pos="0.0 0.0 0.0">
            <geom name="pad_geom" type="box" size="{WORKSPACE_XY_LIMIT} {WORKSPACE_XY_LIMIT} {PAD_HALF_HEIGHT}"
                  rgba="0.4 0.4 0.4 0.5" friction="8.0 5.0 0.8"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", pad_xml + "\n  </worldbody>", 1)

    # Custom cube: 40 mm side, 60 g, with high grip friction
    if 'name="target_cube"' not in xml_content:
        custom_cube_xml = f"""
        <body name="target_cube" pos="{BALL_POS[0]} {BALL_POS[1]} {CUBE_REST_Z}">
            <freejoint/>
            <geom name="cube_geom" type="box" size="{CUBE_HALF_SIZE} {CUBE_HALF_SIZE} {CUBE_HALF_SIZE}"
                  rgba="1 0.3 0.3 0.9" mass="0.060" condim="6"
                  friction="65.0 28.0 10.0" solimp="0.97 0.995 0.0005"
                  solref="0.010 1"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", custom_cube_xml + "\n  </worldbody>")
    xml_content = inject_wrist_camera(xml_content)
    # (inject_soft_ball is skipped because target_cube already exists)
    # Configure the real finger mesh and hide the old visible marker only after
    # wrist camera insertion, because that helper uses the marker as its XML
    # anchor.
    xml_content = configure_finger_mesh_collision(xml_content)
    xml_content = inject_finger_skin_sensors(xml_content)
    xml_content = inject_actuators(xml_content)
    # Must be AFTER inject_actuators 鈥?inserts into the existing <actuator>
    xml_content = inject_control_actuators(xml_content)

    # 鈹€鈹€ Boost PD gains for precise positioning under gravity 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    xml_content = xml_content.replace('kp="90"', 'kp="350"')
    xml_content = xml_content.replace('kp="70"', 'kp="250"')
    xml_content = xml_content.replace('kp="45"', 'kp="180"')
    # Stronger real gripper forces.  The default MJCF limits are too weak for
    # a 40 mm cube, but excessive force/servo stiffness causes stick-slip
    # chatter in the contact solver.  Use a moderately stronger, well-damped
    # finger servo so real two-sided contact has enough normal force.
    xml_content = xml_content.replace(
        '<joint name="left_finger" pos="0 0 0" axis="0 0 1" type="slide" range="-0.025 0" actuatorfrcrange="-5 5" />',
        '<joint name="left_finger" pos="0 0 0" axis="0 0 1" type="slide" range="-0.032 0.007" actuatorfrcrange="-180 180" />',
    )
    xml_content = xml_content.replace(
        '<joint name="right_finger" pos="0 0 0" axis="0 0 -1" type="slide" range="0 0.025" actuatorfrcrange="-5 5" />',
        '<joint name="right_finger" pos="0 0 0" axis="0 0 -1" type="slide" range="-0.007 0.032" actuatorfrcrange="-180 180" />',
    )
    xml_content = xml_content.replace(
        '<position name="left_finger_act" joint="left_finger" kp="55" kv="7"\n              forcerange="-8 8"/>',
        '<position name="left_finger_act" joint="left_finger" kp="900" kv="190"\n              forcerange="-180 180"/>',
    )
    xml_content = xml_content.replace(
        '<position name="right_finger_act" joint="right_finger" kp="55" kv="7"\n              forcerange="-8 8"/>',
        '<position name="right_finger_act" joint="right_finger" kp="900" kv="190"\n              forcerange="-180 180"/>',
    )

    xml_dir = os.path.dirname(model_path)
    os.chdir(xml_dir)
    model = mujoco.MjModel.from_xml_string(xml_content)
    os.chdir(repo_root)

    data = mujoco.MjData(model)
    model.opt.timestep = 1.0 / SIM_HZ

    # Joint classification
    arm_joints: list[int] = []
    gripper_joints: list[int] = []
    for jid in range(model.njnt):
        jt = model.jnt_type[jid]
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if "finger" in name.lower():
                gripper_joints.append(jid)
            else:
                arm_joints.append(jid)

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link7")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link8")
    tool_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link6")
    left_finger_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_collision")
    right_finger_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_collision")
    left_skin_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "left_inner_skin_site")
    right_skin_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_inner_skin_site")
    pinch_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper_pinch_center_body")
    if left_id < 0 or right_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger bodies.")
    if tool_id < 0:
        raise RuntimeError("Could not find Link6 tool body.")
    if left_finger_geom_id < 0 or right_finger_geom_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger mesh collision geoms.")
    if left_skin_site_id < 0 or right_skin_site_id < 0:
        raise RuntimeError("Could not find inner tactile skin sites.")
    if pinch_body_id < 0:
        raise RuntimeError("Could not find gripper_pinch_center_body.")

    joint_to_actuator: dict[int, int] = {}
    for act_id in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id) or ""
        joint_name = act_name[:-4] if act_name.endswith("_act") else act_name
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id >= 0:
            joint_to_actuator[int(jnt_id)] = int(act_id)

    mujoco.mj_forward(model, data)

    # 鈹€鈹€ Initial setup 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    gripper_limits = build_gripper_limits(model, data, gripper_joints)
    command_gripper(model, data, gripper_limits, joint_to_actuator, "open")
    for jid, act in joint_to_actuator.items():
        data.ctrl[act] = data.qpos[model.jnt_qposadr[jid]]
    mujoco.mj_forward(model, data)

    finger_body_mid = (data.xpos[left_id] + data.xpos[right_id]) * 0.5
    print(f"Real gripper pinch center: {np.round(data.xpos[pinch_body_id], 4)} "
          f"finger_body_mid={np.round(finger_body_mid, 4)}")

    # 鈹€鈹€ Workspace 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    show_workspace = os.environ.get("SYNRIA_SHOW_WORKSPACE", "").lower() in {
        "1", "true", "yes", "on",
    }
    if show_workspace:
        workspace_pts = compute_workspace(
            model, data, arm_joints, left_id, right_id, resolution=0.08)
    else:
        workspace_pts = []
        print("Workspace overlay skipped (set SYNRIA_SHOW_WORKSPACE=1 to draw it).")

    # 鈹€鈹€ Controllers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    arm_ctrl = SmoothArmController(model, data, arm_joints, joint_to_actuator)
    gripper_ctrl = SmoothGripperController(model, data, gripper_limits, joint_to_actuator)

    # 鈹€鈹€ Cube detector + interactive placement 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rgb_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA_NAME)
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if ball_body_id < 0 or cube_geom_id < 0:
        raise RuntimeError("Could not find target_cube / cube_geom.")
    if rgb_cam_id >= 0:
        cam_to_pinch = float(np.linalg.norm(data.cam_xpos[rgb_cam_id] - data.xpos[pinch_body_id]))
        nearest_gripped_face = max(0.0, cam_to_pinch - CUBE_HALF_SIZE)
        print(f"D405C wrist camera standoff: center={cam_to_pinch*100:.1f}cm "
              f"nearest_cube_face={nearest_gripped_face*100:.1f}cm "
              f"valid_depth={MIN_RGBD_DEPTH_M*100:.1f}-{MAX_RGBD_DEPTH_M*100:.0f}cm")
    detector = BallDetector(model, rgb_cam_id, ball_body_id)
    box_detector = TargetBoxDetector(model, rgb_cam_id)

    # Find the cube's freejoint qpos address (for slider placement)
    ball_qpos_adr = -1
    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == ball_body_id:
            ball_qpos_adr = model.jnt_qposadr[jid]
            break

    # Find control actuator IDs (dummy sliders in the viewer's control panel)
    start_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "start_demo")
    cube_x_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cube_x")
    cube_y_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cube_y")
    cube_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "cube_z")
    # Initialise cube sliders to the default cube position
    _last_bx = float(BALL_POS[0])
    _last_by = float(BALL_POS[1])
    _last_bz = float(CUBE_REST_Z)
    auto_cube_xyz: np.ndarray | None = None
    auto_cube_env = (
        os.environ.get("SYNRIA_CUBE_X"),
        os.environ.get("SYNRIA_CUBE_Y"),
        os.environ.get("SYNRIA_CUBE_Z"),
    )
    if any(value is not None for value in auto_cube_env):
        try:
            auto_cube_xyz = np.array([
                float(auto_cube_env[0]) if auto_cube_env[0] is not None else _last_bx,
                float(auto_cube_env[1]) if auto_cube_env[1] is not None else _last_by,
                float(auto_cube_env[2]) if auto_cube_env[2] is not None else _last_bz,
            ], dtype=np.float64)
            auto_cube_xyz[:2] = workspace_project_xy(auto_cube_xyz[:2])
            auto_cube_xyz[2] = float(np.clip(auto_cube_xyz[2], CUBE_REST_Z, 0.30))
        except ValueError:
            print(f"WARNING: ignoring invalid SYNRIA_CUBE_X/Y/Z={auto_cube_env}")
            auto_cube_xyz = None
    if start_act_id >= 0:
        data.ctrl[start_act_id] = 0.0
    if cube_x_act_id >= 0:
        data.ctrl[cube_x_act_id] = _last_bx
        data.ctrl[cube_y_act_id] = _last_by
        data.ctrl[cube_z_act_id] = _last_bz
    # Freejoint qvel  鈮? qpos address  (qpos=7, qvel=6 elements)
    ball_dof_adr = -1
    for jid in range(model.njnt):
        if model.jnt_bodyid[jid] == ball_body_id:
            ball_dof_adr = model.jnt_dofadr[jid]
            break

    def current_ball_xyz() -> np.ndarray:
        """Read the cube's current freejoint position, not the cached body xpos."""
        if ball_qpos_adr >= 0:
            return np.array([
                float(data.qpos[ball_qpos_adr]),
                float(data.qpos[ball_qpos_adr + 1]),
                max(float(data.qpos[ball_qpos_adr + 2]), CUBE_REST_Z),
            ], dtype=np.float64)
        mujoco.mj_forward(model, data)
        xyz = data.xpos[ball_body_id].copy()
        xyz[2] = max(float(xyz[2]), CUBE_REST_Z)
        return xyz

    # 鈹€鈹€ Gripper tactile skin 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    skin_sensor = TactileSkinSensor(
        model,
        filter_alpha=SKIN_FILTER_ALPHA,
        force_min=SKIN_FORCE_RANGE_MIN_N,
        force_max=SKIN_FORCE_RANGE_MAX_N,
        resolution=SKIN_FORCE_RESOLUTION_N,
        recognition_threshold=SKIN_FORCE_MIN_RECOGNITION_N,
    )
    skin_display = TactileSkinDisplay(width=500, height=260, history_len=180)

    # 鈹€鈹€ Eye-in-hand camera 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rgbd_window = None
    if rgb_cam_id >= 0:
        try:
            rgbd_window = RGBDCameraWindow(
                model, rgb_cam_id, width=D405C_DEPTH_WIDTH, height=D405C_DEPTH_HEIGHT,
                render_every_n=CAMERA_EVERY_N,
                window_name="Eye-in-Hand RGB-D Camera")
            print(
                "Eye-in-hand RGB-D camera: active "
                f"({D405C_DEPTH_WIDTH}x{D405C_DEPTH_HEIGHT} color + depth, "
                f"{D405C_DEPTH_FOVX_DEG:.0f}x{D405C_DEPTH_FOVY_DEG:.0f} deg FOV, "
                f"{MIN_RGBD_DEPTH_M:.2f}-{MAX_RGBD_DEPTH_M:.2f}m depth gate)"
            )
        except Exception as exc:
            print(f"Camera init: {exc}")

    # 鈹€鈹€ Joint state panel 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    joint_panel = None
    if CV2_AVAILABLE:
        try:
            joint_panel = JointStatePanel()
            print("Joint-state panel: active")
        except Exception as exc:
            print(f"Joint panel init: {exc}")

    # 鈹€鈹€ State machine 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    # -1 = idle      0 = scan   1 = reserved 2 = approach   3 = descend
    #  4 = close     5 = verify 6 = move    7 = release    8 = re-grasp
    #  9 = done
    phase = -1
    sub = 0
    finished = False
    scan_idx = 0
    scan_round = 0
    scan_targets: list[ScanTarget] = []
    box_scan_idx = 0
    box_scan_targets: list[ScanTarget] = []
    box_search_hint_xy: np.ndarray | None = None
    box_scan_hint_xy_used: np.ndarray | None = None
    detected_ball_pos: np.ndarray | None = None
    detected_box_pos: np.ndarray | None = None
    detected_opening_hints: list[np.ndarray] = []
    dynamic_ik_plan: dict[str, IKResult] = {}
    dynamic_grasp_xmat: np.ndarray | None = None
    dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
    rejected_grasp_frames: list[tuple[np.ndarray, np.ndarray]] = []
    rgbd_scan_estimates: list[np.ndarray] = []
    rgbd_scan_hints: list[np.ndarray] = []
    box_rgbd_scan_estimates: list[np.ndarray] = []
    last_scan_rgbd_sample_frame = -SCAN_RGBD_CAPTURE_EVERY_N
    carried_cube_offset: np.ndarray | None = None
    visual_grip_reference_offset: np.ndarray | None = None
    visual_grip_filtered_xyz: np.ndarray | None = None
    visual_grip_last_time = 0.0
    visual_grip_slip_m = 0.0
    visual_grip_slip_rate_mps = 0.0
    visual_grip_drop_m = 0.0
    visual_grip_weight = 0.0
    visual_grip_miss_count = 0
    last_visual_grip_frame = -VISUAL_GRIP_CAPTURE_EVERY_N
    last_visual_grip_log_frame = -VISUAL_GRIP_LOG_EVERY_N
    last_visual_secure_override_frame = -TACTILE_SECURE_LOG_EVERY_N
    transport_slip_abort_frames = 0
    transport_regrip_count = 0
    transport_regrip_frames = 0
    transport_regrip_stable_frames = 0
    transport_regrip_resume_sub = 2
    regrasp_count = 0
    MAX_REGRASP = 4
    pregrasp_replan_count = 0
    local_replan_count = 0
    grip_contact_hold = 0
    grip_close_frames = 0
    gripper_open_wait_frames = 0
    preclose_stable_frames = 0
    place_settle_frames = 0
    place_correction_count = 0
    place_support_rescan_count = 0
    place_release_support_override = False
    prelift_lost_frames = 0
    lift_motion_frames = 0
    carry_motion_frames = 0
    carry_last_remaining_xy = float("inf")
    carry_stall_steps = 0
    prelift_hold_frames = 0
    prelift_ready_frames = 0
    ball_z_before_lift = 0.0
    status_msg = "IDLE 鈥?use cube_x/y/z & start_demo sliders"
    box_verify_left = 0
    recovery_wait_frames = 0
    recovery_stable_frames = 0
    recovery_reason = ""
    grip_locked = False
    grip_lock_offset = np.zeros(3, dtype=np.float64)
    grip_lock_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cube_static_anchor_active = False
    cube_static_pos = np.zeros(3, dtype=np.float64)
    cube_static_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def sync_cube_sliders(xyz: np.ndarray) -> None:
        nonlocal _last_bx, _last_by, _last_bz
        _last_bx = float(xyz[0])
        _last_by = float(xyz[1])
        _last_bz = max(float(xyz[2]), CUBE_REST_Z)
        if cube_x_act_id >= 0:
            data.ctrl[cube_x_act_id] = _last_bx
            data.ctrl[cube_y_act_id] = _last_by
            data.ctrl[cube_z_act_id] = _last_bz

    def zero_cube_velocity() -> None:
        if ball_dof_adr >= 0:
            data.qvel[ball_dof_adr:ball_dof_adr + 6] = 0.0

    def set_cube_xyz(xyz: np.ndarray, *, sync_sliders: bool = False) -> None:
        if ball_qpos_adr < 0:
            return
        xyz = np.asarray(xyz, dtype=np.float64).copy()
        xyz[:2] = workspace_project_xy(xyz[:2])
        xyz[2] = max(float(xyz[2]), CUBE_REST_Z)
        data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = xyz
        zero_cube_velocity()
        mujoco.mj_forward(model, data)
        if sync_sliders:
            sync_cube_sliders(xyz)

    if auto_cube_xyz is not None:
        set_cube_xyz(auto_cube_xyz, sync_sliders=True)
        print(f"Auto cube initial position from env: {np.round(current_ball_xyz(), 4)}")

    def current_cube_quat() -> np.ndarray:
        if ball_qpos_adr < 0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        quat = data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7].copy()
        norm = float(np.linalg.norm(quat))
        if norm < 1e-6:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    def set_cube_static_anchor(context: str) -> None:
        nonlocal cube_static_anchor_active, cube_static_pos, cube_static_quat
        if ball_qpos_adr < 0:
            cube_static_anchor_active = False
            return
        candidate = current_ball_xyz()
        if candidate[2] > CUBE_STATIC_ANCHOR_MAX_Z:
            cube_static_anchor_active = False
            print(f"    Static cube anchor refused ({context}): "
                  f"cube_z={candidate[2]:.3f}m is above "
                  f"max_anchor_z={CUBE_STATIC_ANCHOR_MAX_Z:.3f}m")
            return
        cube_static_anchor_active = True
        cube_static_pos = candidate.copy()
        cube_static_quat = current_cube_quat()
        data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube_static_pos
        data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = cube_static_quat
        zero_cube_velocity()
        mujoco.mj_forward(model, data)

    def release_cube_static_anchor(context: str) -> None:
        nonlocal cube_static_anchor_active
        cube_static_anchor_active = False

    def release_cube_for_soft_close(context: str) -> None:
        if ball_qpos_adr < 0:
            release_cube_static_anchor(context)
            return
        if cube_static_anchor_active:
            data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube_static_pos
            data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = cube_static_quat
        zero_cube_velocity()
        mujoco.mj_forward(model, data)
        if cube_static_anchor_active:
            print(f"    Static cube anchor released for soft close ({context})")
        release_cube_static_anchor(context)

    def apply_cube_static_anchor() -> None:
        if (not cube_static_anchor_active or grip_locked or
                ball_qpos_adr < 0 or phase == -1):
            return
        if phase == 4:
            # The anchor is only a pre-contact anti-jitter helper. Holding the
            # cube fixed while the fingers close can build penetration impulse
            # and launch the cube when the anchor is released.
            release_cube_for_soft_close("phase 4 physical close")
            return
        if phase not in (0, 1, 2, 3, 8):
            return
        data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube_static_pos
        data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = cube_static_quat
        zero_cube_velocity()
        mujoco.mj_forward(model, data)

    def cube_velocity_norms() -> tuple[float, float]:
        if ball_dof_adr < 0:
            return 0.0, 0.0
        lin = float(np.linalg.norm(data.qvel[ball_dof_adr:ball_dof_adr + 3]))
        ang = float(np.linalg.norm(data.qvel[ball_dof_adr + 3:ball_dof_adr + 6]))
        return lin, ang

    def contact_inside_skin_site(point: np.ndarray, site_id: int) -> bool:
        site_pos = data.site_xpos[site_id].copy()
        site_xmat = data.site_xmat[site_id].reshape(3, 3)
        local = site_xmat.T @ (np.asarray(point, dtype=np.float64) - site_pos)
        size = model.site_size[site_id].copy()
        return bool(np.all(np.abs(local) <= size + 0.004))

    def tactile_contact_forces() -> tuple[float, float]:
        left_force = 0.0
        right_force = 0.0
        force6 = np.zeros(6, dtype=np.float64)
        finger_bodies = {int(left_id), int(right_id)}
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body not in finger_bodies:
                continue
            try:
                mujoco.mj_contactForce(model, data, ci, force6)
                normal_force = max(0.0, float(force6[0]))
            except Exception:
                normal_force = 0.0
            contact_point = np.array(con.pos, dtype=np.float64)
            if (other_body == int(left_id) and
                    contact_inside_skin_site(contact_point, left_skin_site_id)):
                left_force += normal_force
            elif (other_body == int(right_id) and
                  contact_inside_skin_site(contact_point, right_skin_site_id)):
                right_force += normal_force
        return left_force, right_force

    def quantize_skin_force(force: float) -> float:
        force = float(np.clip(
            force,
            SKIN_FORCE_RANGE_MIN_N,
            SKIN_FORCE_RANGE_MAX_N,
        ))
        if force < SKIN_FORCE_MIN_RECOGNITION_N:
            return 0.0
        if SKIN_FORCE_RESOLUTION_N > 0.0:
            force = round(force / SKIN_FORCE_RESOLUTION_N) * SKIN_FORCE_RESOLUTION_N
        return float(np.clip(
            force,
            SKIN_FORCE_RANGE_MIN_N,
            SKIN_FORCE_RANGE_MAX_N,
        ))

    def read_skin() -> TactileReading:
        raw = skin_sensor.read(data)
        left_contact, right_contact = tactile_contact_forces()
        return TactileReading(
            left_force=quantize_skin_force(max(raw.left_force, left_contact)),
            right_force=quantize_skin_force(max(raw.right_force, right_contact)),
            timestamp=raw.timestamp,
        )

    def skin_squeeze_ok(reading, min_force: float = SKIN_CONFIRM_FORCE) -> bool:
        return (
            reading.left_force >= min_force and
            reading.right_force >= min_force and
            reading.balance <= SKIN_BALANCE_MAX_RATIO
        )

    def cube_touching_fingers() -> bool:
        finger_bodies = {int(left_id), int(right_id)}
        for ci in range(data.ncon):
            con = data.contact[ci]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 == ball_body_id and b2 in finger_bodies) or
                    (b2 == ball_body_id and b1 in finger_bodies)):
                return True
        return False

    def cube_touching_robot() -> bool:
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
            if body_name.startswith("Link"):
                return True
        return False

    def cube_robot_min_contact_dist() -> float:
        min_dist = float("inf")
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other])) or ""
            if body_name.startswith("Link"):
                min_dist = min(min_dist, float(con.dist))
        return min_dist

    def finger_contact_count() -> int:
        count = 0
        finger_bodies = {int(left_id), int(right_id)}
        for ci in range(data.ncon):
            con = data.contact[ci]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 == ball_body_id and b2 in finger_bodies) or
                    (b2 == ball_body_id and b1 in finger_bodies)):
                count += 1
        return count

    def pad_cube_contact_sides() -> tuple[bool, bool, int]:
        left_hit = False
        right_hit = False
        count = 0
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            pair = {g1, g2}
            if cube_geom_id not in pair:
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body == int(left_id):
                left_hit = True
                count += 1
            if other_body == int(right_id):
                right_hit = True
                count += 1
        return left_hit, right_hit, count

    def finger_contact_patch_span() -> float:
        points: list[np.ndarray] = []
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body in (int(left_id), int(right_id)):
                points.append(np.array(con.pos, dtype=np.float64).copy())
        return contact_patch_span(points, dynamic_grasp_xmat)

    def finger_contact_alignment() -> tuple[float, float]:
        left_points: list[np.ndarray] = []
        right_points: list[np.ndarray] = []
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body == int(left_id):
                left_points.append(np.array(con.pos, dtype=np.float64).copy())
            elif other_body == int(right_id):
                right_points.append(np.array(con.pos, dtype=np.float64).copy())
        return contact_alignment_metrics(
            left_points, right_points, current_ball_xyz(), dynamic_grasp_xmat)

    def cube_nonfinger_robot_contact() -> tuple[bool, str]:
        finger_bodies = {int(left_id), int(right_id)}
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body in finger_bodies:
                continue
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom#{other}"
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, other_body) or f"body#{other_body}"
            if (geom_name == "pad_geom" or geom_name.startswith("box_") or
                    body_name in {"world", "target_cube", "target_box", "friction_pad"}):
                continue
            if body_name.startswith("Link"):
                return True, f"{body_name}/{geom_name}"
        return False, ""

    def gripper_contact_center() -> np.ndarray:
        return data.xpos[pinch_body_id].copy()

    def gripper_open_error() -> float:
        err = 0.0
        for item in gripper_limits:
            q = float(data.qpos[model.jnt_qposadr[item["joint"]]])
            err = max(err, abs(q - float(item["open"])))
        return err

    def gripper_is_open(tol: float = GRIPPER_OPEN_TOL) -> bool:
        return gripper_open_error() <= tol

    def scratch_gripper_contact_center(scratch) -> np.ndarray:
        return scratch.xpos[pinch_body_id].copy()

    def grasp_target_bias(xmat: np.ndarray | None) -> np.ndarray:
        if xmat is None:
            return np.zeros(3, dtype=np.float64)
        face_axis = _unit(xmat[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
        approach_axis = _unit(xmat[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
        return (
            GRASP_TARGET_FACE_BIAS * face_axis +
            GRASP_TARGET_APPROACH_BIAS * approach_axis
        )

    def gripper_object_center() -> np.ndarray:
        return gripper_contact_center() - grasp_target_bias(dynamic_grasp_xmat)

    def scratch_gripper_object_center(scratch, xmat: np.ndarray) -> np.ndarray:
        return scratch_gripper_contact_center(scratch) - grasp_target_bias(xmat)

    def gripper_body_target_for_contact(contact_target: np.ndarray,
                                        xmat: np.ndarray) -> np.ndarray:
        return np.asarray(contact_target, dtype=np.float64) + grasp_target_bias(xmat)

    def gripper_body_target_for_carried_cube(
        cube_target: np.ndarray,
        xmat: np.ndarray,
        cube_offset_from_gripper: np.ndarray,
    ) -> np.ndarray:
        object_center_target = (
            np.asarray(cube_target, dtype=np.float64) -
            np.asarray(cube_offset_from_gripper, dtype=np.float64)
        )
        return object_center_target + grasp_target_bias(xmat)

    def refresh_carried_cube_offset(context: str) -> np.ndarray:
        nonlocal carried_cube_offset
        measured = current_ball_xyz() - gripper_object_center()
        if carried_cube_offset is None:
            carried_cube_offset = measured.copy()
        else:
            carried_cube_offset = (
                0.65 * carried_cube_offset + 0.35 * measured
            )
        print(f"    Carried cube offset ({context}): "
              f"measured={np.round(measured, 4)} "
              f"used={np.round(carried_cube_offset, 4)}")
        return carried_cube_offset.copy()

    def normalized_quat(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64).copy()
        norm = float(np.linalg.norm(quat))
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm

    def quat_conjugate(quat: np.ndarray) -> np.ndarray:
        quat = normalized_quat(quat)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = normalized_quat(a)
        bw, bx, by, bz = normalized_quat(b)
        return normalized_quat(np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dtype=np.float64))

    def grip_geometry_metrics() -> dict[str, float | bool]:
        cube = current_ball_xyz()
        if dynamic_grasp_xmat is not None:
            opening_axis = _unit(
                dynamic_grasp_xmat[:, 1],
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
            face_axis = _unit(
                dynamic_grasp_xmat[:, 0],
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            approach_axis = _unit(
                dynamic_grasp_xmat[:, 2],
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
            )
            center_gap = float(np.linalg.norm(data.xpos[right_id] - data.xpos[left_id]))
        else:
            left = data.xpos[left_id].copy()
            right = data.xpos[right_id].copy()
            span = right - left
            center_gap = float(np.linalg.norm(span))
            opening_axis = _unit(span, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            face_axis = _unit(
                np.cross(opening_axis, approach_axis),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
        delta = cube - gripper_object_center()
        open_axis_err = abs(float(np.dot(delta, opening_axis)))
        face_axis_err = abs(float(np.dot(delta, face_axis)))
        approach_axis_err = abs(float(np.dot(delta, approach_axis)))
        center_err = float(np.linalg.norm(delta))
        ok = (
            open_axis_err <= GRIP_LOCK_MAX_OPEN_AXIS_ERR and
            face_axis_err <= GRIP_LOCK_MAX_FACE_AXIS_ERR and
            approach_axis_err <= GRIP_LOCK_MAX_APPROACH_AXIS_ERR and
            center_err <= GRIP_LOCK_MAX_ERR
        )
        return {
            "ok": bool(ok),
            "center_err": center_err,
            "open_axis_err": open_axis_err,
            "face_axis_err": face_axis_err,
            "approach_axis_err": approach_axis_err,
            "surface_gap": float(center_gap),
        }

    def activate_grip_lock(context: str) -> bool:
        nonlocal grip_locked, grip_lock_offset, grip_lock_quat
        if ball_qpos_adr < 0:
            return False
        cube = current_ball_xyz()
        gc = gripper_object_center()
        err = float(np.linalg.norm(cube - gc))
        contacts = finger_contact_count()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        patch_span = finger_contact_patch_span()
        pair_skew, contact_center_err = finger_contact_alignment()
        min_dist = cube_robot_min_contact_dist()
        skin = read_skin()
        geom = grip_geometry_metrics()
        if (not left_pad_hit or not right_pad_hit or
                not contact_patch_is_large(pad_contacts, patch_span) or
                pair_skew > MAX_CONTACT_PAIR_SKEW or
                contact_center_err > MAX_CONTACT_CENTER_ERR or
                min_dist > CONTACT_CONFIRM_MAX_DIST or
                not skin_squeeze_ok(skin) or err > GRIP_LOCK_MAX_ERR or
                not bool(geom["ok"])):
            print(f"    Physical grasp refused ({context}): contacts={contacts} "
                  f"finger_contacts={pad_contacts} left={left_pad_hit} right={right_pad_hit} "
                  f"patch_span={patch_span*1000:.1f}mm "
                  f"pair_skew={pair_skew*1000:.1f}mm "
                  f"contact_center_err={contact_center_err*1000:.1f}mm "
                  f"contact_dist={min_dist*1000:.2f}mm "
                  f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                  f"bal={skin.balance:.2f} err={err*100:.1f}cm "
                  f"geom_center={geom['center_err']*100:.1f}cm "
                  f"open_axis={geom['open_axis_err']*100:.1f}cm "
                  f"face_axis={geom['face_axis_err']*100:.1f}cm "
                  f"approach_axis={geom['approach_axis_err']*100:.1f}cm "
                  f"surface_gap={geom['surface_gap']*100:.1f}cm")
            return False
        release_cube_static_anchor("physical grasp confirmed")
        grip_locked = True
        grip_lock_offset = cube - gc
        grip_lock_quat = data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7].copy()
        print(f"    Physical grasp confirmed ({context}): contacts={contacts} "
              f"finger_contacts={pad_contacts} patch_span={patch_span*1000:.1f}mm "
              f"pair_skew={pair_skew*1000:.1f}mm "
              f"contact_center_err={contact_center_err*1000:.1f}mm "
              f"contact_dist={min_dist*1000:.2f}mm "
              f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
              f"bal={skin.balance:.2f} "
              f"geom_center={geom['center_err']*100:.1f}cm")
        return True

    def release_grip_lock(context: str) -> None:
        nonlocal grip_locked
        if grip_locked:
            grip_locked = False
            zero_cube_velocity()
            mujoco.mj_forward(model, data)
            print(f"    Physical grasp state cleared ({context})")
        reset_visual_grip_feedback(context)

    def apply_grip_lock() -> None:
        if not grip_locked or ball_qpos_adr < 0 or phase not in (5, 6, 7):
            return
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        skin = read_skin()
        cube = current_ball_xyz()
        target = gripper_object_center() + grip_lock_offset
        err_vec = target - cube
        err = float(np.linalg.norm(err_vec))
        if err <= 1e-6 or err > GRIP_STICTION_MAX_ERR:
            return
        bilateral_force = (
            left_pad_hit and right_pad_hit and
            pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
            skin_squeeze_ok(skin, TACTILE_SECURE_OVERRIDE_MIN_FORCE)
        )
        centered_contact = (
            (pad_contacts >= MIN_STABLE_FINGER_CONTACTS or err <= 0.004) and
            (left_pad_hit or right_pad_hit) and
            err <= 0.012 and
            max(skin.left_force, skin.right_force) >= SKIN_TOUCH_FORCE
        )
        if not (bilateral_force or centered_contact):
            return
        step = err_vec * min(GRIP_STICTION_GAIN, GRIP_STICTION_STEP_M / err)
        data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube + step
        if ball_dof_adr >= 0:
            data.qvel[ball_dof_adr:ball_dof_adr + 3] *= 0.35
            data.qvel[ball_dof_adr + 3:ball_dof_adr + 6] *= 0.55
        mujoco.mj_forward(model, data)

    def stabilize_cube_at_rest() -> None:
        if (ball_qpos_adr < 0 or
                (cube_touching_fingers() and
                 cube_robot_min_contact_dist() <= CONTACT_CONFIRM_MAX_DIST)):
            return
        # Before the fingers touch the cube, keep solver jitter from creeping
        # into the freejoint. Real user perturbations still move it because
        # their displacement is written into qpos by the viewer.
        if phase not in (-1, 0, 1, 2, 8, 9):
            return
        lin_v, ang_v = cube_velocity_norms()
        z = float(data.qpos[ball_qpos_adr + 2])
        near_table = abs(z - CUBE_REST_Z) <= CUBE_PRECONTACT_REST_Z_TOL
        if near_table and not cube_touching_robot():
            data.qpos[ball_qpos_adr + 2] = CUBE_REST_Z
            zero_cube_velocity()
            mujoco.mj_forward(model, data)
            return
        if not near_table:
            return
        if lin_v <= CUBE_IDLE_FREEZE_SPEED and ang_v <= CUBE_IDLE_FREEZE_ANG_SPEED:
            data.qpos[ball_qpos_adr + 2] = CUBE_REST_Z
            zero_cube_velocity()
            mujoco.mj_forward(model, data)

    def capture_cube_from_rgbd(label: str, *, draw: bool = True,
                               log_miss: bool = True,
                               log_success: bool = True) -> np.ndarray | None:
        nonlocal detected_opening_hints
        if rgbd_window is None:
            print(f"    RGB-D camera unavailable during {label}; refusing fixed target.")
            return None
        try:
            rgb_img, depth_img = rgbd_window.render_rgbd(data)
            center_uv, radius = detector.detect(rgb_img, data)
            if center_uv is None:
                if draw and CV2_AVAILABLE:
                    annotated = draw_detection_overlay(rgb_img, False)
                    depth_bgr = rgbd_window._depth_to_bgr(depth_img)
                    cv2.imshow("Eye-in-Hand RGB-D Camera",
                               np.hstack((cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                          depth_bgr)))
                    cv2.waitKey(1)
                if log_miss:
                    print(f"    Vision {label}: cube not in RGB-D color view")
                return None
            est_xyz = detector.estimate_3d_from_depth(
                center_uv, depth_img, data, rgbd_window.width, rgbd_window.height,
            )
            if est_xyz is None:
                if log_miss:
                    print(f"    Vision {label}: no valid RGB-D depth at uv={center_uv}")
                return None
            # Before grasping, the task cube rests on the known tabletop.
            # Side views can see mostly one vertical face and bias the fitted
            # centre above or below the real centre.  Use the table/cube-size
            # prior only in scan / pre-grasp / re-grasp phases; once the cube
            # is lifted, keep the measured height for visual slip feedback.
            est_xyz = est_xyz.copy()
            if phase in (0, 1, 2, 3, 8) and est_xyz[2] <= CUBE_REST_Z + CUBE_HALF_SIZE:
                est_xyz[2] = CUBE_REST_Z
            else:
                est_xyz[2] = max(float(est_xyz[2]), CUBE_REST_Z)
            in_workspace = (
                workspace_contains_xy(est_xyz[:2]) and
                VISION_Z_MIN <= float(est_xyz[2]) <= VISION_Z_MAX
            )
            if not in_workspace:
                if log_miss:
                    print(f"    Vision {label}: rejected estimate {np.round(est_xyz, 3)}")
                return None
            if cube_static_anchor_active and phase in (0, 1, 2, 3, 8):
                anchor_err = float(np.linalg.norm(est_xyz - cube_static_pos))
                if anchor_err > RGBD_ANCHORED_CUBE_MAX_ERR_M:
                    if log_miss:
                        print(f"    Vision {label}: rejected estimate inconsistent "
                              f"with stationary cube "
                              f"rgbd={np.round(est_xyz, 3)} "
                              f"anchor={np.round(cube_static_pos, 3)} "
                              f"err={anchor_err*100:.1f}cm "
                              f"limit={RGBD_ANCHORED_CUBE_MAX_ERR_M*100:.1f}cm")
                    return None
            plane_ok = (
                detector.last_plane_inliers >= RGBD_MIN_PLANE_INLIERS and
                detector.last_plane_residual_m <= RGBD_MAX_PLANE_RESIDUAL_M
            )
            if (not plane_ok and
                    detector.last_estimate_spread_m > RGBD_MAX_ESTIMATE_SPREAD_M):
                if log_miss:
                    print(f"    Vision {label}: rejected unstable RGB-D estimate "
                          f"{np.round(est_xyz, 3)} "
                          f"spread={detector.last_estimate_spread_m*100:.1f}cm "
                          f"plane_res={detector.last_plane_residual_m*1000:.1f}mm "
                          f"inliers={detector.last_plane_inliers}")
                return None
            detected_opening_hints = detector.last_opening_hints_world(data)
            if draw and CV2_AVAILABLE:
                annotated = draw_detection_overlay(rgb_img, True, est_xyz)
                annotated = draw_projected_cuboid_overlay(
                    annotated,
                    data,
                    rgb_cam_id,
                    est_xyz,
                    np.array([CUBE_HALF_SIZE, CUBE_HALF_SIZE, CUBE_HALF_SIZE],
                             dtype=np.float64),
                    "red cube",
                )
                depth_bgr = rgbd_window._depth_to_bgr(depth_img)
                cv2.imshow("Eye-in-Hand RGB-D Camera",
                           np.hstack((cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                      depth_bgr)))
                cv2.waitKey(1)
            hint_msg = ""
            if detected_opening_hints:
                hint_msg = f" opening_hint={np.round(detected_opening_hints[0][:2], 3)}"
            truth = current_ball_xyz()
            truth_err = float(np.linalg.norm(est_xyz - truth))
            spread_msg = (
                f" spread={detector.last_estimate_spread_m*100:.1f}cm/"
                f"{detector.last_detection_spread_px:.1f}px"
                f" plane={detector.last_plane_residual_m*1000:.1f}mm/"
                f"{detector.last_plane_inliers}"
                f" true_err={truth_err*100:.1f}cm"
            )
            if log_success:
                print(f"    Vision {label}: uv={center_uv} r={radius}px "
                      f"xyz={np.round(est_xyz, 3)}{spread_msg}{hint_msg}")
            return est_xyz
        except Exception as exc:
            print(f"    Vision {label}: ERROR {exc}")
            import traceback; traceback.print_exc()
            return None

    def capture_box_from_rgbd(label: str, *,
                              log_miss: bool = True) -> np.ndarray | None:
        nonlocal box_search_hint_xy
        if rgbd_window is None:
            print(f"    RGB-D camera unavailable during {label}; cannot locate box.")
            return None
        try:
            rgb_img, depth_img = rgbd_window.render_rgbd(data)
            est_xyz = box_detector.estimate_3d_from_rgbd(
                rgb_img, depth_img, data, rgbd_window.width, rgbd_window.height,
            )
            if est_xyz is None:
                if log_miss:
                    print(f"    Vision {label}: target box not in RGB-D view")
                return None
            box_search_hint_xy = est_xyz[:2].copy()
            extent = box_detector.last_extent_m
            if (box_detector.last_area_px < BOX_ACCEPT_MIN_AREA_PX or
                    extent[0] < BOX_ACCEPT_MIN_XY_EXTENT_M or
                    extent[1] < BOX_ACCEPT_MIN_XY_EXTENT_M or
                    extent[2] < BOX_ACCEPT_MIN_Z_EXTENT_M):
                if log_miss:
                    print(f"    Vision {label}: rejected incomplete box view "
                          f"xyz={np.round(est_xyz, 3)} "
                          f"extent={np.round(extent, 3)} "
                          f"area={box_detector.last_area_px:.0f}px")
                return None
            if not workspace_contains_xy(est_xyz[:2], margin=0.22):
                if log_miss:
                    print(f"    Vision {label}: rejected box estimate "
                          f"{np.round(est_xyz, 3)}")
                return None
            if CV2_AVAILABLE:
                annotated = draw_projected_cuboid_overlay(
                    rgb_img.copy(),
                    data,
                    rgb_cam_id,
                    np.array([est_xyz[0], est_xyz[1], BOX_CENTER_Z_PRIOR],
                             dtype=np.float64),
                    BOX_SIZE,
                    "blue box",
                )
                depth_bgr = rgbd_window._depth_to_bgr(depth_img)
                cv2.imshow("Eye-in-Hand RGB-D Camera",
                           np.hstack((cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                                      depth_bgr)))
                cv2.waitKey(1)
            print(f"    Vision {label}: box_xyz={np.round(est_xyz, 3)} "
                  f"raw={np.round(box_detector.last_raw_center_xyz, 3) if box_detector.last_raw_center_xyz is not None else None} "
                  f"corr={box_detector.last_center_correction_m*1000:.1f}mm "
                  f"inside={box_detector.last_footprint_inside_ratio:.2f} "
                  f"extent={np.round(box_detector.last_extent_m, 3)} "
                  f"area={box_detector.last_area_px:.0f}px "
                  f"pts={box_detector.last_points}")
            return est_xyz
        except Exception as exc:
            print(f"    Vision {label}: BOX ERROR {exc}")
            import traceback; traceback.print_exc()
            return None

    def rgbd_frustum_status(xyz: np.ndarray) -> tuple[bool, str]:
        if rgb_cam_id < 0:
            return False, "camera unavailable"
        cam_pos = data.cam_xpos[rgb_cam_id].copy()
        cam_mat = data.cam_xmat[rgb_cam_id].reshape(3, 3)
        local = cam_mat.T @ (np.asarray(xyz, dtype=np.float64) - cam_pos)
        depth = -float(local[2])
        if depth < MIN_RGBD_DEPTH_M:
            return False, f"too close ({depth*100:.1f}cm)"
        if depth > MAX_RGBD_DEPTH_M:
            return False, f"too far ({depth*100:.1f}cm)"
        h_ang = abs(math.degrees(math.atan2(float(local[0]), max(depth, 1e-6))))
        v_ang = abs(math.degrees(math.atan2(float(local[1]), max(depth, 1e-6))))
        if h_ang > D405C_DEPTH_FOVX_DEG * 0.5:
            return False, f"outside horizontal FOV ({h_ang:.1f}deg)"
        if v_ang > D405C_DEPTH_FOVY_DEG * 0.5:
            return False, f"outside vertical FOV ({v_ang:.1f}deg)"
        return True, (
            f"in frustum depth={depth*100:.1f}cm "
            f"angles=({h_ang:.1f},{v_ang:.1f})deg"
        )

    def reset_visual_grip_feedback(context: str) -> None:
        nonlocal visual_grip_reference_offset, visual_grip_filtered_xyz
        nonlocal visual_grip_last_time, visual_grip_slip_m
        nonlocal visual_grip_slip_rate_mps, visual_grip_drop_m
        nonlocal visual_grip_weight, visual_grip_miss_count
        nonlocal last_visual_grip_frame, last_visual_grip_log_frame
        nonlocal last_visual_secure_override_frame
        nonlocal transport_slip_abort_frames
        visual_grip_reference_offset = None
        visual_grip_filtered_xyz = None
        visual_grip_last_time = float(data.time)
        visual_grip_slip_m = 0.0
        visual_grip_slip_rate_mps = 0.0
        visual_grip_drop_m = 0.0
        visual_grip_weight = 0.0
        visual_grip_miss_count = 0
        last_visual_grip_frame = -VISUAL_GRIP_CAPTURE_EVERY_N
        last_visual_grip_log_frame = -VISUAL_GRIP_LOG_EVERY_N
        last_visual_secure_override_frame = -TACTILE_SECURE_LOG_EVERY_N
        transport_slip_abort_frames = 0

    def _ramp01(value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 1.0 if value >= hi else 0.0
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def update_visual_grip_feedback(label: str, *, force: bool = False) -> None:
        nonlocal visual_grip_reference_offset, visual_grip_filtered_xyz
        nonlocal visual_grip_last_time, visual_grip_slip_m
        nonlocal visual_grip_slip_rate_mps, visual_grip_drop_m
        nonlocal visual_grip_weight, visual_grip_miss_count
        nonlocal last_visual_grip_frame, last_visual_grip_log_frame
        if rgbd_window is None:
            return
        if not force and frame - last_visual_grip_frame < VISUAL_GRIP_CAPTURE_EVERY_N:
            return
        last_visual_grip_frame = frame
        est_xyz = capture_cube_from_rgbd(
            f"grip {label}",
            draw=False,
            log_miss=False,
            log_success=False,
        )
        if est_xyz is None:
            visual_grip_miss_count += 1
            visual_grip_weight *= 0.88
            return

        visual_grip_miss_count = 0
        if visual_grip_filtered_xyz is None:
            visual_grip_filtered_xyz = est_xyz.copy()
        else:
            visual_grip_filtered_xyz = (
                visual_grip_filtered_xyz +
                VISUAL_GRIP_FILTER_ALPHA * (est_xyz - visual_grip_filtered_xyz)
            )

        now = float(data.time)
        rel = visual_grip_filtered_xyz - gripper_object_center()
        rel_xy = float(np.linalg.norm(rel[:2]))
        rel_z = abs(float(rel[2]))
        if phase in (5, 6, 7) and (
                rel_xy > VISUAL_GRIP_MAX_REL_XY_M or
                rel_z > VISUAL_GRIP_MAX_REL_Z_M):
            visual_grip_miss_count += 1
            visual_grip_weight *= 0.72
            if frame - last_visual_grip_log_frame >= VISUAL_GRIP_LOG_EVERY_N:
                last_visual_grip_log_frame = frame
                print(f"    Visual grip estimate ignored ({label}): "
                      f"implausible rel={np.round(rel, 3)} "
                      f"limit_xy={VISUAL_GRIP_MAX_REL_XY_M*1000:.0f}mm "
                      f"limit_z={VISUAL_GRIP_MAX_REL_Z_M*1000:.0f}mm")
            return
        if visual_grip_reference_offset is None:
            visual_grip_reference_offset = rel.copy()
            visual_grip_last_time = now
            visual_grip_slip_m = 0.0
            visual_grip_slip_rate_mps = 0.0
            visual_grip_drop_m = 0.0
            visual_grip_weight = 0.0
            print(f"    Visual grip reference acquired ({label}): "
                  f"cube={np.round(visual_grip_filtered_xyz, 3)} "
                  f"rel={np.round(rel, 3)}")
            return

        slip_vec = rel - visual_grip_reference_offset
        slip_mag = float(np.linalg.norm(slip_vec))
        drop = max(0.0, float(visual_grip_reference_offset[2] - rel[2]))
        slip_measure = max(slip_mag, drop * 1.35)
        dt = max(now - visual_grip_last_time, 1e-3)
        slip_rate = max(0.0, (slip_measure - visual_grip_slip_m) / dt)
        pos_risk = _ramp01(slip_mag, VISUAL_GRIP_ERR_SOFT_M, VISUAL_GRIP_ERR_HARD_M)
        drop_risk = _ramp01(drop, VISUAL_GRIP_DROP_SOFT_M, VISUAL_GRIP_DROP_HARD_M)
        rate_risk = _ramp01(
            slip_rate, VISUAL_GRIP_RATE_SOFT_MPS, VISUAL_GRIP_RATE_HARD_MPS)
        risk = max(pos_risk, drop_risk, rate_risk)
        visual_grip_weight = float(np.clip(
            0.72 * visual_grip_weight + 0.28 * risk,
            0.0,
            1.0,
        ))
        visual_grip_last_time = now
        visual_grip_slip_m = slip_measure
        visual_grip_slip_rate_mps = slip_rate
        visual_grip_drop_m = drop

        if (visual_grip_weight > 0.20 and
                frame - last_visual_grip_log_frame >= VISUAL_GRIP_LOG_EVERY_N):
            last_visual_grip_log_frame = frame
            print(f"    Visual grip assist ({label}): "
                  f"w={visual_grip_weight:.2f} "
                  f"slip={slip_mag*1000:.1f}mm "
                  f"drop={drop*1000:.1f}mm "
                  f"rate={slip_rate*1000:.1f}mm/s")

    def maintain_fused_grip_force(
        label: str,
        reading: TactileReading | None = None,
        *,
        force_visual: bool = False,
    ) -> TactileReading:
        skin = read_skin() if reading is None else reading
        update_visual_grip_feedback(label, force=force_visual)
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        tactile_geometry_secure = (
            lifted > CARRY_MIN_LIFT and
            err <= TACTILE_SECURE_OVERRIDE_MAX_ERR and
            left_pad_hit and right_pad_hit and
            pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
            skin_squeeze_ok(skin, SKIN_HOLD_TARGET_FORCE) and
            visual_grip_drop_m <= VISUAL_GRIP_DROP_SOFT_M * 1.8
        )
        visual_force_weight = visual_grip_weight
        allow_visual_position_comp = True
        if tactile_geometry_secure and visual_grip_slip_m > VISUAL_GRIP_ERR_HARD_M:
            # A large lateral RGB-D jump with no downward motion and strong
            # bilateral tactile hold is usually an occlusion/segmentation
            # outlier.  Keep logging it, but do not over-tighten the gripper.
            visual_force_weight = min(visual_force_weight, 0.25)
            allow_visual_position_comp = False
        min_force = min(
            SKIN_FORCE_RANGE_MAX_N,
            SKIN_HOLD_MIN_FORCE + visual_force_weight * VISUAL_GRIP_MIN_FORCE_BOOST,
        )
        target_force = min(
            SKIN_FORCE_RANGE_MAX_N,
            SKIN_HOLD_TARGET_FORCE +
            visual_force_weight * VISUAL_GRIP_TARGET_FORCE_BOOST,
        )
        max_force = min(
            SKIN_FORCE_RANGE_MAX_N,
            SKIN_HOLD_MAX_FORCE + visual_force_weight * VISUAL_GRIP_MAX_FORCE_BOOST,
        )
        target_force = min(target_force, max_force)
        min_force = min(min_force, target_force)
        gripper_ctrl.maintain_tactile_force(
            skin,
            min_force=min_force,
            target_force=target_force,
            max_force=max_force,
        )
        if allow_visual_position_comp:
            gripper_ctrl.compensate_visual_slip(
                visual_grip_weight,
                visual_grip_slip_m,
                visual_grip_drop_m,
            )
        return skin

    def ensure_transport_grip_hold(label: str) -> None:
        if gripper_ctrl.holding:
            return
        print(f"    Transport grip hold re-armed ({label}).")
        gripper_ctrl.hold(duration_frames=GRIP_TRANSPORT_HOLD_FRAMES)

    def visual_grip_summary() -> str:
        if visual_grip_reference_offset is None:
            return "visual_grip=uninitialized"
        return (
            f"visual_grip w={visual_grip_weight:.2f} "
            f"slip={visual_grip_slip_m*1000:.1f}mm "
            f"drop={visual_grip_drop_m*1000:.1f}mm "
            f"rate={visual_grip_slip_rate_mps*1000:.1f}mm/s "
            f"miss={visual_grip_miss_count}"
        )

    def plan_is_ready(plan: dict[str, IKResult]) -> bool:
        required = {
            "approach", "descend_mid", "grasp",
            "lift", "carry_mid", "place_above", "place_drop",
        }
        return (
            set(plan) == required and
            all(plan[name].success for name in required)
        )

    def normalize_box_center_estimate(box: np.ndarray) -> np.ndarray:
        box = np.asarray(box, dtype=np.float64).copy()
        box[2] = BOX_CENTER_Z_PRIOR
        return box

    def box_target_center() -> np.ndarray | None:
        if detected_box_pos is None:
            return None
        return normalize_box_center_estimate(detected_box_pos)

    def box_bottom_top_z(box: np.ndarray) -> float:
        return float(box[2] - BOX_SIZE[2] + BOX_WALL * 0.5)

    def place_release_cube_center(box: np.ndarray) -> np.ndarray:
        box = np.asarray(box, dtype=np.float64)
        bottom_release = (
            box_bottom_top_z(box) + CUBE_HALF_SIZE + PLACE_RELEASE_CLEARANCE
        )
        return np.array([
            box[0],
            box[1],
            bottom_release,
        ], dtype=np.float64)

    def box_release_inner_half(margin: float = PLACE_RELEASE_XY_MARGIN) -> np.ndarray:
        return np.maximum(
            BOX_SIZE[:2] - BOX_WALL - CUBE_HALF_SIZE - float(margin),
            np.array([0.004, 0.004], dtype=np.float64),
        )

    def box_entry_gap_xy(cube_xyz: np.ndarray, box: np.ndarray) -> float:
        cube_xyz = np.asarray(cube_xyz, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)
        inner_half = box_release_inner_half()
        over = np.maximum(np.abs(cube_xyz[:2] - box[:2]) - inner_half, 0.0)
        return float(np.linalg.norm(over))

    def adaptive_box_release_cube_center(
        box: np.ndarray,
        cube_xyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Choose a safe point inside the box, biased toward the entry side.

        Requiring the carried cube to reach the exact box center made edge-of-
        workspace placements loop near the box wall.  The release condition is
        "inside the usable opening", so the planned target should be any safe
        interior point, preferably the closest one from the current approach.
        """
        target = place_release_cube_center(box)
        if cube_xyz is None:
            return target
        cube_xyz = np.asarray(cube_xyz, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)
        inner_half = box_release_inner_half()
        safe_half = inner_half * BOX_ENTRY_TARGET_FRACTION
        center_band = inner_half * BOX_ENTRY_CENTER_BLEND
        for axis in (0, 1):
            delta = float(cube_xyz[axis] - box[axis])
            if abs(delta) > center_band[axis]:
                target[axis] = box[axis] + math.copysign(safe_half[axis], delta)
            else:
                target[axis] = float(np.clip(
                    cube_xyz[axis],
                    box[axis] - center_band[axis],
                    box[axis] + center_band[axis],
                ))
        return target

    def carry_step_cube_target(
        cube_xyz: np.ndarray,
        box: np.ndarray,
        *,
        direct_to_box: bool = False,
        high_clearance: bool = False,
    ) -> np.ndarray:
        cube_xyz = np.asarray(cube_xyz, dtype=np.float64)
        box = np.asarray(box, dtype=np.float64)
        release = adaptive_box_release_cube_center(
            box,
            cube_xyz if direct_to_box else None,
        )
        target_xy = release[:2].copy()
        delta = target_xy - cube_xyz[:2]
        dist = float(np.linalg.norm(delta))
        if (not direct_to_box) and dist > CARRY_STEP_FINAL_DIRECT_XY_M:
            step_max = CARRY_HIGH_STEP_MAX_XY_M if high_clearance else CARRY_STEP_MAX_XY_M
            target_xy = cube_xyz[:2] + delta / dist * min(step_max, dist)
        target_z = carry_height_for_xy(target_xy, box)
        if high_clearance:
            target_z = max(target_z, float(box[2] + PLACE_ABOVE_HEIGHT + 0.010))
        return np.array([
            target_xy[0],
            target_xy[1],
            target_z,
        ], dtype=np.float64)

    def cube_target_box_contacts() -> tuple[int, bool, bool, float]:
        count = 0
        bottom_hit = False
        wall_hit = False
        min_dist = float("inf")
        if cube_geom_id < 0:
            return count, bottom_hit, wall_hit, min_dist
        for ci in range(data.ncon):
            con = data.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if not geom_name.startswith("box_"):
                continue
            count += 1
            min_dist = min(min_dist, float(con.dist))
            if geom_name == "box_bottom":
                bottom_hit = True
            else:
                wall_hit = True
        return count, bottom_hit, wall_hit, min_dist

    def cube_release_window(*, require_box_support: bool = True) -> tuple[bool, str]:
        box = box_target_center()
        if box is None:
            return False, "target box unknown"
        cube = current_ball_xyz()
        target = adaptive_box_release_cube_center(box, cube)
        inner_half = box_release_inner_half()
        dx = abs(float(cube[0] - box[0]))
        dy = abs(float(cube[1] - box[1]))
        xy_ok = dx <= inner_half[0] and dy <= inner_half[1]
        bottom = box_bottom_top_z(box)
        z_low = bottom + CUBE_HALF_SIZE - 0.006
        z_high = target[2] + PLACE_RELEASE_Z_TOL
        z = float(cube[2])
        z_ok = z_low <= z <= z_high
        contact_count, bottom_hit, wall_hit, contact_dist = cube_target_box_contacts()
        support_ok = (
            (not require_box_support) or
            (not PLACE_REQUIRE_BOX_BOTTOM_CONTACT) or
            bottom_hit
        )
        ok = bool(xy_ok and z_ok and support_ok)
        dist_msg = (
            "inf" if contact_dist == float("inf")
            else f"{contact_dist*1000:.2f}mm"
        )
        reason = (
            f"cube={np.round(cube, 4)} box={np.round(box, 4)} "
            f"target={np.round(target, 4)} "
            f"dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm "
            f"z={z:.3f} range=({z_low:.3f},{z_high:.3f}) "
            f"box_contacts={contact_count} bottom={bottom_hit} "
            f"wall={wall_hit} dist={dist_msg} "
            f"support_required={require_box_support}"
        )
        return ok, reason

    def cube_inside_box() -> bool:
        box = box_target_center()
        if box is None:
            return False
        cube = current_ball_xyz()
        inner_half = BOX_SIZE[:2] - BOX_WALL - CUBE_HALF_SIZE - 0.004
        x_ok = abs(float(cube[0] - box[0])) <= inner_half[0]
        y_ok = abs(float(cube[1] - box[1])) <= inner_half[1]
        bottom = box_bottom_top_z(box)
        target = place_release_cube_center(box)
        contact_count, bottom_hit, _, _ = cube_target_box_contacts()
        z_ok = (
            bottom + CUBE_HALF_SIZE - PLACE_INSIDE_Z_TOL <= float(cube[2]) <=
            target[2] + PLACE_RELEASE_Z_TOL
        )
        return bool(x_ok and y_ok and z_ok and (bottom_hit or contact_count > 0))

    def cube_over_box_entry() -> bool:
        box = box_target_center()
        if box is None:
            return False
        cube = current_ball_xyz()
        inner_half = box_release_inner_half()
        dx = abs(float(cube[0] - box[0]))
        dy = abs(float(cube[1] - box[1]))
        target = place_release_cube_center(box)
        return bool(
            dx <= inner_half[0] and
            dy <= inner_half[1] and
            float(cube[2]) >= target[2] + 0.020
        )

    def cube_ready_for_guided_place_drop() -> bool:
        box = box_target_center()
        if box is None:
            return False
        cube = current_ball_xyz()
        target = adaptive_box_release_cube_center(box, cube)
        inner_half = box_release_inner_half() + PLACE_DIRECT_DROP_XY_MARGIN_M
        dx = abs(float(cube[0] - box[0]))
        dy = abs(float(cube[1] - box[1]))
        return bool(
            dx <= inner_half[0] and
            dy <= inner_half[1] and
            float(cube[2]) >= target[2] + 0.030 and
            (
                cube_transport_contact_ok() or
                cube_transport_relaxed_hold_ok() or
                cube_transport_geometry_hold_ok() or
                cube_transport_stiction_hold_ok()
            )
        )

    def cube_in_box_drop_corridor() -> bool:
        box = box_target_center()
        if box is None:
            return False
        cube = current_ball_xyz()
        inner_half = box_release_inner_half()
        dx = abs(float(cube[0] - box[0]))
        dy = abs(float(cube[1] - box[1]))
        target = place_release_cube_center(box)
        z = float(cube[2])
        return bool(
            dx <= inner_half[0] and
            dy <= inner_half[1] and
            target[2] - 0.010 <= z <= box[2] + BOX_SIZE[2] + CUBE_HALF_SIZE + 0.020
        )

    def diagnose_failure(context: str) -> None:
        cube = current_ball_xyz()
        gc = gripper_object_center()
        lin_v, ang_v = cube_velocity_norms()
        print(f"    DIAG [{context}]")
        print(f"      cube={np.round(cube, 4)}  grasp_center={np.round(gc, 4)}  "
              f"err={np.linalg.norm(cube-gc)*100:.1f}cm")
        skin = read_skin()
        print(f"      cube_v={lin_v:.4f}m/s  cube_w={ang_v:.4f}rad/s  "
              f"contacts={finger_contact_count()}  "
              f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
              f"bal={skin.balance:.2f}")
        print(f"      {visual_grip_summary()}")
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        print(f"      finger_contacts={pad_contacts}  left={left_pad_hit} right={right_pad_hit}")
        if detected_ball_pos is not None:
            print(f"      last_rgb_xyz={np.round(detected_ball_pos, 4)}")
        if detected_box_pos is not None:
            print(f"      box_rgb_xyz={np.round(detected_box_pos, 4)}")
        if dynamic_ik_plan:
            for name, result in dynamic_ik_plan.items():
                print(f"      ik[{name}]: ok={result.success} "
                      f"pos_err={result.error_norm:.4f} "
                      f"ori_err={result.orientation_error_norm:.3f}")

    def solve_oriented_target(
        scratch,
        target: np.ndarray,
        xmat: np.ndarray,
        rest: np.ndarray | None = None,
        pos_tol: float = 0.005,
        ori_tol: float = 0.16,
    ) -> IKResult:
        return solve_gripper_center_ik(
            model, scratch, target, pinch_body_id, pinch_body_id, arm_joints,
            max_iter=700, tol=pos_tol,
            target_xmat=xmat, orientation_body_id=tool_id,
            orientation_weight=GRASP_ORI_WEIGHT, orientation_tol=ori_tol,
            rest_angles=rest, rest_weight=0.015,
        )

    def update_place_plan_for_carried_offset(
        context: str,
        *,
        direct_to_box: bool = False,
        high_clearance: bool = False,
    ) -> bool:
        nonlocal dynamic_ik_plan
        if dynamic_grasp_xmat is None:
            print(f"    Cannot update place plan ({context}): grasp frame unknown.")
            return False
        box = box_target_center()
        if box is None:
            print(f"    Cannot update place plan ({context}): target box unknown.")
            return False
        offset = refresh_carried_cube_offset(context)
        cube_now = current_ball_xyz()
        release_cube_target = adaptive_box_release_cube_center(
            box,
            cube_now if direct_to_box else None,
        )
        above_cube_target = carry_step_cube_target(
            cube_now,
            box,
            direct_to_box=direct_to_box,
            high_clearance=high_clearance,
        )
        if direct_to_box:
            above_cube_target[:2] = release_cube_target[:2]
            above_cube_target[2] = carry_height_for_xy(
                release_cube_target[:2],
                box,
            )
            if high_clearance:
                above_cube_target[2] = max(
                    above_cube_target[2],
                    float(box[2] + PLACE_ABOVE_HEIGHT + 0.010),
                )
        cube_targets = {
            "carry_mid": carry_midpoint_for_grasp(
                cube_now,
                box,
                dynamic_grasp_xmat,
            ),
            "place_above": above_cube_target,
            "place_drop": release_cube_target,
        }
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos[:]
        scratch.qvel[:] = 0.0
        mujoco.mj_forward(model, scratch)
        rest = arm_ctrl.current()
        updated: dict[str, IKResult] = {}
        for name in ("carry_mid", "place_above", "place_drop"):
            target_offset = offset.copy()
            if name in ("carry_mid", "place_above"):
                target_offset[2] = 0.0
            body_target = gripper_body_target_for_carried_cube(
                cube_targets[name],
                dynamic_grasp_xmat,
                target_offset,
            )
            result = solve_oriented_target(
                scratch,
                body_target,
                dynamic_grasp_xmat,
                rest=rest,
                pos_tol=0.009,
                ori_tol=PLACE_ORI_TOL,
            )
            if not result.success:
                result = solve_oriented_target(
                    scratch,
                    body_target,
                    dynamic_grasp_xmat,
                    rest=rest,
                    pos_tol=0.012,
                    ori_tol=0.35,
                )
            if not result.success:
                if name in ("carry_mid", "place_above"):
                    existing = dynamic_ik_plan.get(name)
                    if existing is not None and existing.success:
                        print(f"    Keeping pre-validated {name} plan ({context}); "
                              "oriented compensation failed, and position-only "
                              "transport would rotate the held cube.")
                        result = existing
                    else:
                        print(f"    Place compensation IK failed ({context}) "
                              f"{name}: cube_target={np.round(cube_targets[name], 4)} "
                              f"body_target={np.round(body_target, 4)} "
                              f"pos_err={result.error_norm:.4f} "
                              f"ori_err={result.orientation_error_norm:.3f}.")
                        return False
                else:
                    result = solve_gripper_center_ik(
                        model, scratch, body_target,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=0.010,
                        rest_angles=rest, rest_weight=0.015,
                    )
                    if result.success:
                        print(f"    Place compensation used position IK ({context}) "
                              f"{name}: cube_target={np.round(cube_targets[name], 4)} "
                              f"body_target={np.round(body_target, 4)} "
                              "actual release remains gated by cube_release_window().")
            if direct_to_box and name in ("place_above", "place_drop"):
                pos_result = solve_gripper_center_ik(
                    model, scratch, body_target,
                    pinch_body_id, pinch_body_id, arm_joints,
                    max_iter=900, tol=0.010,
                    rest_angles=rest, rest_weight=0.015,
                )
                if pos_result.success:
                    if result.success:
                        old_span = float(np.max(np.abs(result.angles - rest)))
                        new_span = float(np.max(np.abs(pos_result.angles - rest)))
                    else:
                        old_span = float("inf")
                        new_span = 0.0
                    if (not result.success or name == "place_above" or
                            new_span <= old_span * 1.15):
                        print(f"    Direct box {name} uses position-priority IK "
                              f"({context}): pos_err={pos_result.error_norm:.4f} "
                              f"joint_span={new_span:.3f} "
                              f"oriented_span={old_span:.3f}")
                        result = pos_result
            if not result.success:
                print(f"    Place compensation IK failed ({context}) "
                      f"{name}: cube_target={np.round(cube_targets[name], 4)} "
                      f"body_target={np.round(body_target, 4)} "
                      f"pos_err={result.error_norm:.4f} "
                      f"ori_err={result.orientation_error_norm:.3f}.")
                return False
            updated[name] = result
            set_joint_positions(model, scratch, arm_joints, result.angles)
            mujoco.mj_forward(model, scratch)
            rest = result.angles
        dynamic_ik_plan.update(updated)
        release_target = cube_targets["place_drop"]
        predicted_body = gripper_body_target_for_carried_cube(
            release_target, dynamic_grasp_xmat, offset)
        print(f"    Place plan compensated ({context}): "
              f"box={np.round(box, 4)} "
              f"cube_release={np.round(release_target, 4)} "
              f"entry_gap={box_entry_gap_xy(current_ball_xyz(), box)*100:.1f}cm "
              f"offset={np.round(offset, 4)} "
              f"gripper_target={np.round(predicted_body, 4)} "
              f"direct={direct_to_box} high={high_clearance}")
        return True

    def scratch_finger_contact_count(scratch) -> int:
        count = 0
        finger_bodies = {int(left_id), int(right_id)}
        for ci in range(scratch.ncon):
            con = scratch.contact[ci]
            b1 = int(model.geom_bodyid[con.geom1])
            b2 = int(model.geom_bodyid[con.geom2])
            if ((b1 == ball_body_id and b2 in finger_bodies) or
                    (b2 == ball_body_id and b1 in finger_bodies)):
                count += 1
        return count

    def scratch_finger_contact_sides(scratch) -> tuple[bool, bool, int]:
        left_hit = False
        right_hit = False
        count = 0
        for ci in range(scratch.ncon):
            con = scratch.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body == int(left_id):
                left_hit = True
                count += 1
            elif other_body == int(right_id):
                right_hit = True
                count += 1
        return left_hit, right_hit, count

    def scratch_finger_contact_patch_span(scratch, xmat: np.ndarray) -> float:
        points: list[np.ndarray] = []
        for ci in range(scratch.ncon):
            con = scratch.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body in (int(left_id), int(right_id)):
                points.append(np.array(con.pos, dtype=np.float64).copy())
        return contact_patch_span(points, xmat)

    def scratch_finger_contact_alignment(scratch, xmat: np.ndarray) -> tuple[float, float]:
        left_points: list[np.ndarray] = []
        right_points: list[np.ndarray] = []
        for ci in range(scratch.ncon):
            con = scratch.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body == int(left_id):
                left_points.append(np.array(con.pos, dtype=np.float64).copy())
            elif other_body == int(right_id):
                right_points.append(np.array(con.pos, dtype=np.float64).copy())
        cube = scratch.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        return contact_alignment_metrics(left_points, right_points, cube, xmat)

    def scratch_nonfinger_robot_contact(scratch) -> bool:
        finger_bodies = {int(left_id), int(right_id)}
        for ci in range(scratch.ncon):
            con = scratch.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            other_body = int(model.geom_bodyid[other])
            if other_body in finger_bodies:
                continue
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
            if (geom_name == "pad_geom" or geom_name.startswith("box_") or
                    body_name in {"world", "target_cube", "target_box", "friction_pad"}):
                continue
            if body_name.startswith("Link"):
                return True
        return False

    def scratch_cube_robot_min_contact_dist(scratch) -> float:
        min_dist = float("inf")
        for ci in range(scratch.ncon):
            con = scratch.contact[ci]
            g1 = int(con.geom1)
            g2 = int(con.geom2)
            if cube_geom_id not in (g1, g2):
                continue
            other = g2 if g1 == cube_geom_id else g1
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[other])) or ""
            if body_name.startswith("Link"):
                min_dist = min(min_dist, float(con.dist))
        return min_dist

    def grasp_frame_penalty(xmat: np.ndarray) -> float:
        if not rejected_grasp_frames:
            return 0.0
        y_axis = _unit(xmat[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))
        z_axis = _unit(xmat[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
        penalty = 0.0
        for rejected_y, rejected_z in rejected_grasp_frames:
            same_opening = abs(float(np.dot(y_axis, rejected_y)))
            same_approach = float(np.dot(z_axis, rejected_z))
            if same_opening > 0.94 and same_approach > 0.90:
                penalty = max(penalty, REJECTED_GRASP_FRAME_PENALTY)
            elif same_opening > 0.94:
                penalty = max(penalty, REJECTED_GRASP_FRAME_PENALTY * 0.35)
        return penalty

    def remember_failed_grasp_frame(reason: str) -> None:
        nonlocal dynamic_grasp_xmat, rejected_grasp_frames
        if dynamic_grasp_xmat is None:
            return
        y_axis = _unit(dynamic_grasp_xmat[:, 1],
                       np.array([0.0, 1.0, 0.0], dtype=np.float64))
        z_axis = _unit(dynamic_grasp_xmat[:, 2],
                       np.array([0.0, 0.0, -1.0], dtype=np.float64))
        for old_y, old_z in rejected_grasp_frames:
            if (abs(float(np.dot(y_axis, old_y))) > 0.97 and
                    float(np.dot(z_axis, old_z)) > 0.94):
                return
        rejected_grasp_frames.append((y_axis.copy(), z_axis.copy()))
        if len(rejected_grasp_frames) > 8:
            rejected_grasp_frames = rejected_grasp_frames[-8:]
        print(f"    Marking failed grasp posture ({reason}); "
              f"will try a different opening/approach axis next.")

    def physical_plan_penalty(plan: dict[str, IKResult],
                              ball_xyz: np.ndarray,
                              box_xyz: np.ndarray,
                              xmat: np.ndarray) -> float:
        """Reject IK poses that reach mathematically but drop the cube physically."""
        if ball_qpos_adr < 0 or ball_dof_adr < 0:
            return 0.0

        sim = mujoco.MjData(model)
        sim.qpos[:] = data.qpos[:]
        sim.qvel[:] = 0.0
        sim.qpos[ball_qpos_adr:ball_qpos_adr + 3] = ball_xyz
        if np.linalg.norm(sim.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7]) < 1e-6:
            sim.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        sim.qvel[ball_dof_adr:ball_dof_adr + 6] = 0.0

        def write_arm(angles: np.ndarray) -> None:
            for idx, jid in enumerate(arm_joints):
                act = joint_to_actuator.get(jid)
                if act is not None:
                    sim.ctrl[act] = float(angles[idx])

        def write_gripper(t: float) -> None:
            t = float(np.clip(t, 0.0, 1.0))
            for item in gripper_limits:
                act = joint_to_actuator.get(item["joint"])
                if act is not None:
                    sim.ctrl[act] = (
                        item["open"] + (item["closed"] - item["open"]) * t
                    )

        def step_substeps() -> None:
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, sim)

        def scratch_cube_inside_box() -> bool:
            cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
            box_eval = normalize_box_center_estimate(box_xyz)
            inner_half = BOX_SIZE[:2] - BOX_WALL - CUBE_HALF_SIZE - 0.004
            x_ok = abs(float(cube[0] - box_eval[0])) <= inner_half[0]
            y_ok = abs(float(cube[1] - box_eval[1])) <= inner_half[1]
            bottom = box_bottom_top_z(box_eval)
            target = place_release_cube_center(box_eval)
            z_ok = (
                bottom + CUBE_HALF_SIZE - PLACE_INSIDE_Z_TOL <= float(cube[2]) <=
                target[2] + PLACE_RELEASE_Z_TOL
            )
            return bool(x_ok and y_ok and z_ok)

        def scratch_cube_release_window() -> tuple[bool, float, float, float, float]:
            cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
            box_eval = normalize_box_center_estimate(box_xyz)
            target = adaptive_box_release_cube_center(box_eval, cube)
            inner_half = box_release_inner_half()
            dx = abs(float(cube[0] - box_eval[0]))
            dy = abs(float(cube[1] - box_eval[1]))
            bottom = box_bottom_top_z(box_eval)
            z_low = bottom + CUBE_HALF_SIZE - 0.006
            z_high = target[2] + PLACE_RELEASE_Z_TOL
            z = float(cube[2])
            target_err = float(np.linalg.norm(cube - target))
            ok = bool(
                dx <= inner_half[0] and
                dy <= inner_half[1] and
                z_low <= z <= z_high
            )
            return ok, dx, dy, z, target_err

        def scratch_contact_inside_site(point: np.ndarray, site_id: int) -> bool:
            site_pos = sim.site_xpos[site_id].copy()
            site_xmat = sim.site_xmat[site_id].reshape(3, 3)
            local = site_xmat.T @ (np.asarray(point, dtype=np.float64) - site_pos)
            size = model.site_size[site_id].copy()
            return bool(np.all(np.abs(local) <= size + 0.004))

        def scratch_skin_reading() -> TactileReading:
            left_force = 0.0
            right_force = 0.0
            force6 = np.zeros(6, dtype=np.float64)
            for ci in range(sim.ncon):
                con = sim.contact[ci]
                g1 = int(con.geom1)
                g2 = int(con.geom2)
                if cube_geom_id not in (g1, g2):
                    continue
                other = g2 if g1 == cube_geom_id else g1
                other_body = int(model.geom_bodyid[other])
                if other_body not in (int(left_id), int(right_id)):
                    continue
                try:
                    mujoco.mj_contactForce(model, sim, ci, force6)
                    normal_force = max(0.0, float(force6[0]))
                except Exception:
                    normal_force = 0.0
                if (other_body == int(left_id) and
                        scratch_contact_inside_site(con.pos, left_skin_site_id)):
                    left_force += normal_force
                elif (other_body == int(right_id) and
                      scratch_contact_inside_site(con.pos, right_skin_site_id)):
                    right_force += normal_force
            return TactileReading(
                left_force=quantize_skin_force(left_force),
                right_force=quantize_skin_force(right_force),
                timestamp=float(sim.time),
            )

        def scratch_grip_metrics() -> tuple[float, float, float, float]:
            cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
            pinch = scratch_gripper_object_center(sim, xmat)
            delta = cube - pinch
            open_axis = _unit(
                xmat[:, 1], np.array([0.0, 1.0, 0.0], dtype=np.float64))
            face_axis = _unit(
                xmat[:, 0], np.array([1.0, 0.0, 0.0], dtype=np.float64))
            approach_axis = _unit(
                xmat[:, 2], np.array([0.0, 0.0, -1.0], dtype=np.float64))
            return (
                float(np.linalg.norm(delta)),
                abs(float(np.dot(delta, open_axis))),
                abs(float(np.dot(delta, face_axis))),
                abs(float(np.dot(delta, approach_axis))),
            )

        set_joint_positions(model, sim, arm_joints, plan["grasp"].angles)
        for item in gripper_limits:
            sim.qpos[model.jnt_qposadr[item["joint"]]] = item["open"]
        write_arm(plan["grasp"].angles)
        write_gripper(0.0)
        mujoco.mj_forward(model, sim)

        for _ in range(50):
            write_arm(plan["grasp"].angles)
            write_gripper(0.0)
            mujoco.mj_step(model, sim)

        pre_left, pre_right, pre_contacts = scratch_finger_contact_sides(sim)
        pre_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        pre_pair_skew, pre_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        pre_min_dist = scratch_cube_robot_min_contact_dist(sim)
        if (scratch_nonfinger_robot_contact(sim) or
                pre_min_dist < -EARLY_CONTACT_MAX_PENETRATION or
                pre_contacts > 0):
            penetration = 0.0 if pre_min_dist == float("inf") else max(0.0, -pre_min_dist)
            return (
                120.0 + penetration * 1000.0 + pre_contacts * 5.0 +
                max(0.0, pre_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                max(0.0, pre_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
            )

        grip_sim = SmoothGripperController(
            model, sim, gripper_limits, joint_to_actuator)
        grip_sim.close(GRIP_CLOSE_FRAMES)
        for frame_idx in range(GRIP_CLOSE_TIMEOUT_FRAMES):
            write_arm(plan["grasp"].angles)
            grip_sim.step()
            step_substeps()
            if grip_sim.done and frame_idx > GRIP_CLOSE_FRAMES:
                break

        left_hit, right_hit, contact_count = scratch_finger_contact_sides(sim)
        close_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        close_pair_skew, close_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        close_min_dist = scratch_cube_robot_min_contact_dist(sim)
        if (not left_hit or not right_hit or
                not contact_patch_is_large(contact_count, close_patch_span) or
                close_pair_skew > MAX_CONTACT_PAIR_SKEW or
                close_contact_center_err > MAX_CONTACT_CENTER_ERR or
                close_min_dist > CONTACT_CONFIRM_MAX_DIST or
                scratch_nonfinger_robot_contact(sim)):
            missing = 0
            if not left_hit:
                missing += 1
            if not right_hit:
                missing += 1
            gap_penalty = 0.0
            if close_min_dist != float("inf"):
                gap_penalty = max(0.0, close_min_dist - CONTACT_CONFIRM_MAX_DIST) * 1000.0
            area_penalty = (
                max(0, MIN_STABLE_FINGER_CONTACTS - contact_count) * 6.0 +
                max(0, TARGET_FINGER_CONTACTS - contact_count) * 1.5 +
                max(0.0, MIN_CONTACT_DIVERSITY - close_patch_span) * 900.0 +
                max(0.0, close_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                max(0.0, close_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
            )
            return 80.0 + 10.0 * missing + area_penalty + gap_penalty
        center_err, open_err, face_err, approach_err = scratch_grip_metrics()
        if (center_err > GRIP_LOCK_MAX_ERR or
                open_err > GRIP_LOCK_MAX_OPEN_AXIS_ERR or
                face_err > GRIP_LOCK_MAX_FACE_AXIS_ERR or
                approach_err > GRIP_LOCK_MAX_APPROACH_AXIS_ERR):
            return (
                55.0 +
                max(0.0, center_err - GRIP_LOCK_MAX_ERR) * 200.0 +
                max(0.0, open_err - GRIP_LOCK_MAX_OPEN_AXIS_ERR) * 250.0 +
                max(0.0, face_err - GRIP_LOCK_MAX_FACE_AXIS_ERR) * 200.0 +
                max(0.0, approach_err - GRIP_LOCK_MAX_APPROACH_AXIS_ERR) * 200.0
            )

        grip_sim.hold(GRIP_HOLD_FRAMES)
        for frame_idx in range(GRIP_HOLD_FRAMES):
            write_arm(plan["grasp"].angles)
            grip_sim.maintain_tactile_force(
                scratch_skin_reading(),
                min_force=SKIN_HOLD_MIN_FORCE,
                target_force=SKIN_HOLD_TARGET_FORCE,
                max_force=SKIN_HOLD_MAX_FORCE,
            )
            grip_sim.step()
            step_substeps()
            left_hit, right_hit, contact_count = scratch_finger_contact_sides(sim)
            hold_patch_span = scratch_finger_contact_patch_span(sim, xmat)
            hold_pair_skew, hold_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
            hold_min_dist = scratch_cube_robot_min_contact_dist(sim)
            if (not left_hit or not right_hit or
                    not contact_patch_is_large(contact_count, hold_patch_span) or
                    hold_pair_skew > MAX_CONTACT_PAIR_SKEW or
                    hold_contact_center_err > MAX_CONTACT_CENTER_ERR or
                    hold_min_dist > CONTACT_CONFIRM_MAX_DIST or
                    scratch_nonfinger_robot_contact(sim)):
                return (
                    70.0 +
                    max(0, MIN_STABLE_FINGER_CONTACTS - contact_count) * 8.0 +
                    max(0.0, MIN_CONTACT_DIVERSITY - hold_patch_span) * 900.0 +
                    max(0.0, hold_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                    max(0.0, hold_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
                )

        grip_sim.hold(GRIP_TRANSPORT_HOLD_FRAMES)
        z_before = float(sim.qpos[ball_qpos_adr + 2])
        arm_sim = SmoothArmController(model, sim, arm_joints, joint_to_actuator)
        arm_sim.set_target(plan["lift"].angles,
                           speed=SPEED_LIFT, min_frames=LIFT_MIN_FRAMES)
        for frame_idx in range(LIFT_MIN_FRAMES + 180):
            arm_sim.step()
            grip_sim.maintain_tactile_force(
                scratch_skin_reading(),
                min_force=SKIN_HOLD_MIN_FORCE,
                target_force=SKIN_HOLD_TARGET_FORCE,
                max_force=SKIN_HOLD_MAX_FORCE,
            )
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > LIFT_MIN_FRAMES - 40:
                break

        cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        err = float(np.linalg.norm(cube - scratch_gripper_object_center(sim, xmat)))
        lifted = float(cube[2] - z_before)
        left_hit, right_hit, contacts = scratch_finger_contact_sides(sim)
        lift_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        lift_pair_skew, lift_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        lift_min_dist = scratch_cube_robot_min_contact_dist(sim)
        center_err, open_err, face_err, approach_err = scratch_grip_metrics()
        lift_preview_min = GRASP_LIFT_HEIGHT * 0.70
        if (not left_hit or not right_hit or
                not contact_patch_is_large(contacts, lift_patch_span) or
                lift_pair_skew > MAX_CONTACT_PAIR_SKEW or
                lift_contact_center_err > MAX_CONTACT_CENTER_ERR or
                scratch_nonfinger_robot_contact(sim) or
                lift_min_dist > CONTACT_CONFIRM_MAX_DIST or
                lifted < lift_preview_min or err > min(CARRY_MAX_ERR, 0.028)):
            return (
                60.0 +
                max(0.0, lift_preview_min - lifted) * 120.0 +
                err * 25.0 +
                max(0, MIN_STABLE_FINGER_CONTACTS - contacts) * 8.0 +
                max(0.0, MIN_CONTACT_DIVERSITY - lift_patch_span) * 900.0 +
                max(0.0, lift_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                max(0.0, lift_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
            )
        if (open_err > GRIP_LOCK_MAX_OPEN_AXIS_ERR * 1.15 or
                face_err > GRIP_LOCK_MAX_FACE_AXIS_ERR * 1.15 or
                approach_err > GRIP_LOCK_MAX_APPROACH_AXIS_ERR * 1.15):
            return 50.0 + center_err * 40.0

        arm_sim.set_target(plan["carry_mid"].angles,
                           speed=SPEED_CARRY, min_frames=CARRY_MID_MIN_FRAMES)
        for frame_idx in range(CARRY_MID_MIN_FRAMES + 180):
            arm_sim.step()
            grip_sim.maintain_tactile_force(
                scratch_skin_reading(),
                min_force=SKIN_HOLD_MIN_FORCE,
                target_force=SKIN_HOLD_TARGET_FORCE,
                max_force=SKIN_HOLD_MAX_FORCE,
            )
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > CARRY_MID_MIN_FRAMES - 40:
                break

        cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        mid_err = float(np.linalg.norm(cube - scratch_gripper_object_center(sim, xmat)))
        mid_left, mid_right, mid_contacts = scratch_finger_contact_sides(sim)
        mid_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        mid_pair_skew, mid_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        mid_min_dist = scratch_cube_robot_min_contact_dist(sim)
        if (not mid_left or not mid_right or
                not contact_patch_is_large(mid_contacts, mid_patch_span) or
                mid_pair_skew > MAX_CONTACT_PAIR_SKEW or
                mid_contact_center_err > MAX_CONTACT_CENTER_ERR or
                scratch_nonfinger_robot_contact(sim) or
                mid_min_dist > CONTACT_CONFIRM_MAX_DIST or
                mid_err > min(CARRY_MAX_ERR, 0.030)):
            if scratch_cube_inside_box():
                return mid_err * 3.0
            return (
                85.0 +
                mid_err * 35.0 +
                max(0, MIN_STABLE_FINGER_CONTACTS - mid_contacts) * 12.0 +
                max(0.0, MIN_CONTACT_DIVERSITY - mid_patch_span) * 900.0 +
                max(0.0, mid_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                max(0.0, mid_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
            )

        arm_sim.set_target(plan["place_above"].angles,
                           speed=SPEED_CARRY, min_frames=CARRY_STEP_MIN_FRAMES)
        for frame_idx in range(CARRY_STEP_MIN_FRAMES + 180):
            arm_sim.step()
            grip_sim.maintain_tactile_force(
                scratch_skin_reading(),
                min_force=SKIN_HOLD_MIN_FORCE,
                target_force=SKIN_HOLD_TARGET_FORCE,
                max_force=SKIN_HOLD_MAX_FORCE,
            )
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > CARRY_STEP_MIN_FRAMES - 25:
                break

        cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        carry_err = float(np.linalg.norm(cube - scratch_gripper_object_center(sim, xmat)))
        carry_left, carry_right, carry_contacts = scratch_finger_contact_sides(sim)
        carry_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        carry_pair_skew, carry_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        carry_min_dist = scratch_cube_robot_min_contact_dist(sim)
        if (not carry_left or not carry_right or
                not contact_patch_is_large(carry_contacts, carry_patch_span) or
                carry_pair_skew > MAX_CONTACT_PAIR_SKEW or
                carry_contact_center_err > MAX_CONTACT_CENTER_ERR or
                scratch_nonfinger_robot_contact(sim) or
                carry_min_dist > CONTACT_CONFIRM_MAX_DIST or
                carry_err > min(CARRY_MAX_ERR, 0.030)):
            if scratch_cube_inside_box():
                return carry_err * 3.0
            return (
                90.0 +
                carry_err * 35.0 +
                max(0, MIN_STABLE_FINGER_CONTACTS - carry_contacts) * 12.0 +
                max(0.0, MIN_CONTACT_DIVERSITY - carry_patch_span) * 900.0 +
                max(0.0, carry_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                max(0.0, carry_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
            )

        arm_sim.set_target(plan["place_drop"].angles,
                           speed=SPEED_PLACE, min_frames=PLACE_DROP_MIN_FRAMES)
        for frame_idx in range(PLACE_DROP_MIN_FRAMES + 180):
            arm_sim.step()
            grip_sim.maintain_tactile_force(
                scratch_skin_reading(),
                min_force=SKIN_HOLD_MIN_FORCE,
                target_force=SKIN_HOLD_TARGET_FORCE,
                max_force=SKIN_HOLD_MAX_FORCE,
            )
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > PLACE_DROP_MIN_FRAMES - 40:
                break

        cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        low_ready, low_dx, low_dy, low_z, low_target_err = scratch_cube_release_window()
        drop_left, drop_right, drop_contacts = scratch_finger_contact_sides(sim)
        drop_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        drop_pair_skew, drop_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        drop_min_dist = scratch_cube_robot_min_contact_dist(sim)
        held_low = (
            drop_left and drop_right and
            contact_patch_is_large(drop_contacts, drop_patch_span) and
            drop_pair_skew <= MAX_CONTACT_PAIR_SKEW and
            drop_contact_center_err <= MAX_CONTACT_CENTER_ERR and
            drop_min_dist <= CONTACT_CONFIRM_MAX_DIST and
            not scratch_nonfinger_robot_contact(sim)
        )
        if not low_ready or (not held_low and not scratch_cube_inside_box()):
            return (
                115.0 +
                low_target_err * 350.0 +
                max(0.0, low_dx - (BOX_SIZE[0] - BOX_WALL - CUBE_HALF_SIZE -
                                   PLACE_RELEASE_XY_MARGIN)) * 2500.0 +
                max(0.0, low_dy - (BOX_SIZE[1] - BOX_WALL - CUBE_HALF_SIZE -
                                   PLACE_RELEASE_XY_MARGIN)) * 2500.0 +
                max(0, MIN_STABLE_FINGER_CONTACTS - drop_contacts) * 12.0 +
                max(0.0, MIN_CONTACT_DIVERSITY - drop_patch_span) * 900.0 +
                max(0.0, drop_pair_skew - MAX_CONTACT_PAIR_SKEW) * 1000.0 +
                max(0.0, drop_contact_center_err - MAX_CONTACT_CENTER_ERR) * 1000.0
            )

        grip_sim.open(duration_frames=PLACE_RELEASE_OPEN_FRAMES)
        for frame_idx in range(PLACE_RELEASE_OPEN_FRAMES + BOX_VERIFY_FRAMES + 80):
            write_arm(plan["place_drop"].angles)
            grip_sim.step()
            step_substeps()
            if grip_sim.done and frame_idx >= PLACE_RELEASE_OPEN_FRAMES:
                bv = float(np.linalg.norm(sim.qvel[ball_dof_adr:ball_dof_adr + 3]))
                if frame_idx >= PLACE_RELEASE_OPEN_FRAMES + BOX_VERIFY_FRAMES or bv < 0.010:
                    break

        if not scratch_cube_inside_box():
            cube_after = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
            box_eval = normalize_box_center_estimate(box_xyz)
            miss_xy = float(np.linalg.norm(cube_after[:2] - box_eval[:2]))
            miss_z = abs(float(cube_after[2] - place_release_cube_center(box_eval)[2]))
            return 145.0 + miss_xy * 1200.0 + miss_z * 500.0

        contact_bonus_penalty = max(
            0, TARGET_FINGER_CONTACTS - min(
                contact_count, contacts, mid_contacts, carry_contacts, drop_contacts)
        ) * 0.6
        patch_bonus_penalty = max(
            0.0,
            MIN_CONTACT_DIVERSITY * 1.5 -
            min(close_patch_span, lift_patch_span, mid_patch_span,
                carry_patch_span, drop_patch_span),
        ) * 120.0
        center_quality_penalty = max(
            0.0,
            max(
                close_contact_center_err,
                hold_contact_center_err,
                lift_contact_center_err,
                mid_contact_center_err,
                carry_contact_center_err,
                drop_contact_center_err,
            ) - PREFERRED_CONTACT_CENTER_ERR,
        ) * 900.0
        skew_quality_penalty = max(
            0.0,
            max(
                close_pair_skew,
                hold_pair_skew,
                lift_pair_skew,
                mid_pair_skew,
                carry_pair_skew,
                drop_pair_skew,
            ) - PREFERRED_CONTACT_PAIR_SKEW,
        ) * 650.0
        return (
            err * 2.0 + open_err * 4.0 + face_err * 2.0 +
            approach_err * GRASP_APPROACH_ERR_SCORE_WEIGHT +
            low_target_err * 35.0 +
            contact_bonus_penalty + patch_bonus_penalty +
            center_quality_penalty + skew_quality_penalty
        )

    def compute_dynamic_ik(
        ball_xyz: np.ndarray,
        box_xyz: np.ndarray,
        grasp_z_offs: float | None = None,
        opening_hints: list[np.ndarray] | None = None,
        preview_top_k: int = PHYSICAL_EVAL_TOP_K,
    ) -> dict[str, IKResult]:
        """Compute a pose-aware IK plan for the detected cube position."""
        nonlocal dynamic_grasp_xmat, dynamic_grasp_z_offset
        dynamic_grasp_xmat = None
        ball_xyz = np.asarray(ball_xyz, dtype=np.float64).copy()
        ball_xyz[2] = float(np.clip(ball_xyz[2], VISION_Z_MIN, VISION_Z_MAX))
        box_xyz = np.asarray(box_xyz, dtype=np.float64).copy()
        box_xyz[2] = float(np.clip(box_xyz[2], CUBE_REST_Z, BOX_DETECTION_Z_MAX))
        z_offset_candidates = (
            list(GRASP_Z_OFFSETS) if grasp_z_offs is None else [float(grasp_z_offs)]
        )
        place_drop_contact_target = adaptive_box_release_cube_center(
            box_xyz,
            ball_xyz,
        )
        place_above_contact_target = place_drop_contact_target.copy()
        place_above_contact_target[2] = carry_height_for_xy(
            place_drop_contact_target[:2],
            box_xyz,
        )

        best_plan: dict[str, IKResult] | None = None
        best_xmat: np.ndarray | None = None
        best_z_offs = float(z_offset_candidates[0])
        best_score = float("inf")
        candidate_plans: list[tuple[float, dict[str, IKResult], np.ndarray, float]] = []
        home_rest = np.array(
            [data.qpos[model.jnt_qposadr[j]] for j in arm_joints],
            dtype=np.float64,
        )

        for current_z_offs in z_offset_candidates:
            grasp_contact_target = ball_xyz + np.array(
                [0.0, 0.0, float(current_z_offs)], dtype=np.float64)
            for xmat in generate_grasp_orientations(ball_xyz, opening_hints):
                z_axis = xmat[:, 2]
                grasp_target = gripper_body_target_for_contact(grasp_contact_target, xmat)
                lift_target = gripper_body_target_for_contact(
                    grasp_contact_target + np.array([0.0, 0.0, GRASP_LIFT_HEIGHT]),
                    xmat,
                )
                place_above_target = gripper_body_target_for_contact(
                    place_above_contact_target,
                    xmat,
                )
                carry_mid_contact_target = carry_midpoint_for_grasp(
                    ball_xyz,
                    box_xyz,
                    xmat,
                )
                carry_mid_target = gripper_body_target_for_contact(
                    carry_mid_contact_target,
                    xmat,
                )
                place_drop_target = gripper_body_target_for_contact(
                    place_drop_contact_target,
                    xmat,
                )
                targets = {
                    "approach": grasp_target - 0.13 * z_axis,
                    "descend_mid": grasp_target - DESCEND_MID_CLEARANCE * z_axis,
                    "grasp": grasp_target,
                    "lift": lift_target,
                    "carry_mid": carry_mid_target,
                    "place_above": place_above_target,
                    "place_drop": place_drop_target,
                }
                scratch = mujoco.MjData(model)
                scratch.qpos[:] = data.qpos[:]
                scratch.qvel[:] = 0.0
                for item in gripper_limits:
                    scratch.qpos[model.jnt_qposadr[item["joint"]]] = item["open"]
                mujoco.mj_forward(model, scratch)

                plan: dict[str, IKResult] = {}
                ok = True
                score = 0.0
                rest = home_rest
                for name, target in targets.items():
                    if name == "approach":
                        # Approach is only a staging pose.  The strict safety
                        # gate is the pre-close alignment check near the cube;
                        # rejecting a 5-8 mm approach error made edge-of-
                        # workspace moved cubes impossible to grasp.
                        pos_tol = APPROACH_IK_POS_TOL
                        ori_tol = GRASP_ORI_TOL
                    elif name in ("descend_mid", "grasp"):
                        pos_tol = GRASP_IK_POS_TOL
                        ori_tol = GRASP_ORI_TOL
                    elif name == "lift":
                        pos_tol = 0.007
                        ori_tol = LIFT_ORI_TOL
                    else:
                        pos_tol = 0.008
                        ori_tol = PLACE_ORI_TOL
                    result = solve_oriented_target(
                        scratch, target, xmat, rest=rest,
                        pos_tol=pos_tol,
                        ori_tol=ori_tol,
                    )
                    relaxed_lift = False
                    if not result.success and name == "lift":
                        # The lift pose is a clearance waypoint.  Being a couple
                        # of centimeters below the requested high clearance is
                        # still safe and avoids rejecting otherwise good grasps.
                        result = solve_oriented_target(
                            scratch, target, xmat, rest=rest,
                            pos_tol=0.025, ori_tol=0.35,
                        )
                        relaxed_lift = result.success
                    relaxed_place = False
                    if (not result.success and
                            name in ("carry_mid", "place_above", "place_drop")):
                        result = solve_oriented_target(
                            scratch, target, xmat, rest=rest,
                            pos_tol=0.012, ori_tol=0.35,
                        )
                        relaxed_place = result.success
                    if not result.success and name in ("carry_mid", "place_above", "place_drop"):
                        result = solve_gripper_center_ik(
                            model, scratch, target,
                            pinch_body_id, pinch_body_id, arm_joints,
                            max_iter=700, tol=0.008,
                        )
                        relaxed_place = result.success
                    plan[name] = result
                    score += result.error_norm * 100.0 + result.orientation_error_norm
                    if relaxed_lift:
                        score += 0.5
                    if relaxed_place:
                        score += 1.4
                    if not result.success:
                        ok = False
                        score += 20.0
                        break
                    set_joint_positions(model, scratch, arm_joints, result.angles)
                    mujoco.mj_forward(model, scratch)
                    rest = result.angles

                if ok:
                    transport_alignment = grasp_transport_alignment(
                        xmat, ball_xyz, box_xyz)
                    if transport_alignment < GRASP_TRANSPORT_TARGET_ALIGNMENT:
                        continue
                    hint_penalty = grasp_opening_hint_score(xmat, opening_hints)
                    if opening_hints and hint_penalty > 1e-6:
                        continue
                    score += 4.0 * max(
                        0.0,
                        GRASP_TRANSPORT_TARGET_ALIGNMENT - transport_alignment,
                    )
                    score += 0.02 * float(np.linalg.norm(plan["grasp"].angles - home_rest))
                    score += grasp_stability_score(xmat, ball_xyz, box_xyz)
                    score += hint_penalty
                    score += grasp_frame_penalty(xmat)
                    score += (
                        abs(float(current_z_offs) - PREFERRED_GRASP_Z_OFFSET) *
                        GRASP_HEIGHT_SCORE_WEIGHT
                    )
                    candidate_plans.append(
                        (score, plan, xmat.copy(), float(current_z_offs)))
                    if score < best_score:
                        best_score = score
                        best_plan = plan
                        best_xmat = xmat.copy()
                        best_z_offs = float(current_z_offs)
                elif best_plan is None and score < best_score:
                    best_score = score
                    best_plan = plan
                    best_z_offs = float(current_z_offs)

        preview_count = max(0, int(preview_top_k))
        if grasp_z_offs is None and candidate_plans:
            preview_count = len(candidate_plans)
        if candidate_plans and preview_count > 0:
            candidate_plans.sort(key=lambda item: item[0])
            preview_best_plan: dict[str, IKResult] | None = None
            preview_best_xmat: np.ndarray | None = None
            preview_best_score = float("inf")
            for idx, (base_score, plan, xmat, current_z_offs) in enumerate(
                    candidate_plans[:preview_count], start=1):
                penalty = physical_plan_penalty(plan, ball_xyz, box_xyz, xmat)
                total = base_score + penalty
                y_axis = xmat[:, 1]
                z_axis = xmat[:, 2]
                transport_alignment = grasp_transport_alignment(
                    xmat, ball_xyz, box_xyz)
                print(f"    Physical preview {idx}: total={total:.2f} "
                      f"penalty={penalty:.2f} "
                      f"open={np.round(y_axis[:2], 2)} "
                      f"approach={np.round(z_axis, 2)} "
                      f"z_off={current_z_offs*1000:.0f}mm "
                      f"transport_align={transport_alignment:.2f}")
                if transport_alignment < GRASP_TRANSPORT_TARGET_ALIGNMENT:
                    continue
                if total < preview_best_score:
                    preview_best_score = total
                    preview_best_plan = plan
                    preview_best_xmat = xmat.copy()
                    best_z_offs = float(current_z_offs)
            if preview_best_plan is not None:
                best_plan = preview_best_plan
                best_xmat = preview_best_xmat

        if (best_plan is None or
            set(best_plan) != {
                "approach", "descend_mid", "grasp",
                "lift", "carry_mid", "place_above", "place_drop",
            } or
            any(not result.success for result in best_plan.values())):
            print("    Pose IK incomplete; rejecting position-only fallback "
                  "because it can approach the cube with the wrong gripper "
                  "orientation.")
            dynamic_grasp_xmat = None
            dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
            return {}

        if best_xmat is not None:
            dynamic_grasp_xmat = best_xmat.copy()
            dynamic_grasp_z_offset = float(best_z_offs)
            print(f"    Selected grasp frame: "
                  f"open={np.round(best_xmat[:, 1], 2)} "
                  f"approach={np.round(best_xmat[:, 2], 2)} "
                  f"grasp_z_offset={dynamic_grasp_z_offset*1000:.0f}mm")
        for name in (
            "approach", "descend_mid", "grasp",
            "lift", "carry_mid", "place_above", "place_drop",
        ):
            result = best_plan[name]
            status = "reachable" if result.success else "UNREACHABLE"
            print(f"  IK {name:8s}: {status}, pos={result.error_norm:.4f}m, "
                  f"ori={result.orientation_error_norm:.3f}, "
                  f"target={np.round(result.target, 3)}")
        return best_plan

    def cube_carry_metrics() -> tuple[float, float]:
        gc = gripper_object_center()
        cube = current_ball_xyz()
        err = float(np.linalg.norm(cube - gc))
        lifted = float(cube[2] - CUBE_REST_Z)
        return err, lifted

    def cube_is_secured(*, strict_visual: bool = False) -> bool:
        nonlocal last_visual_secure_override_frame
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        patch_span = finger_contact_patch_span()
        pair_skew, contact_center_err = finger_contact_alignment()
        skin = read_skin()
        contact_pair_ok = left_pad_hit and right_pad_hit and pad_contacts >= MIN_STABLE_FINGER_CONTACTS
        contact_ok = (
            left_pad_hit and right_pad_hit and
            contact_patch_is_large(pad_contacts, patch_span) and
            pair_skew <= MAX_CONTACT_PAIR_SKEW and
            contact_center_err <= MAX_CONTACT_CENTER_ERR
        )
        geometry_ok = lifted > CARRY_MIN_LIFT and err < CARRY_MAX_ERR
        tactile_lock_ok = (
            lifted > CARRY_MIN_LIFT and
            err < TACTILE_SECURE_OVERRIDE_MAX_ERR and
            contact_ok and
            skin_squeeze_ok(skin, TACTILE_SECURE_OVERRIDE_MIN_FORCE) and
            visual_grip_slip_m <= 0.018 and
            visual_grip_drop_m <= 0.010
        )
        visual_ok = (
            visual_grip_reference_offset is None or
            (
                visual_grip_slip_m <= VISUAL_GRIP_ERR_HARD_M and
                visual_grip_drop_m <= VISUAL_GRIP_DROP_HARD_M
            ) or
            (not strict_visual and visual_grip_miss_count > 2)
        )
        tactile_pair_ok = (
            contact_pair_ok and
            skin_squeeze_ok(skin, SKIN_HOLD_MIN_FORCE) and
            visual_ok
        )
        if geometry_ok and (contact_ok or tactile_pair_ok) and visual_ok:
            return True
        strict_tactile_override = (
            contact_pair_ok and
            geometry_ok and
            skin_squeeze_ok(skin, SKIN_HOLD_TARGET_FORCE) and
            visual_grip_slip_m <= 0.012 and
            visual_grip_drop_m <= 0.006
        )
        if strict_visual and not visual_ok and not strict_tactile_override:
            if frame - last_visual_secure_override_frame >= TACTILE_SECURE_LOG_EVERY_N:
                last_visual_secure_override_frame = frame
                print(f"    Strict lift grip check blocked transport: "
                      f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                      f"contacts={pad_contacts} err={err*100:.1f}cm "
                      f"lift={lifted*100:.1f}cm "
                      f"{visual_grip_summary()}")
            return False
        if strict_visual and not visual_ok and strict_tactile_override:
            if frame - last_visual_secure_override_frame >= TACTILE_SECURE_LOG_EVERY_N:
                last_visual_secure_override_frame = frame
                print(f"    Strict lift grip check accepted strong tactile hold: "
                      f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                      f"contacts={pad_contacts} err={err*100:.1f}cm "
                      f"lift={lifted*100:.1f}cm "
                      f"{visual_grip_summary()}")
            return True
        if not visual_ok and tactile_lock_ok:
            if frame - last_visual_secure_override_frame >= TACTILE_SECURE_LOG_EVERY_N:
                last_visual_secure_override_frame = frame
                print(f"    Tactile/contact grip trusted over visual slip: "
                      f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                      f"contacts={pad_contacts} err={err*100:.1f}cm "
                      f"lift={lifted*100:.1f}cm "
                      f"{visual_grip_summary()}")
            return True
        return False

    def cube_transport_safe(label: str, *, require_high_lift: bool = True) -> bool:
        nonlocal transport_slip_abort_frames, last_visual_secure_override_frame
        if cube_inside_box():
            transport_slip_abort_frames = 0
            return True
        contact_ok_now = cube_transport_contact_ok()
        relaxed_hold_now = cube_transport_relaxed_hold_ok()
        geometry_hold_now = cube_transport_geometry_hold_ok()
        stiction_hold_now = cube_transport_stiction_hold_ok()
        visual_ready = visual_grip_reference_offset is not None and visual_grip_weight > 0.35
        moderate_slip = visual_ready and (
            visual_grip_slip_m > VISUAL_TRANSPORT_ABORT_SLIP_M or
            visual_grip_drop_m > VISUAL_TRANSPORT_ABORT_DROP_M
        )
        severe_slip = visual_ready and (
            visual_grip_slip_m > VISUAL_TRANSPORT_SEVERE_SLIP_M or
            visual_grip_drop_m > VISUAL_TRANSPORT_SEVERE_DROP_M
        )
        if ((moderate_slip or severe_slip) and
                (contact_ok_now or relaxed_hold_now or geometry_hold_now or
                 stiction_hold_now)):
            transport_slip_abort_frames = 0
            if frame - last_visual_secure_override_frame >= TACTILE_SECURE_LOG_EVERY_N:
                last_visual_secure_override_frame = frame
                print(f"    Transport visual slip accepted by tactile hold "
                      f"({label}): {visual_grip_summary()}")
            return True
        if moderate_slip:
            transport_slip_abort_frames += 1
        else:
            transport_slip_abort_frames = max(0, transport_slip_abort_frames - 1)
        if severe_slip or transport_slip_abort_frames >= VISUAL_TRANSPORT_ABORT_FRAMES:
            print(f"    Transport slip guard ({label}) blocked path: "
                  f"{visual_grip_summary()} "
                  f"frames={transport_slip_abort_frames}/"
                  f"{VISUAL_TRANSPORT_ABORT_FRAMES}")
            return False
        if require_high_lift and not (
                contact_ok_now or relaxed_hold_now or geometry_hold_now or
                stiction_hold_now):
            print(f"    Transport contact guard ({label}) blocked path: "
                  f"{visual_grip_summary()}")
            return False
        return True

    def cube_transport_contact_ok() -> bool:
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        patch_span = finger_contact_patch_span()
        pair_skew, contact_center_err = finger_contact_alignment()
        skin = read_skin()
        return (
            lifted > CARRY_MIN_LIFT and
            err < TACTILE_SECURE_OVERRIDE_MAX_ERR and
            left_pad_hit and right_pad_hit and
            contact_patch_is_large(pad_contacts, patch_span) and
            pair_skew <= MAX_CONTACT_PAIR_SKEW and
            contact_center_err <= MAX_CONTACT_CENTER_ERR and
            skin_squeeze_ok(skin, TACTILE_SECURE_OVERRIDE_MIN_FORCE)
        )

    def cube_transport_relaxed_hold_ok() -> bool:
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        skin = read_skin()
        return (
            lifted > CARRY_MIN_LIFT and
            err < PLACE_HELD_RECOVERY_MAX_ERR and
            left_pad_hit and right_pad_hit and
            pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
            skin_squeeze_ok(skin, SKIN_HOLD_TARGET_FORCE)
        )

    def cube_transport_geometry_hold_ok() -> bool:
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        patch_span = finger_contact_patch_span()
        pair_skew, contact_center_err = finger_contact_alignment()
        return (
            lifted > CARRY_MIN_LIFT and
            err < 0.018 and
            left_pad_hit and right_pad_hit and
            pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
            contact_patch_is_large(pad_contacts, patch_span) and
            pair_skew <= MAX_CONTACT_PAIR_SKEW and
            contact_center_err <= MAX_CONTACT_CENTER_ERR and
            visual_grip_drop_m <= 0.008
        )

    def cube_transport_stiction_hold_ok() -> bool:
        if not grip_locked:
            return False
        err, lifted = cube_carry_metrics()
        lin_v, _ = cube_velocity_norms()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        skin = read_skin()
        contact_rich_centered = (
            pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
            err <= 0.012 and
            (left_pad_hit or right_pad_hit)
        )
        centered_single_contact = (
            pad_contacts >= 1 and
            (left_pad_hit or right_pad_hit) and
            err <= 0.012 and
            max(skin.left_force, skin.right_force) >= SKIN_TOUCH_FORCE and
            lin_v <= 0.12 and
            visual_grip_drop_m <= 0.006
        )
        return (
            lifted > CARRY_MIN_LIFT and
            err < GRIP_STICTION_MAX_ERR and
            pad_contacts >= 1 and
            (left_pad_hit or right_pad_hit) and
            (
                max(skin.left_force, skin.right_force) >= SKIN_CONFIRM_FORCE or
                contact_rich_centered or
                centered_single_contact
            ) and
            visual_grip_drop_m <= 0.012
        )

    def carry_segment_can_advance(
        min_frames: int,
        *,
        qerr_limit: float = 0.060,
        qvel_limit: float = 0.120,
    ) -> bool:
        if carry_motion_frames < int(min_frames):
            return False
        q_err, max_qvel = arm_ctrl.settle_metrics()
        return q_err <= qerr_limit and max_qvel <= qvel_limit

    def begin_transport_regrip(reason: str, resume_sub: int) -> bool:
        nonlocal sub, transport_regrip_count, transport_regrip_frames
        nonlocal transport_regrip_stable_frames, transport_regrip_resume_sub
        nonlocal transport_slip_abort_frames, status_msg

        if transport_regrip_count >= TRANSPORT_REGRIP_MAX_COUNT:
            return False
        err, lifted = cube_carry_metrics()
        lin_v, _ = cube_velocity_norms()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        skin = read_skin()
        still_near_gripper = (
            lifted > CARRY_MIN_LIFT and
            err <= TRANSPORT_REGRIP_MAX_ERR and
            lin_v <= TRANSPORT_REGRIP_MAX_SPEED
        )
        centered_gap_can_be_caught = (
            grip_locked and
            lifted > CARRY_MIN_LIFT and
            err <= 0.006 and
            lin_v <= TRANSPORT_REGRIP_MAX_SPEED and
            visual_grip_drop_m <= 0.006
        )
        has_recoverable_contact = (
            pad_contacts >= 1 or
            max(skin.left_force, skin.right_force) >= SKIN_CONFIRM_FORCE or
            centered_gap_can_be_caught
        )
        if not (still_near_gripper and has_recoverable_contact):
            return False

        transport_regrip_count += 1
        transport_regrip_frames = 0
        transport_regrip_stable_frames = 0
        transport_regrip_resume_sub = int(resume_sub)
        transport_slip_abort_frames = 0
        arm_ctrl.set_target(arm_ctrl.current(), speed=SPEED_CARRY * 0.35,
                            min_frames=1)
        gripper_ctrl.hold(duration_frames=GRIP_TRANSPORT_HOLD_FRAMES)
        reset_visual_grip_feedback(f"transport regrip: {reason}")
        sub = -30
        status_msg = "Recovering transport grip before continuing ..."
        print(f"    Transport re-grip recovery "
              f"{transport_regrip_count}/{TRANSPORT_REGRIP_MAX_COUNT}: "
              f"{reason}; err={err*100:.1f}cm lift={lifted*100:.1f}cm "
              f"contacts={pad_contacts} left={left_pad_hit} right={right_pad_hit} "
              f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N")
        return True

    def recover_place_while_holding(reason: str) -> bool:
        nonlocal phase, sub, detected_box_pos, box_rgbd_scan_estimates
        nonlocal box_scan_targets, box_scan_idx, box_scan_hint_xy_used
        nonlocal place_settle_frames, place_correction_count, status_msg
        nonlocal transport_slip_abort_frames
        nonlocal carry_motion_frames, carry_last_remaining_xy, carry_stall_steps

        if phase != 6:
            return False

        if cube_inside_box():
            cube = current_ball_xyz()
            print(f"    Placement recovery: cube is already supported in box; "
                  f"continuing to soft release. cube={np.round(cube, 4)}")
            phase = 7
            sub = 0
            return True

        cube = current_ball_xyz()
        err, lifted = cube_carry_metrics()
        skin = read_skin()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        actual_contact = pad_contacts >= MIN_STABLE_FINGER_CONTACTS
        bilateral_contact = left_pad_hit and right_pad_hit
        tactile_pair = skin_squeeze_ok(skin, SKIN_FORCE_MIN_RECOGNITION_N)
        near_gripper = err <= PLACE_HELD_RECOVERY_MAX_ERR
        elevated = (
            lifted >= PLACE_HELD_RECOVERY_MIN_LIFT or
            cube[2] > CUBE_STATIC_ANCHOR_MAX_Z
        )
        still_carried = bool(
            gripper_ctrl.holding and
            near_gripper and
            elevated and
            actual_contact and
            (bilateral_contact or tactile_pair)
        )
        if not still_carried:
            return False

        print(f"    Placement recovery keeps gripper closed ({reason}): "
              f"cube={np.round(cube, 4)} err={err*100:.1f}cm "
              f"lift={lifted*100:.1f}cm "
              f"finger_contacts={pad_contacts} "
              f"left={left_pad_hit} right={right_pad_hit} "
              f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N. "
              "Continuing placement instead of opening in mid-air.")
        ensure_transport_grip_hold(f"place recovery: {reason}")
        maintain_fused_grip_force(
            f"place recovery: {reason}",
            reading=skin,
            force_visual=True,
        )

        if box_target_center() is not None:
            box_now = box_target_center()
            remaining_xy = float(np.linalg.norm(
                box_now[:2] - current_ball_xyz()[:2]))
            entry_gap = box_entry_gap_xy(current_ball_xyz(), box_now)
            direct_recovery = entry_gap <= CARRY_STEP_FINAL_DIRECT_XY_M
            if update_place_plan_for_carried_offset(
                    f"held placement recovery: {reason}",
                    direct_to_box=direct_recovery):
                reset_visual_grip_feedback(f"held placement recovery: {reason}")
                update_visual_grip_feedback(
                    f"held placement recovery: {reason}",
                    force=True,
                )
                transport_slip_abort_frames = 0
                if cube_over_box_entry() or cube_ready_for_guided_place_drop():
                    print("    Held cube is inside the box opening corridor; "
                          "lowering for guarded release.")
                    arm_ctrl.set_target(
                        dynamic_ik_plan["place_drop"].angles,
                        speed=SPEED_PLACE,
                        min_frames=max(80, PLACE_DROP_MIN_FRAMES // 2),
                    )
                    carry_motion_frames = 0
                    sub = 3
                    status_msg = "Lowering held cube into box after recovery ..."
                    return True
                print(f"    Held cube still {remaining_xy*100:.1f}cm from box "
                      f"(entry_gap={entry_gap*100:.1f}cm); "
                      "continuing with a bounded carry step.")
                arm_ctrl.set_target(
                    dynamic_ik_plan["place_above"].angles,
                    speed=SPEED_CARRY * (0.38 if direct_recovery else 0.45),
                    min_frames=max(CARRY_STEP_MIN_FRAMES * 2, 190),
                )
                carry_motion_frames = 0
                sub = 2
                status_msg = "Continuing held cube toward box ..."
                return True
            print("    Held placement recovery could not update the carry plan; "
                  "falling back to a bounded box re-scan.")

        detected_box_pos = None
        box_rgbd_scan_estimates = []
        box_scan_targets = []
        box_scan_idx = 0
        box_scan_hint_xy_used = None
        place_settle_frames = 0
        place_correction_count = 0

        place_above = dynamic_ik_plan.get("place_above")
        if place_above is not None and place_above.success:
            arm_ctrl.set_target(
                place_above.angles,
                speed=SPEED_PLACE,
                min_frames=max(PLACE_RESCAN_RETURN_MIN_FRAMES,
                               PLACE_ABOVE_MIN_FRAMES // 2),
            )
            carry_motion_frames = 0
            sub = -21
            status_msg = "Keeping cube held; returning above box for re-scan ..."
        else:
            arm_ctrl.set_target(arm_ctrl.current(), speed=SPEED_SCAN, min_frames=1)
            sub = -20
            status_msg = "Keeping cube held; re-scanning target box ..."
        return True

    def pre_close_alignment_ok(label: str) -> bool:
        cube = current_ball_xyz()
        object_center = gripper_object_center()
        xy_err = float(np.linalg.norm((object_center - cube)[:2]))
        z_err = abs(float(object_center[2] - cube[2]))
        ori_err = 0.0
        if dynamic_grasp_xmat is not None:
            opening_axis = _unit(
                dynamic_grasp_xmat[:, 1],
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
            face_axis = _unit(
                dynamic_grasp_xmat[:, 0],
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
            actual_xmat = data.xmat[tool_id].reshape(3, 3)
            ori_err = orientation_error_norm(actual_xmat, dynamic_grasp_xmat)
        else:
            left = data.xpos[left_id].copy()
            right = data.xpos[right_id].copy()
            span = right - left
            opening_axis = _unit(span, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            face_axis = _unit(
                np.cross(opening_axis, np.array([0.0, 0.0, -1.0])),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
        center_delta = cube - object_center
        open_axis_err = abs(float(np.dot(center_delta, opening_axis)))
        face_axis_err = abs(float(np.dot(center_delta, face_axis)))
        center_err = float(np.linalg.norm(center_delta))
        bad_contact, bad_name = cube_nonfinger_robot_contact()
        print(f"    {label}: cube={np.round(cube, 4)} "
              f"grasp_center={np.round(object_center, 4)} "
              f"xy_err={xy_err*100:.1f}cm z_err={z_err*100:.1f}cm "
              f"open_axis={open_axis_err*100:.1f}cm "
              f"face_axis={face_axis_err*100:.1f}cm "
              f"center={center_err*100:.1f}cm "
              f"ori_err={ori_err:.3f}")
        if bad_contact:
            print(f"    {label}: rejected, cube is touching non-finger robot part "
                  f"{bad_name}")
            return False
        return (
            xy_err <= PRE_CLOSE_MAX_XY_ERR and
            z_err <= PRE_CLOSE_MAX_Z_ERR and
            open_axis_err <= PRE_CLOSE_MAX_OPEN_AXIS_ERR and
            face_axis_err <= PRE_CLOSE_MAX_FACE_AXIS_ERR and
            center_err <= PRE_CLOSE_MAX_CENTER_ERR and
            ori_err <= PRE_CLOSE_MAX_ORI_ERR
        )

    def restart_search(reason: str) -> None:
        nonlocal phase, sub, scan_idx, scan_targets, detected_ball_pos
        nonlocal box_scan_idx, box_scan_targets, box_search_hint_xy
        nonlocal box_scan_hint_xy_used
        nonlocal detected_opening_hints, dynamic_ik_plan, regrasp_count, scan_round, status_msg
        nonlocal pregrasp_replan_count, local_replan_count, dynamic_grasp_xmat
        nonlocal dynamic_grasp_z_offset, place_settle_frames, place_correction_count
        nonlocal place_support_rescan_count, place_release_support_override
        nonlocal carried_cube_offset
        nonlocal prelift_hold_frames, prelift_ready_frames, lift_motion_frames
        nonlocal carry_motion_frames, carry_last_remaining_xy, carry_stall_steps
        nonlocal transport_slip_abort_frames
        nonlocal transport_regrip_count, transport_regrip_frames
        nonlocal transport_regrip_stable_frames, transport_regrip_resume_sub
        nonlocal rgbd_scan_estimates, rgbd_scan_hints
        nonlocal recovery_wait_frames, recovery_stable_frames, recovery_reason
        if recover_place_while_holding(reason):
            return
        print(f"    Restarting search: {reason}")
        release_grip_lock("restart search")
        release_cube_static_anchor("restart search")
        cube_now = current_ball_xyz()
        placement_or_transport_failure = any(
            key in reason for key in (
                "carry",
                "transport",
                "release",
                "place",
                "box",
                "slipping",
                "dropped",
            )
        )
        high_cube_recovery = (
            ball_qpos_adr >= 0 and
            (
                cube_now[2] > CUBE_STATIC_ANCHOR_MAX_Z or
                (
                    placement_or_transport_failure and
                    cube_now[2] > CUBE_REST_Z + 0.010
                )
            ) and
            not cube_inside_box()
        )
        if high_cube_recovery:
            print(f"    High cube recovery before rescan: "
                  f"cube={np.round(cube_now, 4)} "
                  f"max_anchor_z={CUBE_STATIC_ANCHOR_MAX_Z:.3f}m")
            gripper_ctrl.open(duration_frames=HIGH_CUBE_RECOVERY_OPEN_FRAMES)
            arm_ctrl.set_target(arm_ctrl.current(), speed=SPEED_SCAN, min_frames=1)
        else:
            set_cube_static_anchor("restart search")
        detected_ball_pos = None
        detected_opening_hints = []
        dynamic_ik_plan = {}
        dynamic_grasp_xmat = None
        dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
        rgbd_scan_estimates = []
        rgbd_scan_hints = []
        scan_targets = []
        scan_idx = 0
        box_scan_targets = []
        box_scan_idx = 0
        box_search_hint_xy = None
        box_scan_hint_xy_used = None
        regrasp_count = 0
        pregrasp_replan_count = 0
        local_replan_count = 0
        place_settle_frames = 0
        place_correction_count = 0
        place_support_rescan_count = 0
        place_release_support_override = False
        prelift_hold_frames = 0
        prelift_ready_frames = 0
        lift_motion_frames = 0
        carry_motion_frames = 0
        carry_last_remaining_xy = float("inf")
        carry_stall_steps = 0
        transport_slip_abort_frames = 0
        transport_regrip_count = 0
        transport_regrip_frames = 0
        transport_regrip_stable_frames = 0
        transport_regrip_resume_sub = 2
        carried_cube_offset = None
        scan_round = 0
        if high_cube_recovery:
            recovery_wait_frames = 0
            recovery_stable_frames = 0
            recovery_reason = reason
            phase = 8
            sub = 10
            status_msg = (
                "Recovering high cube before rescan "
                f"({reason}) ..."
            )
        else:
            phase = 0
            sub = 0
            status_msg = f"Rescanning cube ({reason}) ..."

    def begin_local_rgbd_replan(reason: str) -> None:
        """Retreat to the current approach pose and reacquire RGB-D locally."""
        nonlocal phase, sub, local_replan_count, status_msg
        if local_replan_count >= MAX_LOCAL_REPLAN:
            restart_search(f"{reason}; local RGB-D retries exhausted")
            return
        if not plan_is_ready(dynamic_ik_plan):
            restart_search(f"{reason}; no retreat pose for local RGB-D retry")
            return
        local_replan_count += 1
        print(f"    Local RGB-D retry {local_replan_count}/{MAX_LOCAL_REPLAN}: "
              f"{reason}; retreating to approach pose for a clean depth frame.")
        arm_ctrl.set_target(
            dynamic_ik_plan["approach"].angles,
            speed=SPEED_LOCAL_REPLAN,
            min_frames=LOCAL_RETRACT_MIN_FRAMES,
        )
        phase = 2
        sub = 20
        status_msg = "Retreating for local RGB-D reacquire ..."

    def stable_global_rgbd_estimate() -> tuple[np.ndarray | None, float]:
        if not rgbd_scan_estimates:
            return None, float("inf")
        arr = np.vstack(rgbd_scan_estimates)
        med = np.median(arr, axis=0)
        errs = np.linalg.norm(arr - med, axis=1)
        if errs.size == 0:
            return None, float("inf")
        keep = errs <= max(GLOBAL_RGBD_STABILITY_M, float(np.median(errs)) + 0.010)
        if np.any(keep):
            arr = arr[keep]
        robust = np.median(arr, axis=0)
        spread = float(np.max(np.linalg.norm(arr - robust, axis=1))) if len(arr) else 0.0
        return robust, spread

    def global_rgbd_ready_for_planning() -> tuple[bool, np.ndarray | None, float]:
        stable, spread = stable_global_rgbd_estimate()
        if stable is None:
            return False, None, spread
        if len(rgbd_scan_estimates) >= GLOBAL_RGBD_MIN_VIEWS:
            return spread <= GLOBAL_RGBD_STABILITY_M, stable, spread
        return False, stable, spread

    def rgbd_single_view_is_strong() -> bool:
        return (
            detector.last_plane_inliers >= GLOBAL_RGBD_STRONG_SINGLE_VIEW_INLIERS and
            detector.last_plane_residual_m <= GLOBAL_RGBD_STRONG_SINGLE_VIEW_RESIDUAL_M and
            detector.last_estimate_spread_m <= GLOBAL_RGBD_STRONG_SINGLE_VIEW_SPREAD_M
        )

    def stable_box_rgbd_estimate() -> tuple[np.ndarray | None, float]:
        if not box_rgbd_scan_estimates:
            return None, float("inf")
        arr = np.vstack(box_rgbd_scan_estimates)
        med = np.median(arr, axis=0)
        errs = np.linalg.norm(arr - med, axis=1)
        if errs.size == 0:
            return None, float("inf")
        keep = errs <= max(BOX_ESTIMATE_STABILITY_M, float(np.median(errs)) + 0.020)
        if np.any(keep):
            arr = arr[keep]
        robust = np.median(arr, axis=0)
        spread = float(np.max(np.linalg.norm(arr - robust, axis=1))) if len(arr) else 0.0
        return robust, spread

    def box_single_view_is_strong() -> bool:
        extent = box_detector.last_extent_m
        return bool(
            box_detector.last_area_px >= BOX_STRONG_SINGLE_VIEW_AREA_PX and
            extent[0] >= BOX_STRONG_SINGLE_VIEW_MIN_XY_EXTENT_M and
            extent[1] >= BOX_STRONG_SINGLE_VIEW_MIN_XY_EXTENT_M and
            extent[2] >= BOX_STRONG_SINGLE_VIEW_MIN_Z_EXTENT_M
        )

    def try_plan_from_detected_targets(label: str) -> bool:
        nonlocal dynamic_ik_plan, phase, sub, status_msg
        nonlocal pregrasp_replan_count
        nonlocal scan_targets, scan_idx

        if detected_ball_pos is None:
            return False
        if REQUIRE_BOX_BEFORE_GRASP and detected_box_pos is None:
            status_msg = (f"Cube found by RGB-D; continuing autonomous scan "
                          f"for target box before grasp. "
                          f"cube=({detected_ball_pos[0]:.3f},"
                          f"{detected_ball_pos[1]:.3f},"
                          f"{detected_ball_pos[2]:.3f})")
            print("    Cube estimate is ready, but target box is not yet "
                  "reliably localized. Continuing scan before grasp so "
                  "placement is planned from a full box estimate.")
            dynamic_ik_plan = {}
            scan_targets = []
            scan_idx = 0
            sub = -10
            return True

        pregrasp_replan_count = 0
        planning_box = planning_box_for_cube_first(detected_ball_pos)
        planning_hints = (
            detected_opening_hints if detected_opening_hints
            else rgbd_scan_hints[-8:]
        )
        planning_hints = expand_face_opening_hints(planning_hints[:2])
        if planning_hints:
            print("    Planning grasp with RGB-D face-opening candidates: "
                  f"{[np.round(h[:2], 3).tolist() for h in planning_hints[:4]]}")
        dynamic_ik_plan = compute_dynamic_ik(
            detected_ball_pos,
            planning_box,
            opening_hints=planning_hints,
        )
        if plan_is_ready(dynamic_ik_plan):
            if detected_box_pos is None:
                status_msg = (f"Cube found by RGB-D; grasping first, "
                              f"box will be localized after lift. "
                              f"cube=({detected_ball_pos[0]:.3f},"
                              f"{detected_ball_pos[1]:.3f},"
                              f"{detected_ball_pos[2]:.3f})")
                print("    Cube plan ready; target box is still unknown, "
                      "so the robot will grasp first and run a dedicated "
                      "box scan after lift.")
            else:
                status_msg = (f"Cube and box found by RGB-D! "
                              f"cube=({detected_ball_pos[0]:.3f},"
                              f"{detected_ball_pos[1]:.3f},"
                              f"{detected_ball_pos[2]:.3f}) "
                              f"box=({detected_box_pos[0]:.3f},"
                              f"{detected_box_pos[1]:.3f},"
                              f"{detected_box_pos[2]:.3f})")
            arm_ctrl.set_target(arm_ctrl.current(), speed=SPEED_SCAN, min_frames=1)
            phase = 2
            sub = 0
            return True
        print(f"    RGB-D targets rejected ({label}): no complete IK plan.")
        diagnose_failure("RGB-D cube/box estimate without reachable plan")
        dynamic_ik_plan = {}
        return False

    def planning_box_for_cube_first(cube_xyz: np.ndarray) -> np.ndarray:
        """Return the real box center if known; otherwise a temporary grasp-only target."""
        if detected_box_pos is not None:
            return detected_box_pos.copy()
        cube_xyz = np.asarray(cube_xyz, dtype=np.float64)
        return cube_xyz + np.array([0.0, 0.0, 0.037], dtype=np.float64)

    def box_for_pregrasp_replan(cube_xyz: np.ndarray, label: str) -> np.ndarray:
        box = box_target_center()
        if box is not None:
            return box
        provisional = planning_box_for_cube_first(cube_xyz)
        print(f"    {label}: target box is still unknown; keeping cube-first "
              f"grasp plan with provisional placement target "
              f"{np.round(provisional, 4)}. Dedicated box scan will run after lift.")
        return provisional

    def accept_box_scan_estimate(est_xyz: np.ndarray, label: str) -> bool:
        nonlocal detected_box_pos, box_rgbd_scan_estimates, box_search_hint_xy

        if box_scan_hint_xy_used is not None:
            hint_err = float(np.linalg.norm(est_xyz[:2] - box_scan_hint_xy_used))
            if hint_err > BOX_HINT_MAX_DEVIATION_M:
                box_search_hint_xy = box_scan_hint_xy_used.copy()
                print(f"    RGB-D box estimate rejected ({label}): "
                      f"hint_err={hint_err*100:.1f}cm "
                      f"estimate={np.round(est_xyz[:2], 3)} "
                      f"hint={np.round(box_scan_hint_xy_used, 3)}")
                return False

        box_rgbd_scan_estimates.append(est_xyz.copy())
        if len(box_rgbd_scan_estimates) > BOX_ESTIMATE_MAX_VIEWS:
            box_rgbd_scan_estimates = box_rgbd_scan_estimates[-BOX_ESTIMATE_MAX_VIEWS:]

        stable_xyz, stable_spread = stable_box_rgbd_estimate()
        ready = (
            stable_xyz is not None and
            len(box_rgbd_scan_estimates) >= 2 and
            stable_spread <= BOX_ESTIMATE_STABILITY_M
        )
        if not ready and box_single_view_is_strong():
            ready = True
            stable_xyz = est_xyz.copy()
            stable_spread = 0.0
            print("    RGB-D single-view box extent is strong enough for placement.")

        print(f"    RGB-D box estimate set ({label}): "
              f"{len(box_rgbd_scan_estimates)}/2 "
              f"stable={ready} spread={stable_spread*100:.1f}cm "
              f"median={np.round(stable_xyz, 3) if stable_xyz is not None else None}")
        if ready and stable_xyz is not None:
            detected_box_pos = normalize_box_center_estimate(stable_xyz)
            if abs(float(detected_box_pos[2] - stable_xyz[2])) > 0.002:
                print(f"    Box Z normalized for tabletop placement: "
                      f"rgbd_z={stable_xyz[2]:.4f} -> "
                      f"used_z={detected_box_pos[2]:.4f}")
            if phase in (5, 6):
                return True
            return try_plan_from_detected_targets(label)
        return False

    def reprioritize_box_scan_from_hint(reason: str) -> bool:
        nonlocal box_scan_idx, box_scan_targets, box_scan_hint_xy_used

        if box_search_hint_xy is None:
            return False
        if (box_scan_hint_xy_used is not None and
                np.linalg.norm(box_scan_hint_xy_used - box_search_hint_xy) < 0.015):
            return False

        box_scan_targets = generate_box_scan_targets(
            box_search_hint_xy,
            z_plane=HELD_BOX_SCAN_Z_PLANE,
        )
        box_scan_idx = 0
        box_scan_hint_xy_used = box_search_hint_xy.copy()
        print(f"    Reprioritizing box scan around visual hint "
              f"{np.round(box_search_hint_xy, 3)} ({reason}).")
        return True

    def accept_partial_box_hint_for_place(reason: str) -> bool:
        nonlocal detected_box_pos, box_rgbd_scan_estimates
        nonlocal box_scan_targets, box_scan_idx, box_scan_hint_xy_used
        nonlocal sub, status_msg

        if not ALLOW_PARTIAL_BOX_HINT_PLACEMENT:
            if box_search_hint_xy is not None:
                print(f"    Partial RGB-D box hint is not trusted for final "
                      f"placement ({reason}): "
                      f"hint_xy={np.round(box_search_hint_xy, 4)}. "
                      "Continuing scan for a full box estimate.")
            return False
        if box_search_hint_xy is None:
            return False
        candidate = np.array([
            float(box_search_hint_xy[0]),
            float(box_search_hint_xy[1]),
            BOX_CENTER_Z_PRIOR,
        ], dtype=np.float64)
        detected_box_pos = normalize_box_center_estimate(candidate)
        box_rgbd_scan_estimates = [detected_box_pos.copy()]
        box_scan_targets = []
        box_scan_idx = 0
        box_scan_hint_xy_used = box_search_hint_xy.copy()
        print(f"    Using partial RGB-D box hint for guarded placement "
              f"({reason}): hint_xy={np.round(box_search_hint_xy, 4)} "
              f"box={np.round(detected_box_pos, 4)}")
        if update_place_plan_for_carried_offset(f"partial box hint: {reason}"):
            sub = 0
            status_msg = "Using partial RGB-D box hint for placement ..."
            return True
        detected_box_pos = None
        box_rgbd_scan_estimates = []
        return False

    def held_box_scan_target_is_safe(target: ScanTarget) -> bool:
        offset = (
            carried_cube_offset.copy() if carried_cube_offset is not None
            else current_ball_xyz() - gripper_object_center()
        )
        predicted_cube = target.gripper + offset
        if predicted_cube[2] < HELD_BOX_SCAN_MIN_CUBE_Z:
            print(f"    Skipping low held-box scan target: "
                  f"aim={np.round(target.aim, 3)} "
                  f"pred_cube_z={predicted_cube[2]:.3f}m")
            return False
        return True

    def accept_global_scan_estimate(est_xyz: np.ndarray, label: str) -> bool:
        nonlocal detected_ball_pos, detected_opening_hints, dynamic_ik_plan
        nonlocal pregrasp_replan_count, phase, sub, status_msg
        nonlocal rgbd_scan_estimates, rgbd_scan_hints

        if cube_static_anchor_active:
            rgbd_anchor_err = float(np.linalg.norm(est_xyz - cube_static_pos))
            print(f"    RGB-D vs stationary-cube debug: "
                  f"rgbd={np.round(est_xyz, 3)} "
                  f"anchor={np.round(cube_static_pos, 3)} "
                  f"err={rgbd_anchor_err*100:.1f}cm")

        rgbd_scan_estimates.append(est_xyz.copy())
        if detected_opening_hints:
            rgbd_scan_hints.extend(detected_opening_hints)
        if len(rgbd_scan_estimates) > GLOBAL_RGBD_MAX_VIEWS:
            rgbd_scan_estimates = rgbd_scan_estimates[-GLOBAL_RGBD_MAX_VIEWS:]

        ready, stable_xyz, stable_spread = global_rgbd_ready_for_planning()
        if not ready and rgbd_single_view_is_strong():
            ready = True
            stable_xyz = est_xyz.copy()
            stable_spread = detector.last_estimate_spread_m
            print("    RGB-D single-view plane is strong enough for planning.")

        print(f"    RGB-D global estimate set ({label}): "
              f"{len(rgbd_scan_estimates)}/{GLOBAL_RGBD_MIN_VIEWS} "
              f"stable={ready} spread={stable_spread*100:.1f}cm "
              f"median={np.round(stable_xyz, 3) if stable_xyz is not None else None}")
        if ready and stable_xyz is not None:
            detected_ball_pos = stable_xyz.copy()
            detected_opening_hints = rgbd_scan_hints[-8:]
            if try_plan_from_detected_targets(label):
                return True
        elif len(rgbd_scan_estimates) >= GLOBAL_RGBD_MAX_VIEWS:
            print("    RGB-D estimates are not stable enough yet; "
                  "discarding oldest views and continuing scan.")
        return False

    def demo_tick() -> None:
        nonlocal phase, sub, finished, scan_idx, scan_targets, regrasp_count
        nonlocal detected_ball_pos, detected_box_pos, detected_opening_hints
        nonlocal dynamic_ik_plan, status_msg, ball_z_before_lift
        nonlocal scan_round, box_verify_left, pregrasp_replan_count
        nonlocal grip_contact_hold, grip_close_frames, gripper_open_wait_frames, local_replan_count
        nonlocal preclose_stable_frames, place_settle_frames, prelift_lost_frames
        nonlocal prelift_hold_frames, prelift_ready_frames, lift_motion_frames
        nonlocal carry_motion_frames, carry_last_remaining_xy, carry_stall_steps
        nonlocal transport_slip_abort_frames
        nonlocal transport_regrip_count, transport_regrip_frames
        nonlocal transport_regrip_stable_frames, transport_regrip_resume_sub
        nonlocal carried_cube_offset, place_correction_count
        nonlocal place_support_rescan_count, place_release_support_override
        nonlocal dynamic_grasp_xmat, dynamic_grasp_z_offset, rejected_grasp_frames
        nonlocal rgbd_scan_estimates, rgbd_scan_hints, box_rgbd_scan_estimates
        nonlocal last_scan_rgbd_sample_frame
        nonlocal box_scan_idx, box_scan_targets, box_search_hint_xy
        nonlocal box_scan_hint_xy_used
        nonlocal recovery_wait_frames, recovery_stable_frames, recovery_reason

        if finished:
            return

        # 鈹€鈹€ Phase -1:  idle 鈥?wait for SPACE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if phase == -1:
            status_msg = "IDLE 鈥?use cube_x/y/z & start_demo sliders"
            return  # do nothing, wait for keyboard

        # 鈹€鈹€ Phase 0:  scanning 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 0:
            if sub == -10:
                if detected_ball_pos is None:
                    sub = 0
                    return
                scan_targets = generate_box_scan_targets(
                    box_search_hint_xy,
                    z_plane=HELD_BOX_SCAN_Z_PLANE,
                )
                box_scan_hint_xy_used = (
                    box_search_hint_xy.copy()
                    if box_search_hint_xy is not None else None
                )
                scan_idx = 0
                last_scan_rgbd_sample_frame = -SCAN_RGBD_CAPTURE_EVERY_N
                print(f">>> Phase 0b : Target box scan before grasp "
                      f"({len(scan_targets)} candidate views, "
                      f"hint={np.round(box_search_hint_xy, 3) if box_search_hint_xy is not None else None})")
                while scan_idx < len(scan_targets):
                    scan_target = scan_targets[scan_idx]
                    result = solve_gripper_center_ik(
                        model, data, scan_target.gripper,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=SCAN_POS_TOL,
                        target_xmat=scan_target.xmat, orientation_body_id=tool_id,
                        orientation_weight=SCAN_ORI_WEIGHT, orientation_tol=SCAN_ORI_TOL,
                    )
                    if result.success:
                        print(f"    Moving to pre-grasp box scan "
                              f"{scan_idx+1}/{len(scan_targets)} "
                              f"aim={np.round(scan_target.aim, 3)}")
                        arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                        status_msg = (
                            f"Box scan before grasp {scan_idx+1}/"
                            f"{len(scan_targets)} ..."
                        )
                        sub = -9
                        return
                    scan_idx += 1
                print("    No reachable pre-grasp box scan pose; returning to "
                      "global scan.")
                sub = 0
                return

            if sub == -9:
                if not arm_ctrl.done:
                    if (arm_ctrl.near_done(SCAN_MOVING_CAPTURE_PROGRESS) and
                            frame - last_scan_rgbd_sample_frame >= SCAN_RGBD_CAPTURE_EVERY_N):
                        last_scan_rgbd_sample_frame = frame
                        box_xyz = capture_box_from_rgbd(
                            f"moving pre-grasp box scan {scan_idx+1}/"
                            f"{len(scan_targets)}",
                            log_miss=False,
                        )
                        if box_xyz is not None:
                            if accept_box_scan_estimate(
                                box_xyz,
                                f"moving pre-grasp box scan {scan_idx+1}/"
                                f"{len(scan_targets)}",
                            ):
                                return
                            if box_search_hint_xy is not None:
                                scan_targets = generate_box_scan_targets(
                                    box_search_hint_xy,
                                    z_plane=HELD_BOX_SCAN_Z_PLANE,
                                )
                                scan_idx = 0
                    return

                last_scan_rgbd_sample_frame = frame
                box_xyz = capture_box_from_rgbd(
                    f"pre-grasp box scan {scan_idx+1}/{len(scan_targets)}",
                    log_miss=(detected_box_pos is None),
                )
                if box_xyz is not None:
                    if accept_box_scan_estimate(
                        box_xyz,
                        f"pre-grasp box scan {scan_idx+1}/{len(scan_targets)}",
                    ):
                        return
                    if box_search_hint_xy is not None:
                        scan_targets = generate_box_scan_targets(
                            box_search_hint_xy,
                            z_plane=HELD_BOX_SCAN_Z_PLANE,
                        )
                        scan_idx = 0

                scan_idx += 1
                while scan_idx < len(scan_targets):
                    scan_target = scan_targets[scan_idx]
                    result = solve_gripper_center_ik(
                        model, data, scan_target.gripper,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=SCAN_POS_TOL,
                        target_xmat=scan_target.xmat, orientation_body_id=tool_id,
                        orientation_weight=SCAN_ORI_WEIGHT, orientation_tol=SCAN_ORI_TOL,
                    )
                    if result.success:
                        print(f"    Moving to pre-grasp box scan "
                              f"{scan_idx+1}/{len(scan_targets)} "
                              f"aim={np.round(scan_target.aim, 3)}")
                        arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                        status_msg = (
                            f"Box scan before grasp {scan_idx+1}/"
                            f"{len(scan_targets)} ..."
                        )
                        return
                    scan_idx += 1
                print("    Pre-grasp box scan exhausted without a full box "
                      "estimate; returning to global scan.")
                sub = 0
                return

            if sub == 0:
                if rgbd_window is None:
                    print("    ERROR: RGB-D camera is not available; refusing to use a fixed cube target.")
                    status_msg = "ERROR: RGB-D camera unavailable"
                    finished = True
                    return
                print(">>> Phase 0 : Camera scanning for cube and target box ...")
                rgbd_scan_estimates = []
                rgbd_scan_hints = []
                if detected_box_pos is None:
                    box_rgbd_scan_estimates = []
                last_scan_rgbd_sample_frame = -SCAN_RGBD_CAPTURE_EVERY_N
                est_xyz = capture_cube_from_rgbd("initial stationary view")
                if est_xyz is not None:
                    if accept_global_scan_estimate(
                        est_xyz, "initial stationary view"
                    ):
                        return
                box_xyz = capture_box_from_rgbd(
                    "initial stationary view",
                    log_miss=(detected_box_pos is None),
                )
                if box_xyz is not None:
                    if accept_box_scan_estimate(
                        box_xyz, "initial stationary view"
                    ):
                        return
                scan_targets = generate_scan_targets(
                    _last_bz, hint_xy=np.array([_last_bx, _last_by], dtype=np.float64)
                )
                print(f"    Generated {len(scan_targets)} global scan poses "
                      f"(360-degree polar rings + angled edge views)")
                scan_idx = 0
                while scan_idx < len(scan_targets):
                    scan_target = scan_targets[scan_idx]
                    result = solve_gripper_center_ik(
                        model, data, scan_target.gripper,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=SCAN_POS_TOL,
                        target_xmat=scan_target.xmat, orientation_body_id=tool_id,
                        orientation_weight=SCAN_ORI_WEIGHT, orientation_tol=SCAN_ORI_TOL,
                    )
                    if result.success:
                        print(f"    Moving to scan {scan_idx+1}/{len(scan_targets)} "
                              f"aim={np.round(scan_target.aim, 3)}")
                        arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                        status_msg = f"Scanning {scan_idx+1}/{len(scan_targets)} ..."
                        break
                    scan_idx += 1
                if scan_idx >= len(scan_targets):
                    print("    No reachable camera scan pose in this round.")
                    scan_round += 1
                    if scan_round >= 3:
                        print("    Giving up after 3 rounds.")
                        status_msg = "ERROR: no reachable scan pose"
                        finished = True
                        return
                    # Regenerate scan targets and retry
                    scan_targets = generate_scan_targets(
                        CUBE_REST_Z,
                        hint_xy=np.array([_last_bx, _last_by], dtype=np.float64),
                    )
                    scan_idx = 0
                    sub = 0
                    status_msg = f"Regenerating scan poses round {scan_round+1} ..."
                    return
                sub = 1

            elif sub == 1:
                if not arm_ctrl.done:
                    if (arm_ctrl.near_done(SCAN_MOVING_CAPTURE_PROGRESS) and
                            frame - last_scan_rgbd_sample_frame >= SCAN_RGBD_CAPTURE_EVERY_N):
                        last_scan_rgbd_sample_frame = frame
                        est_xyz = capture_cube_from_rgbd(
                            f"moving scan {scan_idx+1}/{len(scan_targets)}",
                            draw=False,
                            log_miss=False,
                        )
                        if est_xyz is not None:
                            if accept_global_scan_estimate(
                                est_xyz,
                                f"moving scan {scan_idx+1}/{len(scan_targets)}",
                            ):
                                return
                        box_xyz = capture_box_from_rgbd(
                            f"moving scan {scan_idx+1}/{len(scan_targets)}",
                            log_miss=False,
                        )
                        if box_xyz is not None:
                            if accept_box_scan_estimate(
                                box_xyz,
                                f"moving scan {scan_idx+1}/{len(scan_targets)}",
                            ):
                                return
                    return

                # Capture and detect using the wrist RGB-D camera. No RGB-D
                # estimate means no grasp target is accepted.
                last_scan_rgbd_sample_frame = frame
                est_xyz = capture_cube_from_rgbd(
                    f"scan {scan_idx+1}/{len(scan_targets)}"
                )
                if est_xyz is not None:
                    if accept_global_scan_estimate(
                        est_xyz,
                        f"settled scan {scan_idx+1}/{len(scan_targets)}",
                    ):
                        return
                box_xyz = capture_box_from_rgbd(
                    f"settled scan {scan_idx+1}/{len(scan_targets)}",
                    log_miss=(detected_box_pos is None),
                )
                if box_xyz is not None:
                    if accept_box_scan_estimate(
                        box_xyz,
                        f"settled scan {scan_idx+1}/{len(scan_targets)}",
                    ):
                        return

                # 鈹€鈹€ Next scan pose 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
                scan_idx += 1
                while scan_idx < len(scan_targets):
                    target = scan_targets[scan_idx]
                    result = solve_gripper_center_ik(
                        model, data, target.gripper,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=SCAN_POS_TOL,
                        target_xmat=target.xmat, orientation_body_id=tool_id,
                        orientation_weight=SCAN_ORI_WEIGHT, orientation_tol=SCAN_ORI_TOL,
                    )
                    if result.success:
                        print(f"    Moving to scan {scan_idx+1}/{len(scan_targets)} "
                              f"aim={np.round(target.aim, 3)}")
                        arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                        status_msg = f"Scanning {scan_idx+1}/{len(scan_targets)} ..."
                        return
                    scan_idx += 1

                # All scan poses exhausted — restart scan loop
                scan_round += 1
                if scan_round >= 3:
                    if detected_ball_pos is not None and detected_box_pos is None:
                        print("    Cube is localized, but target box was not "
                              "found after 3 full scan rounds — stopping.")
                        status_msg = "ERROR: target box not found by RGB-D"
                    else:
                        print("    Cube not found after 3 full scan rounds — stopping.")
                        status_msg = "ERROR: cube not found — check lighting / camera"
                    finished = True
                    return
                print(f"    All {len(scan_targets)} poses scanned — "
                      f"restarting scan round {scan_round+1} ...")
                status_msg = (f"Rescanning round {scan_round+1} "
                              f"— cube not found yet")
                scan_idx = 0
                sub = 0
                return

        # 鈹€鈹€ Phase 1: reserved; keep current posture and continue 鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 1:
            print(">>> Phase 1 : Skip home; planning from current posture")
            phase = 2; sub = 0

        # 鈹€鈹€ Phase 2:  approach 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 2:
            if sub == 0:
                if not plan_is_ready(dynamic_ik_plan):
                    diagnose_failure("approach requested with incomplete plan")
                    restart_search("incomplete IK plan before approach")
                    return
                print(">>> Phase 2 : Open gripper before approach")
                gripper_ctrl.open(duration_frames=55)
                gripper_open_wait_frames = 0
                sub = 10
                status_msg = "Opening gripper before approach ..."
                return
            if sub == 10:
                gripper_open_wait_frames += 1
                if not gripper_ctrl.done:
                    return
                if not gripper_is_open():
                    open_err = gripper_open_error()
                    if gripper_open_wait_frames < GRIPPER_OPEN_WAIT_FRAMES:
                        return
                    print(f"    Gripper open wait timeout; continuing with "
                          f"open_err={open_err:.4f}rad "
                          f"(tol={GRIPPER_OPEN_TOL:.4f}rad)")
                print(">>> Phase 2 : Approach")
                arm_ctrl.set_target(dynamic_ik_plan["approach"].angles, speed=SPEED_NORMAL)
                sub = 1
                status_msg = "Approaching cube ..."
                return
            if sub == 20:
                if not arm_ctrl.near_done(APPROACH_CAPTURE_PROGRESS):
                    return
                fresh = capture_cube_from_rgbd("local retreat re-acquire")
                if fresh is None:
                    if (cube_static_anchor_active and
                            detected_ball_pos is not None and
                            plan_is_ready(dynamic_ik_plan)):
                        anchor_delta = float(np.linalg.norm(
                            detected_ball_pos - cube_static_pos))
                        if anchor_delta <= max(PREGRASP_ANCHORED_CUBE_MAX_ERR_M,
                                               PRE_CLOSE_MAX_CENTER_ERR):
                            print("    Local RGB-D reacquire missed/rejected an "
                                  "anchored cube; using stable global scan plan.")
                            pregrasp_replan_count = 0
                            phase = 3
                            sub = 0
                            return
                    diagnose_failure("local RGB-D reacquire failed")
                    begin_local_rgbd_replan(
                        "local RGB-D reacquire failed after retreat")
                    return
                detected_ball_pos = fresh.copy()
                box = box_for_pregrasp_replan(
                    detected_ball_pos, "Local re-acquire")
                previous_xmat = (
                    None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                )
                replanned = compute_dynamic_ik(
                    detected_ball_pos,
                    box,
                    opening_hints=detected_opening_hints,
                    preview_top_k=LOCAL_PHYSICAL_EVAL_TOP_K,
                )
                if not plan_is_ready(replanned):
                    dynamic_grasp_xmat = previous_xmat
                    diagnose_failure("local RGB-D replan failed")
                    begin_local_rgbd_replan(
                        "local RGB-D replan produced no grasp")
                    return
                dynamic_ik_plan = replanned
                print("    Local RGB-D replan ready; descending again.")
                pregrasp_replan_count = 0
                phase = 3
                sub = 0
                status_msg = "Descending from local RGB-D replan ..."
                return
            if sub == 1:
                if not arm_ctrl.near_done(APPROACH_CAPTURE_PROGRESS):
                    return
                fresh = capture_cube_from_rgbd("pre-grasp confirm")
                if fresh is None:
                    if detected_ball_pos is not None and plan_is_ready(dynamic_ik_plan):
                        if cube_static_anchor_active:
                            anchor_delta = float(np.linalg.norm(
                                detected_ball_pos - cube_static_pos))
                            if anchor_delta <= max(PRE_CLOSE_MAX_CENTER_ERR, 0.030):
                                print("    Pre-grasp RGB-D missed an anchored cube "
                                      f"(anchor_err={anchor_delta*100:.1f}cm); "
                                      "using stable scan estimate and "
                                      "physical-preview plan.")
                                pregrasp_replan_count = 0
                                phase = 3
                                sub = 0
                                return
                        expected_visible, reason = rgbd_frustum_status(detected_ball_pos)
                        if not expected_visible:
                            print("    Pre-grasp RGB-D expected miss from side camera "
                                  f"({reason}); using stable scan estimate and "
                                  "physical-preview plan.")
                            pregrasp_replan_count = 0
                            phase = 3
                            sub = 0
                            return
                    diagnose_failure("pre-grasp RGB-D lost cube")
                    begin_local_rgbd_replan(
                        "cube moved out of RGB-D view before grasp")
                    return
                previous = detected_ball_pos.copy()
                delta = float(np.linalg.norm(fresh - previous))
                if cube_static_anchor_active:
                    anchor_delta = float(np.linalg.norm(fresh - cube_static_pos))
                    print(f"    RGB-D pre-grasp debug: delta={delta*100:.1f}cm "
                          f"anchor_err={anchor_delta*100:.1f}cm")
                    if (anchor_delta > PREGRASP_ANCHORED_CUBE_MAX_ERR_M and
                            plan_is_ready(dynamic_ik_plan)):
                        print("    Pre-grasp RGB-D estimate conflicts with the "
                              "stationary cube anchor; keeping the stable global "
                              "scan plan instead of replanning from an occluded "
                              "close-range frame.")
                        pregrasp_replan_count = 0
                        phase = 3
                        sub = 0
                        return

                if delta <= LOCAL_RGBD_KEEP_PLAN_DELTA and plan_is_ready(dynamic_ik_plan):
                    print(f"    Pre-grasp RGB-D stable (delta={delta*100:.1f}cm); "
                          "using the existing physical-preview plan.")
                    pregrasp_replan_count = 0
                    phase = 3
                    sub = 0
                    return

                detected_ball_pos = fresh.copy()
                box = box_for_pregrasp_replan(
                    detected_ball_pos, "Pre-grasp RGB-D")
                previous_xmat = (
                    None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                )
                replanned = compute_dynamic_ik(
                    detected_ball_pos,
                    box,
                    opening_hints=detected_opening_hints,
                    preview_top_k=LOCAL_PHYSICAL_EVAL_TOP_K,
                )
                if not plan_is_ready(replanned):
                    dynamic_grasp_xmat = previous_xmat
                    diagnose_failure("pre-grasp RGB-D replan failed")
                    begin_local_rgbd_replan(
                        "no reachable grasp from RGB-D pre-grasp estimate")
                    return
                dynamic_ik_plan = replanned

                approach_err = float(np.linalg.norm(
                    gripper_contact_center() -
                    dynamic_ik_plan["approach"].target
                ))
                if delta > VISION_REPLAN_DELTA:
                    if pregrasp_replan_count >= MAX_PREGRASP_REPLANS:
                        print(f"    Vision estimate still shifted {delta*100:.1f}cm "
                              f"after {pregrasp_replan_count} replan(s); "
                              "accepting latest RGB-D estimate to continue grasp.")
                        pregrasp_replan_count = 0
                        phase = 3; sub = 0
                        return
                    pregrasp_replan_count += 1
                    print(f"    RGB-D estimate shifted {delta*100:.1f}cm since scan; "
                          f"re-approaching local RGB-D estimate "
                          f"({pregrasp_replan_count}/{MAX_PREGRASP_REPLANS}).")
                    arm_ctrl.set_target(
                        dynamic_ik_plan["approach"].angles,
                        speed=SPEED_LOCAL_REPLAN,
                        min_frames=LOCAL_REAPPROACH_MIN_FRAMES,
                    )
                    sub = 1
                    status_msg = "Re-approaching local RGB-D estimate ..."
                    return
                if (pregrasp_replan_count < MAX_PREGRASP_REPLANS and
                        approach_err > PREGRASP_REAPPROACH_TOL):
                    pregrasp_replan_count += 1
                    print(f"    Re-approaching RGB-D cube estimate "
                          f"({pregrasp_replan_count}/{MAX_PREGRASP_REPLANS}); "
                          f"approach_err={approach_err*100:.1f}cm")
                    arm_ctrl.set_target(
                        dynamic_ik_plan["approach"].angles,
                        speed=SPEED_LOCAL_REPLAN,
                        min_frames=LOCAL_REAPPROACH_MIN_FRAMES,
                    )
                    sub = 1
                    status_msg = "Re-approaching RGB-D cube ..."
                    return
                pregrasp_replan_count = 0
                phase = 3; sub = 0

        # 鈹€鈹€ Phase 3:  descend 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 3:
            if sub == 9:
                if not gripper_ctrl.done or not gripper_is_open():
                    return
                sub = 0
                return
            if sub == 0:
                if detected_ball_pos is None:
                    print("    No cube position — restarting search")
                    restart_search("lost cube position before descend")
                    return
                if not gripper_is_open():
                    print("    Gripper is not fully open; opening before descent.")
                    gripper_ctrl.open(duration_frames=35)
                    sub = 9
                    status_msg = "Opening gripper before descent ..."
                    return
                # Align the real jaw center with the cube center.  Retries
                # sample slightly different heights instead of driving below
                # the cube, which caused table/box scraping and side misses.
                z_offs = (
                    dynamic_grasp_z_offset if regrasp_count == 0 else
                    GRASP_Z_OFFSETS[min(regrasp_count, len(GRASP_Z_OFFSETS) - 1)]
                )
                print(f">>> Phase 3 : Descend  (grasp Z offset = {z_offs*1000:.0f} mm)")
                # The scan/pre-grasp step already ran the expensive physical
                # preview.  Reuse that plan for the first descent so the robot
                # does not freeze right before grasping.
                replan_needed = (
                    regrasp_count > 0 or not plan_is_ready(dynamic_ik_plan)
                )
                if replan_needed:
                    box = box_target_center()
                    if box is None:
                        print("    Cannot descend: target box has not been located yet.")
                        phase = 0
                        sub = 0
                        return
                    previous_xmat = (
                        None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                    )
                    replanned = compute_dynamic_ik(
                        detected_ball_pos,
                        box,
                        grasp_z_offs=z_offs,
                        opening_hints=detected_opening_hints,
                        preview_top_k=LOCAL_PHYSICAL_EVAL_TOP_K,
                    )
                    if plan_is_ready(replanned):
                        dynamic_ik_plan = replanned
                    else:
                        dynamic_grasp_xmat = previous_xmat
                        diagnose_failure("descend RGB-D replan failed")
                        begin_local_rgbd_replan(
                            "descend RGB-D replan produced no grasp")
                        return
                if not plan_is_ready(dynamic_ik_plan):
                    diagnose_failure("descend requested with incomplete plan")
                    begin_local_rgbd_replan("incomplete IK plan before descend")
                    return
                print("    Descending via midpoint to reduce lateral finger sweep.")
                arm_ctrl.set_target(
                    dynamic_ik_plan["descend_mid"].angles,
                    speed=SPEED_SLOW,
                    min_frames=DESCEND_MID_MIN_FRAMES,
                )
                preclose_stable_frames = 0
                sub = 5
                status_msg = f"Descending to pre-grasp midpoint (retry {regrasp_count}) ..."
            if sub == 5:
                if not arm_ctrl.done:
                    if cube_touching_robot():
                        min_dist = cube_robot_min_contact_dist()
                        bad_contact, bad_name = cube_nonfinger_robot_contact()
                        print(f"    Early contact before descend midpoint "
                              f"(min_dist={min_dist*1000:.2f} mm); "
                              "rejecting this approach path.")
                        if bad_contact:
                            print(f"    Midpoint contact body: {bad_name}")
                        remember_failed_grasp_frame("early contact before descend midpoint")
                        set_cube_static_anchor("early contact before descend midpoint")
                        begin_local_rgbd_replan("early contact before descend midpoint")
                    return
                if cube_touching_robot():
                    min_dist = cube_robot_min_contact_dist()
                    bad_contact, bad_name = cube_nonfinger_robot_contact()
                    print(f"    Contact at descend midpoint "
                          f"(min_dist={min_dist*1000:.2f} mm); "
                          "rejecting this approach path before final descent.")
                    if bad_contact:
                        print(f"    Midpoint contact body: {bad_name}")
                    remember_failed_grasp_frame("contact at descend midpoint")
                    set_cube_static_anchor("contact at descend midpoint")
                    begin_local_rgbd_replan("contact at descend midpoint")
                    return
                arm_ctrl.set_target(dynamic_ik_plan["grasp"].angles, speed=SPEED_SLOW)
                sub = 1
                status_msg = f"Final short descent (retry {regrasp_count}) ..."
            if sub == 1 and not arm_ctrl.done and cube_touching_robot():
                min_dist = cube_robot_min_contact_dist()
                bad_contact, bad_name = cube_nonfinger_robot_contact()
                if bad_contact or min_dist < -EARLY_CONTACT_MAX_PENETRATION:
                    print(f"    Early robot-cube contact during descent "
                          f"(min_dist={min_dist*1000:.2f} mm); "
                          "aborting this grasp posture.")
                    if bad_contact:
                        print(f"    Early contact body: {bad_name}")
                    remember_failed_grasp_frame("early contact during descent")
                    set_cube_static_anchor("early contact during descent")
                    begin_local_rgbd_replan("early cube contact during descent")
                    return
            if arm_ctrl.near_done(DESCEND_CLOSE_PROGRESS):
                if not arm_ctrl.done:
                    preclose_stable_frames = 0
                    q_err, max_qvel = arm_ctrl.settle_metrics()
                    status_msg = (
                        "Finishing descent before soft close "
                        f"(qerr={q_err:.3f}, qvel={max_qvel:.3f}) ..."
                    )
                    return
                preclose_touch = cube_touching_robot()
                bad_contact, bad_name = cube_nonfinger_robot_contact()
                min_dist = cube_robot_min_contact_dist()
                if preclose_touch and (
                        bad_contact or min_dist < -EARLY_CONTACT_MAX_PENETRATION):
                    print(f"    Bad pre-close robot-cube contact "
                          f"(min_dist={min_dist*1000:.2f} mm); "
                          "rejecting this grasp posture.")
                    if bad_contact:
                        print(f"    Pre-close contact body: {bad_name}")
                    remember_failed_grasp_frame("bad pre-close contact")
                    diagnose_failure("bad pre-close contact before close")
                    begin_local_rgbd_replan("bad pre-close contact before close")
                    return
                if not pre_close_alignment_ok("Pre-close alignment"):
                    if not arm_ctrl.done:
                        status_msg = "Finishing descent before final alignment check ..."
                        return
                    remember_failed_grasp_frame("pre-close alignment mismatch")
                    print("    Refusing to close at wrong location; "
                          "using local RGB-D retry instead of global rescan.")
                    diagnose_failure("pre-close alignment mismatch")
                    begin_local_rgbd_replan("pre-close alignment mismatch")
                    return
                if preclose_touch:
                    left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
                    if not (left_pad_hit and right_pad_hit):
                        print(f"    One-sided pre-close fingertip contact: "
                              f"left={left_pad_hit} right={right_pad_hit} "
                              f"finger_contacts={pad_contacts}; rejecting this "
                              "grasp posture before closing.")
                        remember_failed_grasp_frame("one-sided pre-close contact")
                        diagnose_failure("one-sided pre-close contact")
                        begin_local_rgbd_replan("one-sided pre-close contact")
                        return
                    print(f"    Pre-close fingertip contact accepted "
                          f"(min_dist={min_dist*1000:.2f} mm); "
                          "releasing anchor and closing now.")
                else:
                    preclose_stable_frames += 1
                    if preclose_stable_frames < PRE_CLOSE_STABLE_FRAMES:
                        if preclose_stable_frames == 1:
                            q_err, max_qvel = arm_ctrl.settle_metrics()
                            print(f"    Pre-close alignment OK; settling before release "
                                  f"({preclose_stable_frames}/{PRE_CLOSE_STABLE_FRAMES}, "
                                  f"qerr={q_err:.4f}, qvel={max_qvel:.4f})")
                        status_msg = (
                            "Settling at grasp pose before soft close "
                            f"({preclose_stable_frames}/{PRE_CLOSE_STABLE_FRAMES}) ..."
                        )
                        return
                local_replan_count = 0
                ball_z_before_lift = float(current_ball_xyz()[2])
                release_cube_for_soft_close("pre-close alignment passed")
                preclose_stable_frames = 0
                phase = 4; sub = 0

        # 鈹€鈹€ Phase 4:  close gripper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 4:
            if sub == 0:
                release_cube_for_soft_close("start gripper close")
                gc = gripper_object_center()
                ball = current_ball_xyz()
                err = np.linalg.norm(ball - gc)
                print(f"    Grasp center->cube: {err*100:.1f}cm  "
                      f"(grasp Z={gc[2]:.3f}, cube Z={ball[2]:.3f})")
                grip_contact_hold = 0
                grip_close_frames = 0
                prelift_lost_frames = 0
                prelift_hold_frames = 0
                prelift_ready_frames = 0
                reset_visual_grip_feedback("new grasp close")
                gripper_ctrl.close(duration_frames=GRIP_CLOSE_FRAMES)
                sub = 1
                status_msg = "Closing gripper ..."
            if sub == 1:
                grip_close_frames += 1
                contacts = finger_contact_count()
                left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
                patch_span = finger_contact_patch_span()
                pair_skew, contact_center_err = finger_contact_alignment()
                min_dist = cube_robot_min_contact_dist()
                bad_contact, bad_name = cube_nonfinger_robot_contact()
                skin = read_skin()
                geom = grip_geometry_metrics()
                if bad_contact:
                    print(f"    Non-finger robot contact with cube: {bad_name}; "
                          "aborting this grasp.")
                    remember_failed_grasp_frame("non-finger contact")
                    diagnose_failure("cube touched by non-finger robot part")
                    regrasp_count += 1
                    if regrasp_count >= MAX_REGRASP:
                        restart_search("non-finger robot contact")
                    else:
                        phase = 8; sub = 0
                    return
                if (left_pad_hit and right_pad_hit and
                        contact_patch_is_large(pad_contacts, patch_span) and
                        pair_skew <= MAX_CONTACT_PAIR_SKEW and
                        contact_center_err <= MAX_CONTACT_CENTER_ERR and
                        min_dist <= CONTACT_CONFIRM_MAX_DIST and
                        skin_squeeze_ok(skin) and bool(geom["ok"])):
                    grip_contact_hold += 1
                else:
                    grip_contact_hold = max(0, grip_contact_hold - 1)

                if grip_contact_hold == 1:
                    print(f"    Two-sided finger contact detected while closing: "
                          f"finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"geom_center={geom['center_err']*100:.1f}cm")
                elif grip_close_frames % 45 == 0:
                    print(f"    Closing progress: left={left_pad_hit} right={right_pad_hit} "
                          f"finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"hold={grip_contact_hold}/{GRIP_CONTACT_HOLD_FRAMES} "
                          f"geom_ok={bool(geom['ok'])} "
                          f"center={geom['center_err']*100:.1f}cm "
                          f"open_axis={geom['open_axis_err']*100:.1f}cm "
                          f"face_axis={geom['face_axis_err']*100:.1f}cm")

                if grip_contact_hold >= GRIP_CONTACT_HOLD_FRAMES:
                    print(f"    Real two-sided finger contact confirmed: "
                          f"body_contacts={contacts} finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"hold={grip_contact_hold}/{GRIP_CONTACT_HOLD_FRAMES}")
                    if activate_grip_lock("after close"):
                        gripper_ctrl.hold(duration_frames=GRIP_HOLD_FRAMES)
                        prelift_hold_frames = 0
                        prelift_ready_frames = 0
                        sub = 2
                        status_msg = "Stabilizing grasp before lift ..."
                    else:
                        remember_failed_grasp_frame("physical grasp refused")
                        diagnose_failure("contact found but physical grasp refused")
                        regrasp_count += 1
                        if regrasp_count >= MAX_REGRASP:
                            restart_search("physical grasp confirmation failed")
                        else:
                            phase = 8; sub = 0
                    return

                if gripper_ctrl.done or grip_close_frames >= GRIP_CLOSE_TIMEOUT_FRAMES:
                    print(f"    Need two-sided real finger contact: left={left_pad_hit} "
                          f"right={right_pad_hit} finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"hold={grip_contact_hold}/{GRIP_CONTACT_HOLD_FRAMES} "
                          f"geom_ok={bool(geom['ok'])} "
                          f"center={geom['center_err']*100:.1f}cm "
                          f"open_axis={geom['open_axis_err']*100:.1f}cm "
                          f"face_axis={geom['face_axis_err']*100:.1f}cm")
                    diagnose_failure("gripper closed without real cube contact")
                    remember_failed_grasp_frame("closed without contact")
                    regrasp_count += 1
                    if regrasp_count >= MAX_REGRASP:
                        restart_search("no real gripper-cube contact")
                    else:
                        phase = 8; sub = 0
            elif sub == 2:
                contacts = finger_contact_count()
                left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
                patch_span = finger_contact_patch_span()
                pair_skew, contact_center_err = finger_contact_alignment()
                min_dist = cube_robot_min_contact_dist()
                skin = read_skin()
                geom = grip_geometry_metrics()
                maintain_fused_grip_force("pre-lift hold", skin)
                prelift_hold_frames += 1
                strong_tactile_pair = (
                    pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
                    skin_squeeze_ok(skin, SKIN_PRELIFT_READY_FORCE)
                )
                prelift_contact_ok = (
                    left_pad_hit and right_pad_hit and
                    (contact_patch_is_large(pad_contacts, patch_span) or
                     strong_tactile_pair) and
                    pair_skew <= MAX_CONTACT_PAIR_SKEW and
                    contact_center_err <= MAX_CONTACT_CENTER_ERR and
                    min_dist <= CONTACT_CONFIRM_MAX_DIST and
                    skin_squeeze_ok(skin, SKIN_TOUCH_FORCE) and
                    bool(geom["ok"])
                )
                if not prelift_contact_ok:
                    prelift_lost_frames += 1
                    if prelift_lost_frames < PRELIFT_LOST_GRACE_FRAMES:
                        if prelift_lost_frames == 1 or prelift_lost_frames % 6 == 0:
                            print(f"    Pre-lift contact transient: "
                                  f"lost={prelift_lost_frames}/"
                                  f"{PRELIFT_LOST_GRACE_FRAMES} "
                                  f"left={left_pad_hit} right={right_pad_hit} "
                                  f"finger_contacts={pad_contacts} "
                                  f"patch_span={patch_span*1000:.1f}mm "
                                  f"pair_skew={pair_skew*1000:.1f}mm "
                                  f"skin=({skin.left_force:.2f},"
                                  f"{skin.right_force:.2f})N")
                        grip_close_frames += 1
                        return
                    print(f"    Grasp destabilized during pre-lift hold: "
                          f"left={left_pad_hit} right={right_pad_hit} "
                          f"finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"center={geom['center_err']*100:.1f}cm")
                    remember_failed_grasp_frame("pre-lift hold lost contact")
                    diagnose_failure("pre-lift grip lost contact")
                    regrasp_count += 1
                    if regrasp_count >= MAX_REGRASP:
                        restart_search("pre-lift grip unstable")
                    else:
                        phase = 8; sub = 0
                    return
                prelift_lost_frames = 0
                force_ready = skin_squeeze_ok(skin, SKIN_PRELIFT_READY_FORCE)
                if force_ready:
                    prelift_ready_frames += 1
                else:
                    prelift_ready_frames = 0
                if prelift_hold_frames == 1 or prelift_hold_frames % 20 == 0:
                    print(f"    Pre-lift grip hold: contacts={contacts} "
                          f"finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"ready={prelift_ready_frames}/{GRIP_PRELIFT_READY_FRAMES} "
                          f"hold={prelift_hold_frames}/{GRIP_PRELIFT_MIN_FRAMES} "
                          f"center={geom['center_err']*100:.1f}cm")
                grip_close_frames += 1
                if (prelift_hold_frames >= GRIP_PRELIFT_MIN_FRAMES and
                        prelift_ready_frames >= GRIP_PRELIFT_READY_FRAMES):
                    print(f"    Pre-lift grip ready: "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"ready_force={SKIN_PRELIFT_READY_FORCE:.1f}N "
                          f"stable={prelift_ready_frames} frames; lifting now.")
                    phase = 5; sub = 0
                    status_msg = "Grip stable; lifting ..."
                elif gripper_ctrl.done:
                    print(f"    Pre-lift force did not reach target in time: "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"ready={prelift_ready_frames}/{GRIP_PRELIFT_READY_FRAMES} "
                          f"hold={prelift_hold_frames}/{GRIP_HOLD_FRAMES}")
                    remember_failed_grasp_frame("pre-lift force not ready")
                    diagnose_failure("pre-lift force not ready")
                    regrasp_count += 1
                    if regrasp_count >= MAX_REGRASP:
                        restart_search("pre-lift force not ready")
                    else:
                        phase = 8; sub = 0

        # 鈹€鈹€ Phase 5:  lift + verify 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 5:
            if sub == 0:
                print(">>> Phase 5 : Lift & verify")
                gripper_ctrl.hold(duration_frames=GRIP_TRANSPORT_HOLD_FRAMES)
                arm_ctrl.set_target(dynamic_ik_plan["lift"].angles,
                                    speed=SPEED_LIFT, min_frames=LIFT_MIN_FRAMES)
                lift_motion_frames = 0
                sub = 1
                status_msg = "Lifting ..."
            if sub == 1:
                lift_motion_frames += 1
                ensure_transport_grip_hold("lift")
                maintain_fused_grip_force("lift")
            lift_reached_safe_height = (
                lift_motion_frames >= LIFT_MIN_FRAMES + 55 and
                float(current_ball_xyz()[2] - ball_z_before_lift) >=
                GRASP_LIFT_HEIGHT * 0.78 and
                cube_transport_contact_ok()
            )
            if arm_ctrl.done or lift_reached_safe_height:
                # 鈹€鈹€ Verify: did the cube actually rise? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
                if lift_reached_safe_height and not arm_ctrl.done:
                    print(f"    Lift reached safe carry height before full settle "
                          f"({lift_motion_frames} frames); verifying grasp now.")
                ball_z_now = float(current_ball_xyz()[2])
                lifted = ball_z_now - ball_z_before_lift
                print(f"    Cube Z: before={ball_z_before_lift:.3f}  "
                      f"after={ball_z_now:.3f}  螖={lifted*100:.1f}cm")
                carry_err, carry_lift = cube_carry_metrics()
                print(f"    Carry check: err={carry_err*100:.1f}cm  "
                      f"lift={carry_lift*100:.1f}cm")
                if cube_is_secured(strict_visual=True):
                    print(f"    Grasp SUCCESS after {regrasp_count} retries")
                    place_correction_count = 0
                    place_support_rescan_count = 0
                    place_release_support_override = False
                    carry_last_remaining_xy = float("inf")
                    carry_stall_steps = 0
                    transport_regrip_count = 0
                    transport_regrip_frames = 0
                    transport_regrip_stable_frames = 0
                    if detected_box_pos is None:
                        print("    Target box is still unknown after grasp; "
                              "starting dedicated RGB-D box scan while holding.")
                        box_scan_targets = []
                        box_scan_idx = 0
                        rejected_grasp_frames = []
                        phase = 6
                        sub = -20
                        regrasp_count = 0
                        status_msg = "Locating target box while holding cube ..."
                        return
                    if not update_place_plan_for_carried_offset(
                            "lift verification"):
                        diagnose_failure("place compensation failed after lift")
                        restart_search("place compensation failed after lift")
                        return
                    rejected_grasp_frames = []
                    phase = 6; sub = 0; regrasp_count = 0
                else:
                    regrasp_count += 1
                    print(f"    Grasp FAILED (attempt {regrasp_count}/{MAX_REGRASP})")
                    remember_failed_grasp_frame("lift verification failed")
                    diagnose_failure("lift verification failed")
                    if regrasp_count >= MAX_REGRASP:
                        print("    Max retries exhausted - restart scanning")
                        restart_search("grasp not secure")
                    else:
                        phase = 8; sub = 0

        # 鈹€鈹€ Phase 6:  move to box 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 6:
            if sub in (-21, -20, -19, -18, 1, 2, 3, 4):
                ensure_transport_grip_hold(f"carry{sub}")
                maintain_fused_grip_force(f"carry{sub}")
            if sub in (1, 2, 3):
                carry_motion_frames += 1
                if carry_motion_frames % 80 == 0:
                    err, lifted = cube_carry_metrics()
                    q_err, max_qvel = arm_ctrl.settle_metrics()
                    cube_now = current_ball_xyz()
                    box_now = box_target_center()
                    remaining_xy = (
                        float(np.linalg.norm(box_now[:2] - cube_now[:2]))
                        if box_now is not None else float("nan")
                    )
                    print(f"    Phase 6 progress sub={sub} "
                          f"frames={carry_motion_frames} "
                          f"near={arm_ctrl.near_done(0.90)} "
                          f"done={arm_ctrl.done} "
                          f"qerr={q_err:.3f} qvel={max_qvel:.3f} "
                          f"err={err*100:.1f}cm lift={lifted*100:.1f}cm "
                          f"remain_xy={remaining_xy*100:.1f}cm")
            if sub in (1, 2):
                if (sub == 2 and cube_over_box_entry() and
                        visual_grip_weight > 0.80 and
                        visual_grip_drop_m > 0.010):
                    print("    Cube is over box opening with visible downward slip; "
                          "lowering now before grip weakens.")
                    if update_place_plan_for_carried_offset(
                            "proactive low place over box"):
                        arm_ctrl.set_target(
                            dynamic_ik_plan["place_drop"].angles,
                            speed=SPEED_PLACE,
                            min_frames=max(90, PLACE_DROP_MIN_FRAMES // 2),
                        )
                        carry_motion_frames = 0
                        sub = 3
                        status_msg = "Lowering cube into box ..."
                        return
                if not cube_transport_safe(f"carry{sub}", require_high_lift=True):
                    if sub == 2 and cube_over_box_entry():
                        print("    Transport grip degraded over box opening; "
                              "lowering immediately instead of restarting search.")
                        if update_place_plan_for_carried_offset(
                                "degraded carry over box"):
                            arm_ctrl.set_target(
                                dynamic_ik_plan["place_drop"].angles,
                                speed=SPEED_PLACE,
                                min_frames=max(90, PLACE_DROP_MIN_FRAMES // 2),
                            )
                            carry_motion_frames = 0
                            sub = 3
                            status_msg = "Lowering cube into box after degraded carry ..."
                            return
                    if begin_transport_regrip(
                            f"cube slipping during carry{sub}",
                            resume_sub=sub):
                        return
                    diagnose_failure(f"cube slipping during carry{sub}")
                    restart_search(f"cube slipping during carry{sub}")
                    return
            elif sub in (3, 4):
                if cube_in_box_drop_corridor():
                    transport_slip_abort_frames = 0
                elif not cube_transport_safe(f"carry{sub}", require_high_lift=False):
                    diagnose_failure(f"cube slipping during low placement{sub}")
                    restart_search(f"cube slipping during low placement{sub}")
                    return
            if sub == -30:
                ensure_transport_grip_hold("transport regrip")
                update_visual_grip_feedback(
                    "transport regrip",
                    force=True,
                )
                skin = read_skin()
                gripper_ctrl.maintain_tactile_force(
                    skin,
                    min_force=SKIN_HOLD_MIN_FORCE,
                    target_force=SKIN_PRELIFT_READY_FORCE,
                    max_force=SKIN_HOLD_MAX_FORCE,
                    correction_scale=0.30,
                )
                transport_regrip_frames += 1
                err, lifted = cube_carry_metrics()
                lin_v, _ = cube_velocity_norms()
                left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
                pair_skew, contact_center_err = finger_contact_alignment()
                stable = (
                    lifted > CARRY_MIN_LIFT and
                    err <= CARRY_MAX_ERR and
                    lin_v <= TRANSPORT_REGRIP_STABLE_SPEED and
                    left_pad_hit and right_pad_hit and
                    pad_contacts >= MIN_STABLE_FINGER_CONTACTS and
                    pair_skew <= MAX_CONTACT_PAIR_SKEW and
                    contact_center_err <= MAX_CONTACT_CENTER_ERR and
                    skin_squeeze_ok(skin, SKIN_PRELIFT_READY_FORCE)
                )
                if stable:
                    transport_regrip_stable_frames += 1
                else:
                    transport_regrip_stable_frames = max(
                        0,
                        transport_regrip_stable_frames - 1,
                    )
                if (transport_regrip_frames == 1 or
                        transport_regrip_frames % 20 == 0):
                    print(f"    Transport re-grip hold: "
                          f"stable={transport_regrip_stable_frames}/"
                          f"{TRANSPORT_REGRIP_STABLE_FRAMES} "
                          f"err={err*100:.1f}cm lift={lifted*100:.1f}cm "
                          f"v={lin_v:.3f}m/s contacts={pad_contacts} "
                          f"left={left_pad_hit} right={right_pad_hit} "
                          f"skin=({skin.left_force:.2f},"
                          f"{skin.right_force:.2f})N")
                if cube_inside_box():
                    print("    Cube entered box during transport re-grip; "
                          "switching to release.")
                    phase = 7
                    sub = 0
                    return
                if transport_regrip_stable_frames >= TRANSPORT_REGRIP_STABLE_FRAMES:
                    print("    Transport re-grip stable; recomputing next "
                          "short carry step.")
                    reset_visual_grip_feedback("transport regrip stable")
                    update_visual_grip_feedback(
                        "transport regrip stable",
                        force=True,
                    )
                    box_now = box_target_center()
                    direct_after_regrip = (
                        box_now is not None and
                        box_entry_gap_xy(current_ball_xyz(), box_now) <=
                        CARRY_STEP_FINAL_DIRECT_XY_M
                    )
                    if not update_place_plan_for_carried_offset(
                            "transport regrip stable",
                            direct_to_box=direct_after_regrip):
                        diagnose_failure("transport regrip compensation failed")
                        restart_search("transport regrip compensation failed")
                        return
                    transport_slip_abort_frames = 0
                    if cube_over_box_entry():
                        arm_ctrl.set_target(
                            dynamic_ik_plan["place_drop"].angles,
                            speed=SPEED_PLACE,
                            min_frames=PLACE_DROP_MIN_FRAMES,
                        )
                        carry_motion_frames = 0
                        sub = 3
                        status_msg = "Lowering cube into box after re-grip ..."
                    else:
                        arm_ctrl.set_target(
                            dynamic_ik_plan["place_above"].angles,
                            speed=(
                                SPEED_CARRY * 0.38
                                if direct_after_regrip else SPEED_CARRY
                            ),
                            min_frames=(
                                max(CARRY_STEP_MIN_FRAMES, 190)
                                if direct_after_regrip else CARRY_STEP_MIN_FRAMES
                            ),
                        )
                        carry_motion_frames = 0
                        sub = max(1, transport_regrip_resume_sub)
                        status_msg = "Continuing carry after re-grip ..."
                    return
                if transport_regrip_frames >= TRANSPORT_REGRIP_FRAMES:
                    diagnose_failure("transport regrip failed")
                    restart_search("transport regrip failed")
                    return
                return
            if sub == -21:
                if not arm_ctrl.done:
                    status_msg = "Returning above box for visual re-localization ..."
                    return
                detected_box_pos = None
                box_rgbd_scan_estimates = []
                box_scan_targets = []
                box_scan_idx = 0
                box_scan_hint_xy_used = None
                sub = -20
                status_msg = "Re-localizing target box before release ..."
                return
            if sub == -20:
                if not cube_is_secured():
                    diagnose_failure("cube not secure before box scan")
                    restart_search("cube not secure before box scan")
                    return
                box_scan_targets = generate_box_scan_targets(
                    box_search_hint_xy,
                    z_plane=HELD_BOX_SCAN_Z_PLANE,
                )
                box_scan_idx = 0
                box_scan_hint_xy_used = (
                    box_search_hint_xy.copy()
                    if box_search_hint_xy is not None else None
                )
                print(f">>> Phase 6 : Locate target box while holding cube "
                      f"({len(box_scan_targets)} candidate views, "
                      f"hint={np.round(box_search_hint_xy, 3) if box_search_hint_xy is not None else None})")
                sub = -19
                status_msg = "Scanning for target box while holding cube ..."
                return
            if sub == -19:
                box_xyz = capture_box_from_rgbd(
                    "held-cube box current view",
                    log_miss=(detected_box_pos is None),
                )
                if box_xyz is not None and accept_box_scan_estimate(
                        box_xyz, "held-cube current view"):
                    if update_place_plan_for_carried_offset(
                            "box located while holding"):
                        sub = 0
                        return
                    diagnose_failure("place compensation failed after box scan")
                    restart_search("place compensation failed after box scan")
                    return
                if box_xyz is not None and reprioritize_box_scan_from_hint(
                        "current view saw partial box"):
                    return
                while box_scan_idx < len(box_scan_targets):
                    target = box_scan_targets[box_scan_idx]
                    if not held_box_scan_target_is_safe(target):
                        box_scan_idx += 1
                        continue
                    result = solve_gripper_center_ik(
                        model, data, target.gripper,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=SCAN_POS_TOL,
                        target_xmat=target.xmat, orientation_body_id=tool_id,
                        orientation_weight=SCAN_ORI_WEIGHT, orientation_tol=SCAN_ORI_TOL,
                    )
                    if result.success:
                        print(f"    Moving to held-cube box scan "
                              f"{box_scan_idx+1}/{len(box_scan_targets)} "
                              f"aim={np.round(target.aim, 3)}")
                        arm_ctrl.set_target(
                            result.angles,
                            speed=SPEED_CARRY,
                            min_frames=max(120, CARRY_MID_MIN_FRAMES // 2),
                        )
                        sub = -18
                        status_msg = "Moving camera for box scan ..."
                        return
                    box_scan_idx += 1
                if accept_partial_box_hint_for_place("held scan exhausted"):
                    return
                diagnose_failure("target box not found after held scan")
                restart_search("target box not found after held scan")
                return
            if sub == -18:
                if not cube_is_secured():
                    diagnose_failure("cube dropped during held box scan")
                    restart_search("cube dropped during held box scan")
                    return
                if not arm_ctrl.done:
                    if arm_ctrl.near_done(SCAN_MOVING_CAPTURE_PROGRESS):
                        box_xyz = capture_box_from_rgbd(
                            f"moving held-cube box scan {box_scan_idx+1}",
                            log_miss=False,
                        )
                        if box_xyz is not None and accept_box_scan_estimate(
                                box_xyz, f"moving held-cube box scan {box_scan_idx+1}"):
                            if update_place_plan_for_carried_offset(
                                    "box located during held scan"):
                                sub = 0
                                return
                            diagnose_failure("place compensation failed during box scan")
                            restart_search("place compensation failed during box scan")
                            return
                        if box_xyz is not None and reprioritize_box_scan_from_hint(
                                f"moving scan {box_scan_idx+1} saw partial box"):
                            sub = -19
                            return
                    return
                box_xyz = capture_box_from_rgbd(
                    f"held-cube box scan {box_scan_idx+1}",
                    log_miss=(detected_box_pos is None),
                )
                if box_xyz is not None and accept_box_scan_estimate(
                        box_xyz, f"held-cube settled box scan {box_scan_idx+1}"):
                    if update_place_plan_for_carried_offset(
                            "box located after held scan"):
                        sub = 0
                        return
                    diagnose_failure("place compensation failed after settled box scan")
                    restart_search("place compensation failed after settled box scan")
                    return
                if box_xyz is not None and reprioritize_box_scan_from_hint(
                        f"settled scan {box_scan_idx+1} saw partial box"):
                    sub = -19
                    return
                if box_scan_idx >= len(box_scan_targets) - 1:
                    if accept_partial_box_hint_for_place(
                            f"settled scan {box_scan_idx+1} exhausted"):
                        return
                box_scan_idx += 1
                sub = -19
                return
            if sub == 0:
                print(">>> Phase 6 : Carry via midpoint")
                arm_ctrl.set_target(dynamic_ik_plan["carry_mid"].angles,
                                    speed=SPEED_CARRY, min_frames=CARRY_MID_MIN_FRAMES)
                carry_motion_frames = 0
                sub = 1
                status_msg = "Carrying cube ..."
            elif sub == 1 and (
                    arm_ctrl.done or
                    (arm_ctrl.near_done(0.90) and
                     carry_segment_can_advance(CARRY_MID_MIN_FRAMES)) or
                    carry_segment_can_advance(CARRY_MID_MIN_FRAMES) or
                    carry_motion_frames >= CARRY_MID_MIN_FRAMES + 120):
                if not arm_ctrl.done:
                    q_err, max_qvel = arm_ctrl.settle_metrics()
                    print(f"    Carry midpoint accepted without full settle: "
                          f"frames={carry_motion_frames} "
                          f"qerr={q_err:.3f} qvel={max_qvel:.3f}")
                if not (
                        cube_transport_contact_ok() or
                        cube_transport_relaxed_hold_ok() or
                        cube_transport_geometry_hold_ok() or
                        cube_transport_stiction_hold_ok()):
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube already reached box during midpoint carry: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    if begin_transport_regrip(
                            "cube dropped before carry midpoint",
                            resume_sub=1):
                        return
                    diagnose_failure("cube dropped before carry midpoint")
                    restart_search("cube dropped before carry midpoint")
                    return
                if not update_place_plan_for_carried_offset("carry midpoint"):
                    diagnose_failure("place compensation failed at midpoint")
                    restart_search("place compensation failed at midpoint")
                    return
                box_now = box_target_center()
                carry_last_remaining_xy = (
                    box_entry_gap_xy(current_ball_xyz(), box_now)
                    if box_now is not None else float("inf")
                )
                carry_stall_steps = 0
                print(">>> Phase 6a : Move above box")
                arm_ctrl.set_target(dynamic_ik_plan["place_above"].angles,
                                    speed=SPEED_CARRY, min_frames=CARRY_STEP_MIN_FRAMES)
                carry_motion_frames = 0
                sub = 2
                status_msg = "Moving above box ..."
            elif sub == 2 and (
                    arm_ctrl.done or
                    (arm_ctrl.near_done(0.90) and
                     carry_segment_can_advance(CARRY_STEP_MIN_FRAMES)) or
                    carry_segment_can_advance(CARRY_STEP_MIN_FRAMES) or
                    carry_motion_frames >= CARRY_STEP_MIN_FRAMES + 120):
                box_now = box_target_center()
                cube_now = current_ball_xyz()
                remaining_xy = (
                    float(np.linalg.norm(box_now[:2] - cube_now[:2]))
                    if box_now is not None else float("nan")
                )
                entry_gap_xy = (
                    box_entry_gap_xy(cube_now, box_now)
                    if box_now is not None else float("nan")
                )
                entry_ready = cube_over_box_entry()
                drop_ready = cube_ready_for_guided_place_drop()
                progress_xy = (
                    carry_last_remaining_xy - entry_gap_xy
                    if (math.isfinite(carry_last_remaining_xy) and
                        math.isfinite(entry_gap_xy))
                    else float("inf")
                )
                if (math.isfinite(progress_xy) and
                        entry_gap_xy > TRANSPORT_STALL_PROGRESS_EPS_M and
                        progress_xy < TRANSPORT_STALL_PROGRESS_EPS_M):
                    carry_stall_steps += 1
                else:
                    carry_stall_steps = 0
                carry_last_remaining_xy = entry_gap_xy
                if not arm_ctrl.done:
                    q_err, max_qvel = arm_ctrl.settle_metrics()
                    print(f"    Carry step accepted without full settle: "
                          f"frames={carry_motion_frames} "
                          f"qerr={q_err:.3f} qvel={max_qvel:.3f} "
                          f"remain_xy={remaining_xy*100:.1f}cm "
                          f"entry_gap={entry_gap_xy*100:.1f}cm "
                          f"progress={progress_xy*100:.1f}cm "
                          f"stall={carry_stall_steps}/"
                          f"{TRANSPORT_STALL_MAX_STEPS} "
                          f"entry={entry_ready} drop_ready={drop_ready}")
                if not (
                        cube_transport_contact_ok() or
                        cube_transport_relaxed_hold_ok() or
                        cube_transport_geometry_hold_ok() or
                        cube_transport_stiction_hold_ok()):
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube already reached box during carry: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    if cube_over_box_entry():
                        print("    Carry reached box opening with weakening grip; "
                              "lowering immediately.")
                        if update_place_plan_for_carried_offset(
                                "weak grip at box opening"):
                            arm_ctrl.set_target(
                                dynamic_ik_plan["place_drop"].angles,
                                speed=SPEED_PLACE,
                                min_frames=max(90, PLACE_DROP_MIN_FRAMES // 2),
                            )
                            carry_motion_frames = 0
                            sub = 3
                            status_msg = "Lowering cube into box ..."
                            return
                    if begin_transport_regrip(
                            "cube dropped while carrying",
                            resume_sub=2):
                        return
                    diagnose_failure("cube dropped while carrying")
                    restart_search("cube dropped while carrying")
                    return
                if not (cube_over_box_entry() or
                        cube_ready_for_guided_place_drop()):
                    force_high_clearance = (
                        carry_stall_steps >= TRANSPORT_STALL_MAX_STEPS
                    )
                    direct_to_box = (
                        entry_gap_xy <= CARRY_STEP_FINAL_DIRECT_XY_M or
                        carry_stall_steps >= TRANSPORT_STALL_MAX_STEPS
                    )
                    context = (
                        "carry stall recovery" if force_high_clearance
                        else "carry step"
                    )
                    if not update_place_plan_for_carried_offset(
                            context,
                            direct_to_box=direct_to_box,
                            high_clearance=force_high_clearance):
                        diagnose_failure("carry step compensation failed")
                        restart_search("carry step compensation failed")
                        return
                    print(">>> Phase 6a : Continue carry step toward box")
                    if box_now is not None:
                        remain_xy = float(np.linalg.norm(
                            box_now[:2] - cube_now[:2]))
                        entry_gap = box_entry_gap_xy(cube_now, box_now)
                        print(f"    Next carry target: cube={np.round(cube_now, 4)} "
                              f"box={np.round(box_now, 4)} "
                              f"remain_xy={remain_xy*100:.1f}cm "
                              f"entry_gap={entry_gap*100:.1f}cm "
                              f"final_direct={CARRY_STEP_FINAL_DIRECT_XY_M*100:.1f}cm "
                              f"direct={direct_to_box} high={force_high_clearance}")
                    arm_ctrl.set_target(
                        dynamic_ik_plan["place_above"].angles,
                        speed=(
                            SPEED_CARRY * 0.38
                            if direct_to_box else SPEED_CARRY
                        ),
                        min_frames=(
                            max(CARRY_STEP_MIN_FRAMES, 190)
                            if direct_to_box else
                            (
                                max(CARRY_STEP_MIN_FRAMES, 105)
                                if force_high_clearance else
                                CARRY_STEP_MIN_FRAMES
                            )
                        ),
                    )
                    carry_motion_frames = 0
                    status_msg = (
                        "Recovering carry path over box ..."
                        if force_high_clearance else
                        "Carrying cube in short steps ..."
                    )
                    return
                if not update_place_plan_for_carried_offset(
                        "above box", direct_to_box=True):
                    diagnose_failure("place compensation failed above box")
                    restart_search("place compensation failed above box")
                    return
                print(">>> Phase 6b : Lower into box")
                arm_ctrl.set_target(dynamic_ik_plan["place_drop"].angles,
                                    speed=SPEED_PLACE, min_frames=PLACE_DROP_MIN_FRAMES)
                carry_motion_frames = 0
                sub = 3
                status_msg = "Lowering into box ..."
            elif sub == 3 and (
                    arm_ctrl.done or
                    (arm_ctrl.near_done(0.90) and
                     carry_segment_can_advance(
                         PLACE_DROP_MIN_FRAMES,
                         qerr_limit=0.20,
                         qvel_limit=0.28,
                     )) or
                    carry_segment_can_advance(
                        PLACE_DROP_MIN_FRAMES,
                        qerr_limit=0.20,
                        qvel_limit=0.28,
                    ) or
                    carry_motion_frames >= PLACE_DROP_MIN_FRAMES + 120):
                if not arm_ctrl.done:
                    q_err, max_qvel = arm_ctrl.settle_metrics()
                    print(f"    Place drop accepted without full settle: "
                          f"frames={carry_motion_frames} "
                          f"qerr={q_err:.3f} qvel={max_qvel:.3f}")
                if not cube_is_secured():
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube already settled in box during lowering: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    if cube_in_box_drop_corridor():
                        print("    Grip weakened inside box drop corridor; "
                              "continuing release-window correction instead of "
                              "global restart.")
                    else:
                        diagnose_failure("cube dropped before release")
                        restart_search("cube dropped before release")
                        return
                ready, reason = cube_release_window(require_box_support=False)
                if not ready:
                    if place_correction_count < MAX_PLACE_CORRECTIONS:
                        place_correction_count += 1
                        print(f"    Place release correction "
                              f"{place_correction_count}/{MAX_PLACE_CORRECTIONS}: "
                              f"{reason}")
                        if update_place_plan_for_carried_offset(
                                "release-window correction"):
                            arm_ctrl.set_target(
                                dynamic_ik_plan["place_drop"].angles,
                                speed=SPEED_PLACE,
                                min_frames=max(90, PLACE_DROP_MIN_FRAMES // 2),
                            )
                            carry_motion_frames = 0
                            status_msg = "Correcting cube into box ..."
                            return
                    print(f"    Refusing to release outside low target window: {reason}")
                    diagnose_failure("place drop target not reached")
                    restart_search("place drop target not reached")
                    return
                print(f"    Place drop ready: {reason}")
                place_settle_frames = 0
                sub = 4
                status_msg = "Settling cube low in box before release ..."
            elif sub == 4:
                if not cube_is_secured():
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube settled in box before opening gripper: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    diagnose_failure("cube dropped during pre-release settle")
                    restart_search("cube dropped during pre-release settle")
                    return
                low_ready, low_reason = cube_release_window(require_box_support=False)
                ready, reason = cube_release_window()
                if not low_ready:
                    if place_correction_count < MAX_PLACE_CORRECTIONS:
                        place_correction_count += 1
                        print(f"    Pre-release correction "
                              f"{place_correction_count}/{MAX_PLACE_CORRECTIONS}: "
                              f"{low_reason}")
                        if update_place_plan_for_carried_offset(
                                "pre-release correction"):
                            arm_ctrl.set_target(
                                dynamic_ik_plan["place_drop"].angles,
                                speed=SPEED_PLACE,
                                min_frames=max(90, PLACE_DROP_MIN_FRAMES // 2),
                            )
                            carry_motion_frames = 0
                            place_settle_frames = 0
                            sub = 3
                            status_msg = "Correcting low release pose ..."
                            return
                    print(f"    Pre-release settle left target window: {low_reason}")
                    diagnose_failure("cube left place release window")
                    restart_search("cube left place release window")
                    return
                if not ready:
                    place_settle_frames += 1
                    if place_settle_frames % 20 == 1:
                        print(f"    Waiting for box-bottom support before release "
                              f"{place_settle_frames}/{PLACE_SETTLE_HOLD_FRAMES}: "
                              f"{reason}")
                    if place_settle_frames < PLACE_SETTLE_HOLD_FRAMES:
                        return
                    print(f"    No box-bottom support at release pose: {reason}")
                    diagnose_failure("release pose has no box-bottom support")
                    if (place_support_rescan_count < MAX_PLACE_SUPPORT_RESCANS and
                            "place_above" in dynamic_ik_plan):
                        place_support_rescan_count += 1
                        print("    Re-localizing target box while holding cube; "
                              "release is blocked until box-bottom support is confirmed "
                              f"({place_support_rescan_count}/"
                              f"{MAX_PLACE_SUPPORT_RESCANS}).")
                        arm_ctrl.set_target(
                            dynamic_ik_plan["place_above"].angles,
                            speed=SPEED_PLACE,
                            min_frames=max(90, PLACE_ABOVE_MIN_FRAMES // 2),
                        )
                        carry_motion_frames = 0
                        detected_box_pos = None
                        box_rgbd_scan_estimates = []
                        place_settle_frames = 0
                        sub = -21
                        status_msg = "No box support; re-scanning box while holding ..."
                        return
                    print("    Box-bottom contact is still missing after the bounded "
                          "re-scan. Low XY/Z window is valid, so proceeding with a "
                          "guarded slow release to let the cube settle into the box.")
                    place_release_support_override = True
                    phase = 7
                    sub = 0
                    status_msg = "Guarded low release inside box ..."
                    return
                place_settle_frames += 1
                if place_settle_frames % 20 == 1:
                    print(f"    Pre-release low hold "
                          f"{place_settle_frames}/{PLACE_SETTLE_HOLD_FRAMES}: "
                          f"{reason}")
                if place_settle_frames < PLACE_SETTLE_HOLD_FRAMES:
                    return
                place_release_support_override = False
                phase = 7; sub = 0
        elif phase == 7:
            if sub == 0:
                print(">>> Phase 7 : Release into box")
                require_support = not place_release_support_override
                ready, reason = cube_release_window(
                    require_box_support=require_support,
                )
                if not ready:
                    print(f"    Release blocked: {reason}")
                    diagnose_failure("release requested outside box")
                    place_release_support_override = False
                    restart_search("release requested outside box")
                    return
                if place_release_support_override:
                    print(f"    Guarded low release without prior box-bottom "
                          f"contact: {reason}")
                release_cube_static_anchor("release in box")
                release_grip_lock("release in box")
                dynamic_grasp_xmat = None
                dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
                carried_cube_offset = None
                place_release_support_override = False
                gripper_ctrl.open(duration_frames=PLACE_RELEASE_OPEN_FRAMES)
                sub = 1
                status_msg = "Softly opening gripper in box ..."
            elif sub == 1 and gripper_ctrl.done:
                box_verify_left = BOX_VERIFY_FRAMES
                sub = 2
                status_msg = "Verifying cube inside box ..."
            elif sub == 2:
                box_verify_left -= 1
                if box_verify_left > 0:
                    return
                if cube_inside_box():
                    cube = current_ball_xyz()
                    print(f"    Cube placed in box at {np.round(cube, 3)}")
                    phase = 9; sub = 0
                else:
                    diagnose_failure("cube released but not inside box")
                    restart_search("cube missed target box")

        # 鈹€鈹€ Phase 8:  re-grasp 鈥?release, settle, retry 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 8:
            if sub == 0:
                print(f">>> Phase 8 : Re-grasp ({regrasp_count}/{MAX_REGRASP})")
                carried_cube_offset = None
                release_grip_lock("re-grasp")
                gripper_ctrl.open(duration_frames=20)
                sub = 1
            elif sub == 1:
                if not gripper_ctrl.done or not arm_ctrl.done:
                    return
                # Wait for cube to settle (velocity below threshold)
                bv, bw = cube_velocity_norms()
                cube_now = current_ball_xyz()
                if (cube_now[2] > CUBE_STATIC_ANCHOR_MAX_Z and
                        not cube_inside_box()):
                    recovery_wait_frames = 0
                    recovery_stable_frames = 0
                    recovery_reason = "local re-grasp high cube"
                    gripper_ctrl.open(duration_frames=HIGH_CUBE_RECOVERY_OPEN_FRAMES)
                    sub = 10
                    status_msg = (
                        "Cube is high after failed grasp; waiting for it to settle ..."
                    )
                    return
                if bv < 0.02:
                    print(f"    Cube settled (|v|={bv:.3f} m/s)")
                    if cube_inside_box():
                        print("    Cube is already inside the target box after retry.")
                        phase = 7; sub = 0
                        return
                    set_cube_static_anchor("local re-grasp retry")
                    fresh = capture_cube_from_rgbd("local re-grasp")
                    if fresh is None:
                        diagnose_failure("local re-grasp RGB-D reacquire failed")
                        begin_local_rgbd_replan(
                            "local re-grasp RGB-D reacquire failed")
                        return
                    detected_ball_pos = fresh.copy()
                    box = box_for_pregrasp_replan(
                        detected_ball_pos, "Local re-grasp")
                    z_offs = GRASP_Z_OFFSETS[
                        min(regrasp_count, len(GRASP_Z_OFFSETS) - 1)]
                    previous_xmat = (
                        None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                    )
                    replanned = compute_dynamic_ik(
                        detected_ball_pos,
                        box,
                        grasp_z_offs=z_offs,
                        opening_hints=detected_opening_hints,
                        preview_top_k=LOCAL_PHYSICAL_EVAL_TOP_K,
                    )
                    if plan_is_ready(replanned):
                        dynamic_ik_plan = replanned
                        print("    Local re-grasp plan ready; retrying without "
                              "returning to global scan.")
                        pregrasp_replan_count = 0
                        local_replan_count = 0
                        phase = 2; sub = 0
                        status_msg = "Retrying grasp from current cube pose ..."
                        return
                    dynamic_grasp_xmat = previous_xmat
                    diagnose_failure("local re-grasp planning failed")
                    begin_local_rgbd_replan("local re-grasp plan failed")
                    return
                else:
                    status_msg = f"Waiting for cube to settle (|v|={bv:.3f}) ..."
            elif sub == 10:
                release_cube_static_anchor("high-cube recovery")
                if not gripper_ctrl.done:
                    status_msg = "Opening gripper for high-cube recovery ..."
                    return
                recovery_wait_frames += 1
                cube_now = current_ball_xyz()
                bv, bw = cube_velocity_norms()
                transport_recovery = any(
                    key in recovery_reason for key in (
                        "carry",
                        "transport",
                        "release",
                        "place",
                        "box",
                        "slipping",
                        "dropped",
                    )
                )
                recovery_floor_z = (
                    CUBE_REST_Z + 0.004
                    if transport_recovery else CUBE_STATIC_ANCHOR_MAX_Z
                )
                near_recovery_floor = cube_now[2] <= recovery_floor_z
                low_motion = (
                    bv <= HIGH_CUBE_SETTLE_SPEED and
                    bw <= HIGH_CUBE_SETTLE_ANG_SPEED
                )
                if near_recovery_floor and low_motion:
                    recovery_stable_frames += 1
                else:
                    recovery_stable_frames = 0
                if recovery_wait_frames == 1 or recovery_wait_frames % 25 == 0:
                    print(f"    High cube recovery: cube={np.round(cube_now, 4)} "
                          f"|v|={bv:.3f} |w|={bw:.3f} "
                          f"stable={recovery_stable_frames}/"
                          f"{HIGH_CUBE_SETTLE_FRAMES} "
                          f"reason={recovery_reason}")
                if recovery_stable_frames >= HIGH_CUBE_SETTLE_FRAMES:
                    print("    Cube recovered to table-height; restarting RGB-D scan.")
                    set_cube_static_anchor("high-cube recovery settled")
                    phase = 0
                    sub = 0
                    status_msg = "Cube settled; rescanning ..."
                    return
                if recovery_wait_frames >= HIGH_CUBE_SETTLE_TIMEOUT_FRAMES:
                    print("    High cube recovery timed out; rescanning without "
                          "static anchor so RGB-D can localize the current pose.")
                    phase = 0
                    sub = 0
                    status_msg = "Recovery timeout; rescanning current cube pose ..."
                    return
                status_msg = (
                    "Waiting for high cube to fall/settle "
                    f"z={cube_now[2]:.3f} |v|={bv:.3f} ..."
                )

        # 鈹€鈹€ Phase 9:  done 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 9:
            print(">>> Demo complete 鈥?manual control restored.")
            status_msg = "DONE 鈥?manual control"
            finished = True

    # 鈹€鈹€ Main loop 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\nLaunching viewer + windows ...")
    print("  Sliders appear in the MuJoCo viewer Control Panel")
    print()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.26, 0.0, 0.16]

        scene_annotations_ready = False
        frame = 0
        last_skin_reading = read_skin()
        auto_start_demo = os.environ.get("SYNRIA_AUTO_START", "").lower() in {
            "1", "true", "yes", "on",
        }
        auto_exit_when_done = os.environ.get("SYNRIA_AUTO_EXIT", "").lower() in {
            "1", "true", "yes", "on",
        }

        # Warmup 鈥?settle physics
        for _ in range(30):
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, data)
            frame += 1

        while viewer.is_running():
            frame_start = time.perf_counter()

            if auto_start_demo and phase == -1 and start_act_id >= 0:
                data.ctrl[start_act_id] = START_TRIGGER_THRESHOLD
                auto_start_demo = False

            # 鈹€鈹€ 1. Input: slider-based cube placement + START 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if phase == -1 and cube_x_act_id >= 0:
                bx = data.ctrl[cube_x_act_id]
                by = data.ctrl[cube_y_act_id]
                bz = data.ctrl[cube_z_act_id]
                # Only move the cube when a slider actually changed
                if (abs(bx - _last_bx) > 0.0005 or
                    abs(by - _last_by) > 0.0005 or
                    abs(bz - _last_bz) > 0.0005):
                    set_cube_xyz(np.array([bx, by, bz], dtype=np.float64),
                                 sync_sliders=True)

            # start_demo slider 鈫?trigger
            if phase == -1 and ball_qpos_adr >= 0:
                live_ball = current_ball_xyz()
                if (abs(live_ball[0] - _last_bx) > 0.001 or
                    abs(live_ball[1] - _last_by) > 0.001 or
                    abs(live_ball[2] - _last_bz) > 0.001):
                    sync_cube_sliders(live_ball)

            if (phase == -1 and start_act_id >= 0 and
                    data.ctrl[start_act_id] >= START_TRIGGER_THRESHOLD):
                print("\n*** START 鈥?beginning autonomous sequence ***\n")
                # Freeze cube at its current slider position with zero velocity
                if ball_qpos_adr >= 0:
                    current_ball = current_ball_xyz()
                    set_cube_xyz(current_ball, sync_sliders=True)
                    set_cube_static_anchor("start")
                    pregrasp_replan_count = 0
                    local_replan_count = 0
                    place_settle_frames = 0
                    place_correction_count = 0
                    place_support_rescan_count = 0
                    place_release_support_override = False
                    prelift_hold_frames = 0
                    prelift_ready_frames = 0
                    carried_cube_offset = None
                    detected_ball_pos = None
                    detected_box_pos = None
                    rgbd_scan_estimates = []
                    box_rgbd_scan_estimates = []
                    box_search_hint_xy = None
                    box_scan_targets = []
                    box_scan_idx = 0
                    box_scan_hint_xy_used = None
                    dynamic_grasp_xmat = None
                    dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
                    rejected_grasp_frames = []
                phase = 0; sub = 0
                status_msg = "Starting cube search ..."
                # Reset trigger slider
                data.ctrl[start_act_id] = 0.0

            # OpenCV event pump
            if CV2_AVAILABLE:
                cv2.waitKey(1)

            # 鈹€鈹€ 2. State machine 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            try:
                demo_tick()
            except Exception as exc:
                print(f"ERROR in demo_tick: {exc}")
                import traceback; traceback.print_exc()
                finished = True
                status_msg = f"ERROR: {exc}"

            # 鈹€鈹€ 3. Control (skip during idle 鈫?manual sliders work) 鈹€鈹€鈹€鈹€
            try:
                if not finished and phase != -1:
                    arm_ctrl.step()
                    gripper_ctrl.step()
            except Exception as exc:
                print(f"ERROR in controller step: {exc}")
                import traceback; traceback.print_exc()
                finished = True
                status_msg = f"ERROR: {exc}"

            # 鈹€鈹€ 4. Physics substeps 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            for _ in range(PHYSICS_SUBSTEPS):
                apply_cube_static_anchor()
                apply_grip_lock()
                mujoco.mj_step(model, data)
                apply_cube_static_anchor()
                apply_grip_lock()

            # 鈹€鈹€ 4b. Clamp cube inside workspace 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if ball_qpos_adr >= 0:
                bx, by, bz = (data.qpos[ball_qpos_adr],
                              data.qpos[ball_qpos_adr + 1],
                              data.qpos[ball_qpos_adr + 2])
                clamped = False
                if phase in (-1, 0, 1, 2, 8, 9):
                    projected_xy = workspace_project_xy(np.array([bx, by], dtype=np.float64))
                    if np.linalg.norm(projected_xy - np.array([bx, by], dtype=np.float64)) > 1e-5:
                        bx, by = float(projected_xy[0]), float(projected_xy[1])
                        clamped = True
                if bz < CUBE_REST_Z: bz = CUBE_REST_Z; clamped = True
                if bz > 0.30:  bz = 0.30; clamped = True
                if clamped:
                    data.qpos[ball_qpos_adr] = bx
                    data.qpos[ball_qpos_adr + 1] = by
                    data.qpos[ball_qpos_adr + 2] = bz
                    zero_cube_velocity()
                    mujoco.mj_forward(model, data)

            stabilize_cube_at_rest()

            # 鈹€鈹€ 5. Workspace spheres (once) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            try:
                render_scene_annotations(
                    viewer,
                    workspace_pts,
                    detected_ball_pos,
                    box_target_center(),
                )
                scene_annotations_ready = True
            except Exception as exc:
                if not scene_annotations_ready:
                    print(f"Scene annotation init: {exc}")

            # 鈹€鈹€ 6. Displays (throttled) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            if frame % SKIN_EVERY_N == 0:
                last_skin_reading = read_skin()
                skin_display.update(last_skin_reading)
                skin_display.show()

            if joint_panel and frame % JOINT_EVERY_N == 0:
                idle_hint = "" if phase != -1 else " [slide start_demo to 1]"
                joint_panel.update(model, data, arm_joints, gripper_joints,
                                   status_text=status_msg + idle_hint)
                joint_panel.show()

            if rgbd_window and rgbd_window.should_update(frame):
                hint = " | [slide start_demo to 1]" if phase == -1 else ""
                rgbd_window.update(
                    data,
                    overlay_text=(
                        f"Skin L/R={last_skin_reading.left_force:.2f}/"
                        f"{last_skin_reading.right_force:.2f}N{hint}"
                    ),
                )

            # 鈹€鈹€ 7. Viewer sync 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            try:
                viewer.sync()
            except Exception:
                pass

            if auto_exit_when_done and finished:
                break

            frame += 1

            # 鈹€鈹€ 8. Frame pacing 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
            elapsed = time.perf_counter() - frame_start
            if elapsed < FRAME_DT:
                remaining = FRAME_DT - elapsed
                if remaining > 0.003:
                    time.sleep(remaining - 0.0015)
                while time.perf_counter() - frame_start < FRAME_DT:
                    pass

    skin_display.close()
    if rgbd_window:
        rgbd_window.close()
    if joint_panel:
        joint_panel.close()
    print("Done.")


if __name__ == "__main__":
    main()
