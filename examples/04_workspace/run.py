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

from common.model_loader import RGB_CAMERA_NAME, BALL_POS
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

SKIN_EVERY_N = 2       # 25 Hz
JOINT_EVERY_N = 4      # 12.5 Hz
CAMERA_EVERY_N = 2     # 25 Hz
SCAN_RGBD_CAPTURE_EVERY_N = 4  # Opportunistic scan detection while moving.

SPEED_NORMAL = 1.4
SPEED_SLOW = 0.55  # very slow for precise grasp positioning
SPEED_SCAN = 1.0
SPEED_LIFT = 0.12
SPEED_CARRY = 0.12
SPEED_PLACE = 0.32
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

MIN_BALL_AREA_PX = 12       # minimum contour area (small cube at distance 鈮?15-30 px)
CUBE_HALF_SIZE = 0.020  # 40 mm cube side 鈥?good grip depth in 50 mm gripper
PAD_HALF_HEIGHT = 0.003
CUBE_REST_Z = CUBE_HALF_SIZE + PAD_HALF_HEIGHT
CAMERA_TO_GRIPPER_CENTER = 0.046
SCAN_CAMERA_DISTANCE = 0.24
MIN_DEPTH_SAMPLES = 8
MAX_RGBD_DEPTH_M = 2.0
VISION_Z_MIN = -0.03
VISION_Z_MAX = 0.45
GRASP_ORI_WEIGHT = 0.36
GRASP_ORI_TOL = 0.08
LIFT_ORI_TOL = 0.12
PLACE_ORI_TOL = 0.22
GRASP_WORLD_AXIS_WEIGHT = 14.0
RGBD_MAX_ESTIMATE_SPREAD_M = 0.055
RGBD_MAX_PLANE_RESIDUAL_M = 0.010
RGBD_MIN_PLANE_INLIERS = 35
GLOBAL_RGBD_MIN_VIEWS = 2
GLOBAL_RGBD_MAX_VIEWS = 5
GLOBAL_RGBD_STABILITY_M = 0.015
GLOBAL_RGBD_STRONG_SINGLE_VIEW_INLIERS = 80
GLOBAL_RGBD_STRONG_SINGLE_VIEW_RESIDUAL_M = 0.006
GLOBAL_RGBD_STRONG_SINGLE_VIEW_SPREAD_M = 0.035
GRIP_CONTACT_FORCE = 9.0
GRIP_CONFIRM_FORCE = 5.8
GRIP_LOCK_MIN_FORCE = 1.5
GRIP_MAX_FORCE = 80.0
SKIN_TOUCH_FORCE = 0.03
SKIN_CONFIRM_FORCE = 5.0
SKIN_HOLD_MIN_FORCE = 6.0
SKIN_HOLD_TARGET_FORCE = 10.0
SKIN_HOLD_MAX_FORCE = 24.0
SKIN_BALANCE_MAX_RATIO = 0.80
VISUAL_GRIP_CAPTURE_EVERY_N = 8
VISUAL_GRIP_LOG_EVERY_N = 35
VISUAL_GRIP_FILTER_ALPHA = 0.35
VISUAL_GRIP_ERR_SOFT_M = 0.007
VISUAL_GRIP_ERR_HARD_M = 0.026
VISUAL_GRIP_DROP_SOFT_M = 0.004
VISUAL_GRIP_DROP_HARD_M = 0.018
VISUAL_GRIP_RATE_SOFT_MPS = 0.006
VISUAL_GRIP_RATE_HARD_MPS = 0.035
VISUAL_GRIP_MIN_FORCE_BOOST = 4.0
VISUAL_GRIP_TARGET_FORCE_BOOST = 8.0
VISUAL_GRIP_MAX_FORCE_BOOST = 8.0
VISUAL_GRIP_POSITION_CORRECTION_GAIN = 0.040
VISUAL_GRIP_POSITION_CORRECTION_MAX = 0.0012
GRIP_HOLD_PRELOAD = 0.0180
GRIP_HOLD_CORRECTION = 0.00025
GRIP_HOLD_FAST_CORRECTION = 0.00070
GRIP_HOLD_RELAX_CORRECTION = 0.00004
GRIP_FORCE_DEADBAND = 1.1
GRIP_CLOSE_FRAMES = 300
GRIP_CONTACT_HOLD_FRAMES = 10
GRIP_CLOSE_TIMEOUT_FRAMES = GRIP_CLOSE_FRAMES + 100
GRIP_HOLD_FRAMES = 260
GRIP_TRANSPORT_HOLD_FRAMES = 6000
GRASP_LIFT_HEIGHT = 0.20
PLACE_ABOVE_HEIGHT = 0.24
GRASP_TARGET_FACE_BIAS = 0.000
GRASP_TARGET_APPROACH_BIAS = 0.000
CARRY_MIN_LIFT = 0.018
CARRY_MAX_ERR = 0.035
GRIP_LOCK_MAX_ERR = 0.024
GRIP_LOCK_MAX_OPEN_AXIS_ERR = CUBE_HALF_SIZE * 0.65
GRIP_LOCK_MAX_FACE_AXIS_ERR = CUBE_HALF_SIZE * 0.75
GRIP_LOCK_MAX_APPROACH_AXIS_ERR = CUBE_HALF_SIZE * 0.90
CONTACT_CONFIRM_MAX_DIST = 0.0008
MIN_STABLE_FINGER_CONTACTS = 2
TARGET_FINGER_CONTACTS = 4
MIN_CONTACT_DIVERSITY = 0.012
MAX_CONTACT_PAIR_SKEW = 0.034
MAX_CONTACT_CENTER_ERR = 0.026
INITIAL_GRASP_Z_OFFSET = 0.000
GRASP_Z_OFFSETS = [
    INITIAL_GRASP_Z_OFFSET,
    0.004,
    0.008,
    0.012,
    0.016,
]
PRE_CLOSE_MAX_XY_ERR = 0.010
PRE_CLOSE_MAX_Z_ERR = 0.018
PRE_CLOSE_MAX_OPEN_AXIS_ERR = CUBE_HALF_SIZE * 0.45
PRE_CLOSE_MAX_FACE_AXIS_ERR = CUBE_HALF_SIZE * 0.70
PRE_CLOSE_MAX_CENTER_ERR = 0.022
EARLY_CONTACT_MAX_PENETRATION = 0.0018
MAX_LOCAL_REPLAN = 2
PHYSICAL_EVAL_TOP_K = 24
LOCAL_PHYSICAL_EVAL_TOP_K = 6
REJECTED_GRASP_FRAME_PENALTY = 45.0
VISION_REPLAN_DELTA = 0.018
LOCAL_RGBD_KEEP_PLAN_DELTA = 0.008
MAX_PREGRASP_REPLANS = 1
PREGRASP_REAPPROACH_TOL = 0.020
GRIPPER_OPEN_TOL = 0.0045
GRIPPER_OPEN_WAIT_FRAMES = 90
APPROACH_CAPTURE_PROGRESS = 0.88
DESCEND_CLOSE_PROGRESS = 0.92
LOCAL_RETRACT_MIN_FRAMES = 80
LOCAL_REAPPROACH_MIN_FRAMES = 70
ARM_DONE_QPOS_TOL = 0.018
ARM_DONE_QVEL_TOL = 0.08
PRELIFT_LOST_GRACE_FRAMES = 18
CUBE_IDLE_FREEZE_SPEED = 0.010
CUBE_IDLE_FREEZE_ANG_SPEED = 0.050
CUBE_REST_Z_TOL = 0.006
BOX_VERIFY_FRAMES = 80

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# XML injection 鈥?target box
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
BOX_POS = np.array([0.42, -0.15, 0.06], dtype=np.float64)
BOX_SIZE = np.array([0.07, 0.07, 0.04], dtype=np.float64)  # half-extents
BOX_WALL = 0.005  # wall thickness


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
    physical fingertip meshes, with a thin invisible inner skin pad at the
    tactile sites so the cube is held by a real contact patch instead of one
    mesh contact point per side.
    """
    left_marker = '<geom type="mesh" rgba="0.592157 0.666667 0.682353 1" mesh="Link7" />'
    right_marker = '<geom type="mesh" rgba="0.592157 0.666667 0.682353 1" mesh="Link8" />'
    center_marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118" contype="0" conaffinity="0" group="1" density="0" rgba="0 1 0 1" />'
    hidden_center_marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118" contype="0" conaffinity="0" group="1" density="0" rgba="0 1 0 0" />'
    left_collision = (
        '<geom name="left_finger_collision" type="mesh" '
        'rgba="0.592157 0.666667 0.682353 1" mesh="Link7" '
        'condim="6" friction="18.0 8.0 3.0" '
        'solimp="0.90 0.98 0.001" solref="0.012 1" />\n'
        '                    <geom name="left_inner_pad_collision" type="box" '
        'pos="-0.005 -0.0005 0.002" size="0.017 0.020 0.003" '
        'rgba="0 0 0 0" condim="6" friction="24.0 10.0 4.0" '
        'solimp="0.95 0.99 0.0006" solref="0.010 1" />\n'
        '                    <site name="left_inner_skin_site" type="ellipsoid" '
        'pos="-0.005 -0.0005 0.002" size="0.017 0.020 0.004" '
        'rgba="0 0 0 0" group="5" />'
    )
    right_collision = (
        '<geom name="right_finger_collision" type="mesh" '
        'rgba="0.592157 0.666667 0.682353 1" mesh="Link8" '
        'condim="6" friction="18.0 8.0 3.0" '
        'solimp="0.90 0.98 0.001" solref="0.012 1" />\n'
        '                    <geom name="right_inner_pad_collision" type="box" '
        'pos="-0.005 0.0005 0.002" size="0.017 0.020 0.003" '
        'rgba="0 0 0 0" condim="6" friction="24.0 10.0 4.0" '
        'solimp="0.95 0.99 0.0006" solref="0.010 1" />\n'
        '                    <site name="right_inner_skin_site" type="ellipsoid" '
        'pos="-0.005 0.0005 0.002" size="0.017 0.020 0.004" '
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

        fovy = float(self._model.cam_fovy[self._camera_id])
        f_px = (img_h / 2.0) / math.tan(math.radians(fovy) / 2.0)

        depths = depths.astype(np.float64)
        x_cam = (xs.astype(np.float64) - img_w / 2.0) * depths / f_px
        y_cam = (img_h / 2.0 - ys.astype(np.float64)) * depths / f_px
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
        valid = np.isfinite(patch) & (patch > 1e-4) & (patch < MAX_RGBD_DEPTH_M)

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
            (samples > 1e-4) &
            (samples < MAX_RGBD_DEPTH_M)
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

    @property
    def done(self) -> bool:
        """True only after interpolation + settling have both finished."""
        return (
            self._done and
            self._settle_left <= 0 and
            self.physically_settled()
        )

    def near_done(self, fraction: float = 0.92) -> bool:
        """True when a new target can be blended in without a visible stop."""
        return self._done or self._progress >= float(np.clip(fraction, 0.0, 1.0))

    def physically_settled(self) -> bool:
        q_err = float(np.max(np.abs(self.current() - self._target)))
        qvels = []
        for jid in self._joints:
            dof = self._model.jnt_dofadr[jid]
            qvels.append(abs(float(self._data.qvel[dof])))
        max_qvel = max(qvels) if qvels else 0.0
        return q_err <= ARM_DONE_QPOS_TOL and max_qvel <= ARM_DONE_QVEL_TOL

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

    def step(self) -> None:
        if self._done:
            if self._settle_left > 0:
                self._settle_left -= 1
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
            if self._force_ema > max_force:
                self._relax_hold(GRIP_HOLD_CORRECTION * 4.0)
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
    ) -> None:
        if self._mode != "hold" or self._done:
            return
        left = max(0.0, float(reading.left_force))
        right = max(0.0, float(reading.right_force))
        low_side = min(left, right)
        avg_force = 0.5 * (left + right)
        if low_side < min_force:
            deficit = min_force - low_side
            amount = min(
                GRIP_HOLD_FAST_CORRECTION,
                GRIP_HOLD_CORRECTION * (1.0 + deficit / max(min_force, 1e-6)),
            )
            self._tighten_hold(amount)
        elif avg_force < target_force:
            self._tighten_hold(GRIP_HOLD_CORRECTION)
        elif avg_force > max_force and reading.balance < 0.45:
            self._relax_hold(GRIP_HOLD_RELAX_CORRECTION)
        if reading.balance > SKIN_BALANCE_MAX_RATIO and low_side < target_force:
            self._tighten_hold(GRIP_HOLD_CORRECTION)

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


def generate_grasp_orientations(
    ball_xyz: np.ndarray,
    opening_hints: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Generate only vertical top-down grasps for the first cube approach."""
    radial = _radial_xy(ball_xyz[:2])
    tangent = _tangent_xy(radial)
    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    base_hints: list[np.ndarray] = []
    base_hints.extend([
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        tangent, -tangent,
    ])
    if opening_hints:
        # The red square contour is strongly perspective-dependent in the
        # wrist camera.  Use it only to choose the nearest cube-side/world
        # axis; raw diagonal contour axes led to awkward diagonal jaw poses.
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


def grasp_stability_score(xmat: np.ndarray, ball_xyz: np.ndarray) -> float:
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
    return (
        80.0 * horizontal_tilt +
        20.0 * max(0.0, 0.999 - downward) +
        GRASP_WORLD_AXIS_WEIGHT * max(0.0, 0.98 - world_axis_alignment) +
        0.15 * radial_opening +
        0.10 * max(0.0, -tangent_opening_signed) +
        0.08 * max(0.0, 0.75 - tangent_opening)
    )


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
        z_axis = xmat[:, 2]
        gripper = aim - (SCAN_CAMERA_DISTANCE - CAMERA_TO_GRIPPER_CENTER) * z_axis
        key = tuple(np.round(np.concatenate((aim, gripper, xmat.reshape(-1))), 4))
        if key not in seen:
            seen.add(key)
            targets.append(ScanTarget(aim=aim, gripper=gripper, xmat=xmat))

    # Full-workspace polar pass.  These poses visibly move around the base,
    # and each camera ray points straight down at the ring sample.
    for xy in grid:
        aim = np.array([xy[0], xy[1], z_plane], dtype=np.float64)
        add_target(aim, make_tool_xmat(
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
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
                  friction="8.0 3.0 1.2" solimp="0.95 0.99 0.001"
                  solref="0.015 1"/>
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
        '<joint name="left_finger" pos="0 0 0" axis="0 0 1" type="slide" range="-0.025 0.007" actuatorfrcrange="-120 120" />',
    )
    xml_content = xml_content.replace(
        '<joint name="right_finger" pos="0 0 0" axis="0 0 -1" type="slide" range="0 0.025" actuatorfrcrange="-5 5" />',
        '<joint name="right_finger" pos="0 0 0" axis="0 0 -1" type="slide" range="-0.007 0.025" actuatorfrcrange="-120 120" />',
    )
    xml_content = xml_content.replace(
        '<position name="left_finger_act" joint="left_finger" kp="55" kv="7"\n              forcerange="-8 8"/>',
        '<position name="left_finger_act" joint="left_finger" kp="720" kv="160"\n              forcerange="-120 120"/>',
    )
    xml_content = xml_content.replace(
        '<position name="right_finger_act" joint="right_finger" kp="55" kv="7"\n              forcerange="-8 8"/>',
        '<position name="right_finger_act" joint="right_finger" kp="720" kv="160"\n              forcerange="-120 120"/>',
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
    workspace_pts = compute_workspace(model, data, arm_joints, left_id, right_id,
                                      resolution=0.08)

    # 鈹€鈹€ Controllers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    arm_ctrl = SmoothArmController(model, data, arm_joints, joint_to_actuator)
    gripper_ctrl = SmoothGripperController(model, data, gripper_limits, joint_to_actuator)

    # 鈹€鈹€ Cube detector + interactive placement 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rgb_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, RGB_CAMERA_NAME)
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    if ball_body_id < 0 or cube_geom_id < 0:
        raise RuntimeError("Could not find target_cube / cube_geom.")
    detector = BallDetector(model, rgb_cam_id, ball_body_id)

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
    skin_sensor = TactileSkinSensor(model)
    skin_display = TactileSkinDisplay(width=500, height=260, history_len=180)

    # 鈹€鈹€ Eye-in-hand camera 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rgbd_window = None
    if rgb_cam_id >= 0:
        try:
            rgbd_window = RGBDCameraWindow(
                model, rgb_cam_id, width=480, height=360,
                render_every_n=CAMERA_EVERY_N,
                window_name="Eye-in-Hand RGB-D Camera")
            print("Eye-in-hand RGB-D camera: active (480x360 color + depth)")
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
    detected_ball_pos: np.ndarray | None = None
    detected_opening_hints: list[np.ndarray] = []
    dynamic_ik_plan: dict[str, IKResult] = {}
    dynamic_grasp_xmat: np.ndarray | None = None
    dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
    rejected_grasp_frames: list[tuple[np.ndarray, np.ndarray]] = []
    rgbd_scan_estimates: list[np.ndarray] = []
    rgbd_scan_hints: list[np.ndarray] = []
    last_scan_rgbd_sample_frame = -SCAN_RGBD_CAPTURE_EVERY_N
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
    regrasp_count = 0
    MAX_REGRASP = 4
    pregrasp_replan_count = 0
    local_replan_count = 0
    grip_contact_hold = 0
    grip_close_frames = 0
    gripper_open_wait_frames = 0
    prelift_lost_frames = 0
    ball_z_before_lift = 0.0
    status_msg = "IDLE 鈥?use cube_x/y/z & start_demo sliders"
    box_verify_left = 0
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
        cube_static_anchor_active = False
        cube_static_pos = current_ball_xyz()
        cube_static_quat = current_cube_quat()

    def release_cube_static_anchor(context: str) -> None:
        nonlocal cube_static_anchor_active
        cube_static_anchor_active = False

    def apply_cube_static_anchor() -> None:
        return
        if (not cube_static_anchor_active or grip_locked or
                ball_qpos_adr < 0 or phase == -1):
            return
        if phase in (0, 1, 2, 3) and cube_touching_robot():
            # MuJoCo can report very light positive-margin contacts before
            # visible touch.  Keep the cube fixed through those numerical
            # near-contacts; only a real collision/penetration releases it.
            bad_contact, bad_name = cube_nonfinger_robot_contact()
            min_dist = cube_robot_min_contact_dist()
            if bad_contact or min_dist < -EARLY_CONTACT_MAX_PENETRATION:
                reason = "robot touched cube before gripper close"
                if bad_contact:
                    reason += f": {bad_name}"
                release_cube_static_anchor(reason)
                return
        if phase == 4:
            # Keep the cube perfectly still until a real two-sided grasp is
            # confirmed. Releasing on the first one-sided touch lets a finger
            # nudge the cube before the opposing finger has any support, which
            # shows up as pre-grasp jitter and weak, off-center grasps.
            left_finger_hit, right_finger_hit, pad_contacts = pad_cube_contact_sides()
            patch_span = finger_contact_patch_span()
            bad_contact, bad_name = cube_nonfinger_robot_contact()
            if bad_contact:
                data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube_static_pos
                data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = cube_static_quat
                zero_cube_velocity()
                mujoco.mj_forward(model, data)
                return
            min_dist = cube_robot_min_contact_dist()
            skin = read_skin()
            pair_skew, contact_center_err = finger_contact_alignment()
            confirmed_squeeze = (
                left_finger_hit and right_finger_hit and
                contact_patch_is_large(pad_contacts, patch_span) and
                pair_skew <= MAX_CONTACT_PAIR_SKEW and
                contact_center_err <= MAX_CONTACT_CENTER_ERR and
                min_dist <= CONTACT_CONFIRM_MAX_DIST and
                skin_squeeze_ok(skin) and
                grip_contact_hold >= GRIP_CONTACT_HOLD_FRAMES
            )
            if not confirmed_squeeze:
                data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube_static_pos
                data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = cube_static_quat
                zero_cube_velocity()
                mujoco.mj_forward(model, data)
            else:
                release_cube_static_anchor("confirmed two-sided squeeze")
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

    def read_skin() -> TactileReading:
        raw = skin_sensor.read(data)
        left_contact, right_contact = tactile_contact_forces()
        return TactileReading(
            left_force=max(raw.left_force, left_contact),
            right_force=max(raw.right_force, right_contact),
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
        # Real-physics mode: never kinematically attach the cube to the gripper.
        # The cube may only move through MuJoCo contact and friction.
        return

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
        if lin_v <= CUBE_IDLE_FREEZE_SPEED and ang_v <= CUBE_IDLE_FREEZE_ANG_SPEED:
            if abs(z - CUBE_REST_Z) <= CUBE_REST_Z_TOL:
                data.qpos[ball_qpos_adr + 2] = CUBE_REST_Z
            zero_cube_velocity()
            mujoco.mj_forward(model, data)

    def capture_cube_from_rgbd(label: str, *, draw: bool = True,
                               log_miss: bool = True) -> np.ndarray | None:
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
            in_workspace = (
                workspace_contains_xy(est_xyz[:2]) and
                VISION_Z_MIN <= float(est_xyz[2]) <= VISION_Z_MAX
            )
            if not in_workspace:
                if log_miss:
                    print(f"    Vision {label}: rejected estimate {np.round(est_xyz, 3)}")
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
            print(f"    Vision {label}: uv={center_uv} r={radius}px "
                  f"xyz={np.round(est_xyz, 3)}{spread_msg}{hint_msg}")
            return est_xyz
        except Exception as exc:
            print(f"    Vision {label}: ERROR {exc}")
            import traceback; traceback.print_exc()
            return None

    def reset_visual_grip_feedback(context: str) -> None:
        nonlocal visual_grip_reference_offset, visual_grip_filtered_xyz
        nonlocal visual_grip_last_time, visual_grip_slip_m
        nonlocal visual_grip_slip_rate_mps, visual_grip_drop_m
        nonlocal visual_grip_weight, visual_grip_miss_count
        nonlocal last_visual_grip_frame, last_visual_grip_log_frame
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
        min_force = SKIN_HOLD_MIN_FORCE + visual_grip_weight * VISUAL_GRIP_MIN_FORCE_BOOST
        target_force = (
            SKIN_HOLD_TARGET_FORCE +
            visual_grip_weight * VISUAL_GRIP_TARGET_FORCE_BOOST
        )
        max_force = SKIN_HOLD_MAX_FORCE + visual_grip_weight * VISUAL_GRIP_MAX_FORCE_BOOST
        gripper_ctrl.maintain_tactile_force(
            skin,
            min_force=min_force,
            target_force=target_force,
            max_force=max_force,
        )
        gripper_ctrl.compensate_visual_slip(
            visual_grip_weight,
            visual_grip_slip_m,
            visual_grip_drop_m,
        )
        return skin

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
        required = {"approach", "grasp", "lift", "carry_mid", "place_above", "place_drop"}
        return (
            set(plan) == required and
            all(plan[name].success for name in required)
        )

    def cube_inside_box() -> bool:
        cube = current_ball_xyz()
        x_ok = abs(float(cube[0] - BOX_POS[0])) <= (BOX_SIZE[0] - CUBE_HALF_SIZE * 0.25)
        y_ok = abs(float(cube[1] - BOX_POS[1])) <= (BOX_SIZE[1] - CUBE_HALF_SIZE * 0.25)
        z_ok = CUBE_REST_Z - 0.004 <= float(cube[2]) <= BOX_POS[2] + BOX_SIZE[2] + 0.12
        return bool(x_ok and y_ok and z_ok)

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
            x_ok = abs(float(cube[0] - BOX_POS[0])) <= (
                BOX_SIZE[0] - CUBE_HALF_SIZE * 0.25)
            y_ok = abs(float(cube[1] - BOX_POS[1])) <= (
                BOX_SIZE[1] - CUBE_HALF_SIZE * 0.25)
            z_ok = CUBE_REST_Z - 0.004 <= float(cube[2]) <= (
                BOX_POS[2] + BOX_SIZE[2] + 0.12)
            return bool(x_ok and y_ok and z_ok)

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
                           speed=SPEED_LIFT, min_frames=560)
        for frame_idx in range(720):
            arm_sim.step()
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > 520:
                break

        cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        err = float(np.linalg.norm(cube - scratch_gripper_object_center(sim, xmat)))
        lifted = float(cube[2] - z_before)
        left_hit, right_hit, contacts = scratch_finger_contact_sides(sim)
        lift_patch_span = scratch_finger_contact_patch_span(sim, xmat)
        lift_pair_skew, lift_contact_center_err = scratch_finger_contact_alignment(sim, xmat)
        lift_min_dist = scratch_cube_robot_min_contact_dist(sim)
        center_err, open_err, face_err, approach_err = scratch_grip_metrics()
        if (not left_hit or not right_hit or
                not contact_patch_is_large(contacts, lift_patch_span) or
                lift_pair_skew > MAX_CONTACT_PAIR_SKEW or
                lift_contact_center_err > MAX_CONTACT_CENTER_ERR or
                scratch_nonfinger_robot_contact(sim) or
                lift_min_dist > CONTACT_CONFIRM_MAX_DIST or
                lifted < 0.10 or err > min(CARRY_MAX_ERR, 0.028)):
            return (
                60.0 +
                max(0.0, 0.10 - lifted) * 120.0 +
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
                           speed=SPEED_CARRY, min_frames=520)
        for frame_idx in range(700):
            arm_sim.step()
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > 520:
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
                           speed=SPEED_CARRY, min_frames=560)
        for frame_idx in range(760):
            arm_sim.step()
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > 560:
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

        contact_bonus_penalty = max(
            0, TARGET_FINGER_CONTACTS - min(contact_count, contacts, mid_contacts, carry_contacts)
        ) * 0.6
        patch_bonus_penalty = max(
            0.0,
            MIN_CONTACT_DIVERSITY * 1.5 -
            min(close_patch_span, lift_patch_span, mid_patch_span, carry_patch_span),
        ) * 120.0
        return (
            err * 2.0 + open_err * 4.0 + face_err * 2.0 +
            contact_bonus_penalty + patch_bonus_penalty
        )

    def compute_dynamic_ik(
        ball_xyz: np.ndarray,
        grasp_z_offs: float | None = None,
        opening_hints: list[np.ndarray] | None = None,
        preview_top_k: int = PHYSICAL_EVAL_TOP_K,
    ) -> dict[str, IKResult]:
        """Compute a pose-aware IK plan for the detected cube position."""
        nonlocal dynamic_grasp_xmat, dynamic_grasp_z_offset
        dynamic_grasp_xmat = None
        ball_xyz = np.asarray(ball_xyz, dtype=np.float64).copy()
        ball_xyz[2] = float(np.clip(ball_xyz[2], VISION_Z_MIN, VISION_Z_MAX))
        z_offset_candidates = (
            list(GRASP_Z_OFFSETS) if grasp_z_offs is None else [float(grasp_z_offs)]
        )
        carry_mid_contact_target = np.array([
            0.5 * (ball_xyz[0] + BOX_POS[0]),
            0.5 * (ball_xyz[1] + BOX_POS[1]),
            BOX_POS[2] + PLACE_ABOVE_HEIGHT,
        ], dtype=np.float64)
        place_above_contact_target = BOX_POS + np.array([0.0, 0.0, PLACE_ABOVE_HEIGHT])
        place_drop_contact_target = np.array([
            BOX_POS[0], BOX_POS[1],
            BOX_POS[2] + BOX_SIZE[2] + CUBE_HALF_SIZE * 0.35,
        ], dtype=np.float64)

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
                    if name in ("approach", "grasp"):
                        pos_tol = 0.004
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
                        score += 0.6
                    if not result.success:
                        ok = False
                        score += 20.0
                        break
                    set_joint_positions(model, scratch, arm_joints, result.angles)
                    mujoco.mj_forward(model, scratch)
                    rest = result.angles

                if ok:
                    score += 0.02 * float(np.linalg.norm(plan["grasp"].angles - home_rest))
                    score += grasp_stability_score(xmat, ball_xyz)
                    score += grasp_frame_penalty(xmat)
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
                penalty = physical_plan_penalty(plan, ball_xyz, xmat)
                total = base_score + penalty
                y_axis = xmat[:, 1]
                z_axis = xmat[:, 2]
                print(f"    Physical preview {idx}: total={total:.2f} "
                      f"penalty={penalty:.2f} "
                      f"open={np.round(y_axis[:2], 2)} "
                      f"approach={np.round(z_axis, 2)} "
                      f"z_off={current_z_offs*1000:.0f}mm")
                if total < preview_best_score:
                    preview_best_score = total
                    preview_best_plan = plan
                    preview_best_xmat = xmat.copy()
                    best_z_offs = float(current_z_offs)
            if preview_best_plan is not None:
                best_plan = preview_best_plan
                best_xmat = preview_best_xmat

        if (best_plan is None or
            set(best_plan) != {"approach", "grasp", "lift", "carry_mid", "place_above", "place_drop"} or
            any(not result.success for result in best_plan.values())):
            print("    Pose IK incomplete; falling back to position-only IK.")
            best_plan = {}
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = data.qpos[:]
            scratch.qvel[:] = 0.0
            mujoco.mj_forward(model, scratch)
            fallback_z_offs = float(z_offset_candidates[0])
            grasp_contact_target = ball_xyz + np.array(
                [0.0, 0.0, fallback_z_offs], dtype=np.float64)
            fallback_xmat = make_tool_xmat(
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                (_nearest_world_opening_axis(opening_hints[0])
                 if opening_hints else _tangent_xy(_radial_xy(ball_xyz[:2]))),
            )
            fallback_grasp = gripper_body_target_for_contact(
                grasp_contact_target, fallback_xmat)
            fallback_lift = gripper_body_target_for_contact(
                grasp_contact_target + np.array([0.0, 0.0, GRASP_LIFT_HEIGHT]),
                fallback_xmat,
            )
            fallback_place_above = gripper_body_target_for_contact(
                place_above_contact_target, fallback_xmat)
            fallback_carry_mid = gripper_body_target_for_contact(
                carry_mid_contact_target, fallback_xmat)
            fallback_place_drop = gripper_body_target_for_contact(
                place_drop_contact_target, fallback_xmat)
            fallback_targets = {
                "approach": fallback_grasp - 0.13 * fallback_xmat[:, 2],
                "grasp": fallback_grasp,
                "lift": fallback_lift,
                "carry_mid": fallback_carry_mid,
                "place_above": fallback_place_above,
                "place_drop": fallback_place_drop,
            }
            for name, target in fallback_targets.items():
                result = solve_gripper_center_ik(
                    model, scratch, target, pinch_body_id, pinch_body_id, arm_joints,
                    max_iter=700, tol=0.006,
                )
                best_plan[name] = result
                if result.success:
                    set_joint_positions(model, scratch, arm_joints, result.angles)
                    mujoco.mj_forward(model, scratch)
            best_xmat = fallback_xmat.copy()
            best_z_offs = fallback_z_offs

        if best_xmat is not None:
            dynamic_grasp_xmat = best_xmat.copy()
            dynamic_grasp_z_offset = float(best_z_offs)
            print(f"    Selected grasp frame: "
                  f"open={np.round(best_xmat[:, 1], 2)} "
                  f"approach={np.round(best_xmat[:, 2], 2)} "
                  f"grasp_z_offset={dynamic_grasp_z_offset*1000:.0f}mm")
        for name in ("approach", "grasp", "lift", "carry_mid", "place_above", "place_drop"):
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

    def cube_is_secured() -> bool:
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        patch_span = finger_contact_patch_span()
        pair_skew, contact_center_err = finger_contact_alignment()
        visual_ok = (
            visual_grip_reference_offset is None or
            visual_grip_miss_count > 2 or
            (
                visual_grip_slip_m <= VISUAL_GRIP_ERR_HARD_M and
                visual_grip_drop_m <= VISUAL_GRIP_DROP_HARD_M
            )
        )
        return (
            lifted > CARRY_MIN_LIFT and
            err < CARRY_MAX_ERR and
            left_pad_hit and right_pad_hit and
            contact_patch_is_large(pad_contacts, patch_span) and
            pair_skew <= MAX_CONTACT_PAIR_SKEW and
            contact_center_err <= MAX_CONTACT_CENTER_ERR and
            visual_ok
        )

    def pre_close_alignment_ok(label: str) -> bool:
        cube = current_ball_xyz()
        object_center = gripper_object_center()
        xy_err = float(np.linalg.norm((object_center - cube)[:2]))
        z_err = abs(float(object_center[2] - cube[2]))
        if dynamic_grasp_xmat is not None:
            opening_axis = _unit(
                dynamic_grasp_xmat[:, 1],
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
            )
            face_axis = _unit(
                dynamic_grasp_xmat[:, 0],
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
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
              f"center={center_err*100:.1f}cm")
        if bad_contact:
            print(f"    {label}: rejected, cube is touching non-finger robot part "
                  f"{bad_name}")
            return False
        return (
            xy_err <= PRE_CLOSE_MAX_XY_ERR and
            z_err <= PRE_CLOSE_MAX_Z_ERR and
            open_axis_err <= PRE_CLOSE_MAX_OPEN_AXIS_ERR and
            face_axis_err <= PRE_CLOSE_MAX_FACE_AXIS_ERR and
            center_err <= PRE_CLOSE_MAX_CENTER_ERR
        )

    def restart_search(reason: str) -> None:
        nonlocal phase, sub, scan_idx, scan_targets, detected_ball_pos
        nonlocal detected_opening_hints, dynamic_ik_plan, regrasp_count, scan_round, status_msg
        nonlocal pregrasp_replan_count, local_replan_count, dynamic_grasp_xmat
        nonlocal dynamic_grasp_z_offset
        nonlocal rgbd_scan_estimates, rgbd_scan_hints
        print(f"    Restarting search: {reason}")
        release_grip_lock("restart search")
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
        regrasp_count = 0
        pregrasp_replan_count = 0
        local_replan_count = 0
        scan_round = 0
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
            pregrasp_replan_count = 0
            dynamic_ik_plan = compute_dynamic_ik(
                detected_ball_pos,
                opening_hints=detected_opening_hints,
            )
            if plan_is_ready(dynamic_ik_plan):
                status_msg = (f"Cube found by stable RGB-D! "
                              f"X={stable_xyz[0]:.3f} "
                              f"Y={stable_xyz[1]:.3f} "
                              f"Z={stable_xyz[2]:.3f}")
                arm_ctrl.set_target(arm_ctrl.current(), speed=SPEED_SCAN, min_frames=1)
                phase = 2
                sub = 0
                return True
            print("    Stable RGB-D estimate rejected: no complete IK plan.")
            diagnose_failure("stable RGB-D estimate without reachable plan")
            detected_ball_pos = None
            dynamic_ik_plan = {}
        elif len(rgbd_scan_estimates) >= GLOBAL_RGBD_MAX_VIEWS:
            print("    RGB-D estimates are not stable enough yet; "
                  "discarding oldest views and continuing scan.")
        return False

    def demo_tick() -> None:
        nonlocal phase, sub, finished, scan_idx, scan_targets, regrasp_count
        nonlocal detected_ball_pos, detected_opening_hints
        nonlocal dynamic_ik_plan, status_msg, ball_z_before_lift
        nonlocal scan_round, box_verify_left, pregrasp_replan_count
        nonlocal grip_contact_hold, grip_close_frames, gripper_open_wait_frames, local_replan_count
        nonlocal prelift_lost_frames
        nonlocal dynamic_grasp_xmat, dynamic_grasp_z_offset, rejected_grasp_frames
        nonlocal rgbd_scan_estimates, rgbd_scan_hints
        nonlocal last_scan_rgbd_sample_frame

        if finished:
            return

        # 鈹€鈹€ Phase -1:  idle 鈥?wait for SPACE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if phase == -1:
            status_msg = "IDLE 鈥?use cube_x/y/z & start_demo sliders"
            return  # do nothing, wait for keyboard

        # 鈹€鈹€ Phase 0:  scanning 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 0:
            if sub == 0:
                if rgbd_window is None:
                    print("    ERROR: RGB-D camera is not available; refusing to use a fixed cube target.")
                    status_msg = "ERROR: RGB-D camera unavailable"
                    finished = True
                    return
                print(">>> Phase 0 : Camera scanning for cube ...")
                scan_targets = generate_scan_targets(
                    _last_bz, hint_xy=np.array([_last_bx, _last_by], dtype=np.float64)
                )
                rgbd_scan_estimates = []
                rgbd_scan_hints = []
                last_scan_rgbd_sample_frame = -SCAN_RGBD_CAPTURE_EVERY_N
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
                    if frame - last_scan_rgbd_sample_frame >= SCAN_RGBD_CAPTURE_EVERY_N:
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
                    diagnose_failure("local RGB-D reacquire failed")
                    begin_local_rgbd_replan(
                        "local RGB-D reacquire failed after retreat")
                    return
                detected_ball_pos = fresh.copy()
                previous_xmat = (
                    None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                )
                replanned = compute_dynamic_ik(
                    detected_ball_pos,
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
                        print("    Pre-grasp RGB-D lost cube, likely due to close-range "
                              "occlusion; continuing with the last RGB-D estimate.")
                        pregrasp_replan_count = 0
                        phase = 3; sub = 0
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

                if delta <= LOCAL_RGBD_KEEP_PLAN_DELTA and plan_is_ready(dynamic_ik_plan):
                    print(f"    Pre-grasp RGB-D stable (delta={delta*100:.1f}cm); "
                          "using the existing physical-preview plan.")
                    pregrasp_replan_count = 0
                    phase = 3
                    sub = 0
                    return

                detected_ball_pos = fresh.copy()
                previous_xmat = (
                    None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                )
                replanned = compute_dynamic_ik(
                    detected_ball_pos,
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
                    previous_xmat = (
                        None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                    )
                    replanned = compute_dynamic_ik(
                        detected_ball_pos,
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
                arm_ctrl.set_target(dynamic_ik_plan["grasp"].angles, speed=SPEED_SLOW)
                sub = 1
                status_msg = f"Descending (retry {regrasp_count}) ..."
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
                    release_cube_static_anchor("early contact during descent")
                    begin_local_rgbd_replan("early cube contact during descent")
                    return
            if arm_ctrl.near_done(DESCEND_CLOSE_PROGRESS):
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
                    print(f"    Pre-close fingertip contact accepted "
                          f"(min_dist={min_dist*1000:.2f} mm); closing now.")
                local_replan_count = 0
                ball_z_before_lift = float(current_ball_xyz()[2])
                phase = 4; sub = 0

        # 鈹€鈹€ Phase 4:  close gripper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 4:
            if sub == 0:
                gc = gripper_object_center()
                ball = current_ball_xyz()
                err = np.linalg.norm(ball - gc)
                print(f"    Grasp center->cube: {err*100:.1f}cm  "
                      f"(grasp Z={gc[2]:.3f}, cube Z={ball[2]:.3f})")
                grip_contact_hold = 0
                grip_close_frames = 0
                prelift_lost_frames = 0
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
                if not (left_pad_hit and right_pad_hit and
                        contact_patch_is_large(pad_contacts, patch_span) and
                        pair_skew <= MAX_CONTACT_PAIR_SKEW and
                        contact_center_err <= MAX_CONTACT_CENTER_ERR and
                        min_dist <= CONTACT_CONFIRM_MAX_DIST and
                        skin_squeeze_ok(skin, SKIN_TOUCH_FORCE) and
                        bool(geom["ok"])):
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
                if grip_close_frames % 40 == 0:
                    print(f"    Pre-lift grip hold: contacts={contacts} "
                          f"finger_contacts={pad_contacts} "
                          f"patch_span={patch_span*1000:.1f}mm "
                          f"pair_skew={pair_skew*1000:.1f}mm "
                          f"contact_center_err={contact_center_err*1000:.1f}mm "
                          f"contact_dist={min_dist*1000:.2f}mm "
                          f"skin=({skin.left_force:.2f},{skin.right_force:.2f})N "
                          f"bal={skin.balance:.2f} "
                          f"center={geom['center_err']*100:.1f}cm")
                grip_close_frames += 1
                if gripper_ctrl.done:
                    phase = 5; sub = 0
                    status_msg = "Grip stable; lifting ..."

        # 鈹€鈹€ Phase 5:  lift + verify 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 5:
            if sub == 0:
                print(">>> Phase 5 : Lift & verify")
                gripper_ctrl.hold(duration_frames=GRIP_TRANSPORT_HOLD_FRAMES)
                arm_ctrl.set_target(dynamic_ik_plan["lift"].angles,
                                    speed=SPEED_LIFT, min_frames=560)
                sub = 1
                status_msg = "Lifting ..."
            if sub == 1 and gripper_ctrl.holding:
                maintain_fused_grip_force("lift")
            if arm_ctrl.done:
                # 鈹€鈹€ Verify: did the cube actually rise? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
                ball_z_now = float(current_ball_xyz()[2])
                lifted = ball_z_now - ball_z_before_lift
                print(f"    Cube Z: before={ball_z_before_lift:.3f}  "
                      f"after={ball_z_now:.3f}  螖={lifted*100:.1f}cm")
                carry_err, carry_lift = cube_carry_metrics()
                print(f"    Carry check: err={carry_err*100:.1f}cm  "
                      f"lift={carry_lift*100:.1f}cm")
                if cube_is_secured():
                    print(f"    Grasp SUCCESS after {regrasp_count} retries")
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
            if sub in (1, 2, 3) and gripper_ctrl.holding:
                maintain_fused_grip_force(f"carry{sub}")
            if sub == 0:
                print(">>> Phase 6 : Carry via midpoint")
                arm_ctrl.set_target(dynamic_ik_plan["carry_mid"].angles,
                                    speed=SPEED_CARRY, min_frames=520)
                sub = 1
                status_msg = "Carrying cube ..."
            elif sub == 1 and arm_ctrl.done:
                if not cube_is_secured():
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube already reached box during midpoint carry: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    diagnose_failure("cube dropped before carry midpoint")
                    restart_search("cube dropped before carry midpoint")
                    return
                print(">>> Phase 6a : Move above box")
                arm_ctrl.set_target(dynamic_ik_plan["place_above"].angles,
                                    speed=SPEED_CARRY, min_frames=560)
                sub = 2
                status_msg = "Moving above box ..."
            elif sub == 2 and arm_ctrl.done:
                if not cube_is_secured():
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube already reached box during carry: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    diagnose_failure("cube dropped while carrying")
                    restart_search("cube dropped while carrying")
                    return
                print(">>> Phase 6b : Lower into box")
                arm_ctrl.set_target(dynamic_ik_plan["place_drop"].angles,
                                    speed=SPEED_PLACE, min_frames=340)
                sub = 3
                status_msg = "Lowering into box ..."
            elif sub == 3 and arm_ctrl.done:
                if not cube_is_secured():
                    if cube_inside_box():
                        cube = current_ball_xyz()
                        print(f"    Cube already settled in box during lowering: "
                              f"{np.round(cube, 3)}")
                        phase = 7; sub = 0
                        return
                    diagnose_failure("cube dropped before release")
                    restart_search("cube dropped before release")
                    return
                phase = 7; sub = 0
        elif phase == 7:
            if sub == 0:
                print(">>> Phase 7 : Release into box")
                release_cube_static_anchor("release in box")
                release_grip_lock("release in box")
                dynamic_grasp_xmat = None
                dynamic_grasp_z_offset = INITIAL_GRASP_Z_OFFSET
                gripper_ctrl.open(duration_frames=25)
                sub = 1
                status_msg = "Releasing ..."
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
                release_grip_lock("re-grasp")
                gripper_ctrl.open(duration_frames=20)
                sub = 1
            elif sub == 1:
                if not gripper_ctrl.done or not arm_ctrl.done:
                    return
                # Wait for cube to settle (velocity below threshold)
                bv = float(np.linalg.norm(data.qvel[ball_dof_adr:ball_dof_adr+3]))
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
                    z_offs = GRASP_Z_OFFSETS[
                        min(regrasp_count, len(GRASP_Z_OFFSETS) - 1)]
                    previous_xmat = (
                        None if dynamic_grasp_xmat is None else dynamic_grasp_xmat.copy()
                    )
                    replanned = compute_dynamic_ik(
                        detected_ball_pos,
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

        workspace_drawn = False
        frame = 0
        last_skin_reading = read_skin()

        # Warmup 鈥?settle physics
        for _ in range(30):
            for _ in range(PHYSICS_SUBSTEPS):
                mujoco.mj_step(model, data)
            frame += 1

        while viewer.is_running():
            frame_start = time.perf_counter()

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
            if not workspace_drawn and workspace_pts:
                try:
                    render_workspace_spheres(viewer, workspace_pts)
                    workspace_drawn = True
                except Exception:
                    pass

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
