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
  - Real-time 6-axis F/T display, eye-in-hand camera, joint-state panel
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

from common.model_loader import load_and_inject, RGB_CAMERA_NAME, BALL_POS
from common.ik_solver import (
    solve_gripper_center_ik, set_joint_positions, IKResult,
)
from common.motion import build_gripper_limits, command_gripper
from common.force_sensor import ForceTorqueSensor, FTDisplay
from common.camera import RGBCameraWindow

# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Constants
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
TARGET_DISPLAY_HZ = 50
PHYSICS_SUBSTEPS = 5
FRAME_DT = 1.0 / TARGET_DISPLAY_HZ
START_TRIGGER_THRESHOLD = 0.95

FT_EVERY_N = 2         # 25 Hz
JOINT_EVERY_N = 4      # 12.5 Hz
CAMERA_EVERY_N = 2     # 25 Hz

SPEED_NORMAL = 1.4
SPEED_SLOW = 0.55  # very slow for precise grasp positioning
SPEED_SCAN = 1.0
SPEED_LIFT = 0.35
SPEED_CARRY = 0.35
SPEED_PLACE = 0.45

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

MIN_BALL_AREA_PX = 12       # minimum contour area (small cube at distance 鈮?15-30 px)
CUBE_HALF_SIZE = 0.020  # 40 mm cube side 鈥?good grip depth in 50 mm gripper
PAD_HALF_HEIGHT = 0.003
CUBE_REST_Z = CUBE_HALF_SIZE + PAD_HALF_HEIGHT + 0.002
CAMERA_TO_GRIPPER_CENTER = 0.046
SCAN_CAMERA_DISTANCE = 0.24
GRIP_CONTACT_FORCE = 5.5
GRIP_LOCK_MIN_FORCE = 0.4
GRIP_MAX_FORCE = 60.0
GRIP_SQUEEZE_MARGIN = 0.0040
GRIP_CLOSE_FRAMES = 180
GRIP_CONTACT_HOLD_FRAMES = 12
GRIP_CLOSE_TIMEOUT_FRAMES = GRIP_CLOSE_FRAMES + 100
GRIP_HOLD_FRAMES = 160
CARRY_MIN_LIFT = 0.018
CARRY_MAX_ERR = 0.035
GRIP_LOCK_MAX_ERR = 0.024
GRIP_LOCK_MAX_OPEN_AXIS_ERR = CUBE_HALF_SIZE * 0.65
GRIP_LOCK_MAX_FACE_AXIS_ERR = CUBE_HALF_SIZE * 0.85
PRE_CLOSE_MAX_XY_ERR = 0.012
PRE_CLOSE_MAX_Z_ERR = 0.026
PRE_CLOSE_MAX_OPEN_AXIS_ERR = CUBE_HALF_SIZE * 0.55
PRE_CLOSE_MAX_CENTER_ERR = 0.030
EARLY_CONTACT_MAX_PENETRATION = 0.0015
MAX_LOCAL_REPLAN = 2
PHYSICAL_EVAL_TOP_K = 3
REJECTED_GRASP_FRAME_PENALTY = 45.0
VISION_REPLAN_DELTA = 0.018
RGB_ANCHOR_ACCEPT_TOL = 0.055
MAX_PREGRASP_REPLANS = 1
PREGRASP_REAPPROACH_TOL = 0.020
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

    The detailed STL meshes are kept visible.  Transparent inner box geoms are
    attached to the same finger bodies to give MuJoCo stable pad-like contacts;
    they do not kinematically attach or move the cube.
    """
    left_marker = '<geom type="mesh" rgba="0.592157 0.666667 0.682353 1" mesh="Link7" />'
    right_marker = '<geom type="mesh" rgba="0.592157 0.666667 0.682353 1" mesh="Link8" />'
    center_marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118" contype="0" conaffinity="0" group="1" density="0" rgba="0 1 0 1" />'
    hidden_center_marker = '<geom size="0.005" pos="-0.0002 -0.0003 0.13118" contype="0" conaffinity="0" group="1" density="0" rgba="0 1 0 0" />'
    left_collision = (
        '<geom name="left_finger_collision" type="mesh" '
        'rgba="0.592157 0.666667 0.682353 1" mesh="Link7" '
        'condim="6" friction="8.0 3.0 1.0" '
        'solimp="0.95 0.99 0.001" solref="0.006 1" />\n'
        '                    <geom name="left_finger_inner_pad_collision" type="box" '
        'pos="0 -0.022 -0.002" size="0.018 0.022 0.003" '
        'rgba="0 0 0 0" condim="6" friction="10.0 4.0 1.2" '
        'solimp="0.97 0.995 0.0005" solref="0.004 1" />'
    )
    right_collision = (
        '<geom name="right_finger_collision" type="mesh" '
        'rgba="0.592157 0.666667 0.682353 1" mesh="Link8" '
        'condim="6" friction="8.0 3.0 1.0" '
        'solimp="0.95 0.99 0.001" solref="0.006 1" />\n'
        '                    <geom name="right_finger_inner_pad_collision" type="box" '
        'pos="0 0.022 -0.002" size="0.018 0.022 0.003" '
        'rgba="0 0 0 0" condim="6" friction="10.0 4.0 1.2" '
        'solimp="0.97 0.995 0.0005" solref="0.004 1" />'
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
    xml_content = xml_content.replace(
        '<site name="wrist_ft_site" pos="0 0 0.131" size="0.005" rgba="1 1 0 0.3"/>',
        '<site name="wrist_ft_site" pos="0 0 0.131" size="0.001" rgba="1 1 0 0"/>',
        1,
    )
    return xml_content


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Cube detector 鈥?camera-based red-cube localisation
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
class BallDetector:
    """Find the red cube in an RGB image and estimate its 3-D position."""

    def __init__(self, model, camera_id: int, ball_body_id: int):
        self._model = model
        self._camera_id = camera_id
        self._ball_body_id = ball_body_id
        self._last_detection_uv: tuple[int, int] | None = None
        self._last_estimated_xyz: np.ndarray | None = None
        self._last_axis_uv: np.ndarray | None = None
        self._last_axis_confidence = 0.0

    def detect(self, rgb: np.ndarray, data) -> tuple[tuple[int, int] | None, int]:
        """Return ((cx, cy), radius_px) or (None, 0) if not found."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
        mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
        mask = mask1 | mask2
        # Morphological clean-up: remove small noise, merge nearby blobs
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < MIN_BALL_AREA_PX:
            return None, 0
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        self._last_detection_uv = (int(cx), int(cy))
        pts = largest.reshape(-1, 2).astype(np.float64)
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

    def estimate_3d(self, center_uv: tuple[int, int], data,
                    img_w: int, img_h: int,
                    target_z: float | None = None) -> np.ndarray | None:
        """Ray-plane intersection: camera ray 脳 horizontal plane at given Z.

        Returns None when the ray is nearly parallel to the plane or the
        intersection lies behind the camera 鈥?the caller must try another
        viewpoint in that case.
        """
        cam_pos = data.cam_xpos[self._camera_id].copy()
        cam_mat = data.cam_xmat[self._camera_id].reshape(3, 3)

        fovy = float(self._model.cam_fovy[self._camera_id])
        f_px = (img_h / 2.0) / math.tan(math.radians(fovy) / 2.0)

        px = center_uv[0] - img_w / 2.0
        py = img_h / 2.0 - center_uv[1]
        # MuJoCo cameras look along local -Z; image Y points downward.
        ray_cam = np.array([px / f_px, py / f_px, -1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)
        ray_world = cam_mat @ ray_cam

        z_plane = target_z if target_z is not None else CUBE_REST_Z
        if abs(ray_world[2]) < 1e-9:
            return None
        t = (z_plane - cam_pos[2]) / ray_world[2]
        if t <= 0:
            return None
        est = cam_pos + t * ray_world
        est[2] = z_plane
        return est

    @property
    def last_estimated_xyz(self) -> np.ndarray | None:
        return self._last_estimated_xyz

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

    SETTLE_FRAMES = 30  # frames 鈥?higher kp means faster convergence

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
        return self._done and self._settle_left <= 0

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
        self._hold_left = 0

    @property
    def done(self) -> bool:
        return self._done

    @property
    def contact_triggered(self) -> bool:
        return self._contact

    def open(self, duration_frames: int = 25) -> None:
        self._mode = "open"
        self._progress = 0.0
        self._total_frames = max(duration_frames, 1)
        self._done = False
        self._contact = False
        self._hold_left = 0

    def close(self, duration_frames: int = 100) -> None:
        self._mode = "close"
        self._progress = 0.0
        self._total_frames = max(duration_frames, 1)
        self._done = False
        self._contact = False
        self._hold_left = 0

    def hold(self, duration_frames: int = GRIP_HOLD_FRAMES) -> None:
        self._mode = "hold"
        self._progress = 1.0
        self._total_frames = 1
        self._done = False
        self._contact = True
        self._hold_left = max(duration_frames, 1)
        for item in self._limits:
            act = self._j2a.get(item["joint"])
            if act is not None:
                self._last_ctrl[act] = float(self._data.ctrl[act])

    def step(self, force_threshold: float = GRIP_CONTACT_FORCE,
             max_force: float = GRIP_MAX_FORCE) -> None:
        if self._done:
            self._write_final()
            return
        measured_force = self._max_actuator_force()
        if self._mode == "hold":
            if measured_force > max_force:
                self._relax_hold(GRIP_SQUEEZE_MARGIN * 0.5)
            elif measured_force < force_threshold * 0.65:
                self._tighten_hold(GRIP_SQUEEZE_MARGIN * 0.25)
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
            if self._mode == "close":
                val = item["open"] + (item["closed"] - item["open"]) * self._progress
            else:
                val = item["closed"] + (item["open"] - item["closed"]) * self._progress
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


def generate_grasp_orientations(
    ball_xyz: np.ndarray,
    opening_hints: list[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Prefer top-down grasps, then gradually tilt toward edge targets."""
    radial = _radial_xy(ball_xyz[:2])
    tangent = _tangent_xy(radial)
    candidates: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()

    base_hints: list[np.ndarray] = []
    if opening_hints:
        base_hints.extend(opening_hints)
    base_hints.extend([
        tangent, -tangent,
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    ])

    for tilt_deg in [0.0, 15.0, 30.0, 45.0, 58.0]:
        tilt = math.radians(tilt_deg)
        z_axis = _unit(
            math.sin(tilt) * radial + np.array([0.0, 0.0, -math.cos(tilt)]),
            np.array([0.0, 0.0, -1.0], dtype=np.float64),
        )
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
    return (
        3.0 * horizontal_tilt +
        1.0 * max(0.0, 0.98 - downward) +
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
        inject_wrist_camera, inject_force_sensor, inject_actuators,
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
                  friction="4.0 1.5 0.5" solimp="0.95 0.99 0.001"
                  solref="0.015 1"/>
        </body>"""
        xml_content = xml_content.replace("</worldbody>", custom_cube_xml + "\n  </worldbody>")
    xml_content = inject_wrist_camera(xml_content)
    xml_content = inject_force_sensor(xml_content)
    # (inject_soft_ball is skipped because target_cube already exists)
    # Configure the real finger mesh and hide the old visible marker only after
    # wrist camera / F-T sensor insertion, because those helpers use that marker
    # as their XML anchor.
    xml_content = configure_finger_mesh_collision(xml_content)
    xml_content = inject_actuators(xml_content)
    # Must be AFTER inject_actuators 鈥?inserts into the existing <actuator>
    xml_content = inject_control_actuators(xml_content)

    # 鈹€鈹€ Boost PD gains for precise positioning under gravity 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    xml_content = xml_content.replace('kp="90"', 'kp="350"')
    xml_content = xml_content.replace('kp="70"', 'kp="250"')
    xml_content = xml_content.replace('kp="45"', 'kp="180"')
    # Stronger real gripper forces.  The default MJCF limits are too weak for
    # a 40 mm cube and let the fingers get pushed open during contact.
    xml_content = xml_content.replace(
        '<joint name="left_finger" pos="0 0 0" axis="0 0 1" type="slide" range="-0.025 0" actuatorfrcrange="-5 5" />',
        '<joint name="left_finger" pos="0 0 0" axis="0 0 1" type="slide" range="-0.025 0" actuatorfrcrange="-60 60" />',
    )
    xml_content = xml_content.replace(
        '<joint name="right_finger" pos="0 0 0" axis="0 0 -1" type="slide" range="0 0.025" actuatorfrcrange="-5 5" />',
        '<joint name="right_finger" pos="0 0 0" axis="0 0 -1" type="slide" range="0 0.025" actuatorfrcrange="-60 60" />',
    )
    xml_content = xml_content.replace(
        '<position name="left_finger_act" joint="left_finger" kp="55" kv="7"\n              forcerange="-8 8"/>',
        '<position name="left_finger_act" joint="left_finger" kp="220" kv="18"\n              forcerange="-60 60"/>',
    )
    xml_content = xml_content.replace(
        '<position name="right_finger_act" joint="right_finger" kp="55" kv="7"\n              forcerange="-8 8"/>',
        '<position name="right_finger_act" joint="right_finger" kp="220" kv="18"\n              forcerange="-60 60"/>',
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
    pinch_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper_pinch_center_body")
    if left_id < 0 or right_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger bodies.")
    if tool_id < 0:
        raise RuntimeError("Could not find Link6 tool body.")
    if left_finger_geom_id < 0 or right_finger_geom_id < 0:
        raise RuntimeError("Could not find Link7/Link8 finger mesh collision geoms.")
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

    # 鈹€鈹€ Force / torque 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ft_sensor = ForceTorqueSensor(model)
    ft_display = FTDisplay(width=500, height=350, history_len=180)

    # 鈹€鈹€ Eye-in-hand camera 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    rgb_window = None
    if rgb_cam_id >= 0:
        try:
            rgb_window = RGBCameraWindow(
                model, rgb_cam_id, width=480, height=360,
                render_every_n=CAMERA_EVERY_N,
                window_name="Eye-in-Hand RGB Camera")
            print("Eye-in-hand camera: active (480x360)")
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
    rejected_grasp_frames: list[tuple[np.ndarray, np.ndarray]] = []
    regrasp_count = 0
    MAX_REGRASP = 4
    pregrasp_replan_count = 0
    local_replan_count = 0
    grip_contact_hold = 0
    grip_close_frames = 0
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
        if ball_qpos_adr < 0:
            return
        cube_static_pos = current_ball_xyz()
        cube_static_quat = current_cube_quat()
        cube_static_anchor_active = True
        zero_cube_velocity()
        mujoco.mj_forward(model, data)
        print(f"    Cube static anchor ON ({context}): "
              f"{np.round(cube_static_pos, 4)}")

    def release_cube_static_anchor(context: str) -> None:
        nonlocal cube_static_anchor_active
        if cube_static_anchor_active:
            cube_static_anchor_active = False
            zero_cube_velocity()
            mujoco.mj_forward(model, data)
            print(f"    Cube static anchor OFF ({context})")

    def apply_cube_static_anchor() -> None:
        if (not cube_static_anchor_active or grip_locked or
                ball_qpos_adr < 0 or phase == -1):
            return
        if phase in (0, 1, 2, 3) and cube_touching_robot():
            # The anchor is only for idle solver jitter.  Robot contact is a
            # real external force, so pinning the cube here creates apparent
            # penetration and violates the physics we are trying to test.
            release_cube_static_anchor("robot touched cube before gripper close")
            return
        if phase == 4:
            # Keep the cube perfectly still until the fingers really touch it.
            # Once contact exists, let the solver resolve the squeeze.
            left_finger_hit, right_finger_hit, _ = pad_cube_contact_sides()
            bad_contact, bad_name = cube_nonfinger_robot_contact()
            if bad_contact:
                release_cube_static_anchor(f"non-finger robot contact: {bad_name}")
                return
            if not (left_finger_hit or right_finger_hit):
                data.qpos[ball_qpos_adr:ball_qpos_adr + 3] = cube_static_pos
                data.qpos[ball_qpos_adr + 3:ball_qpos_adr + 7] = cube_static_quat
                zero_cube_velocity()
                mujoco.mj_forward(model, data)
            else:
                release_cube_static_anchor("first real finger contact")
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

    def scratch_gripper_contact_center(scratch) -> np.ndarray:
        return scratch.xpos[pinch_body_id].copy()

    def gripper_body_target_for_contact(contact_target: np.ndarray,
                                        xmat: np.ndarray) -> np.ndarray:
        return np.asarray(contact_target, dtype=np.float64)

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
        left = data.xpos[left_id].copy()
        right = data.xpos[right_id].copy()
        span = right - left
        center_gap = float(np.linalg.norm(span))
        if center_gap < 1e-9:
            opening_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            opening_axis = span / center_gap
        delta = cube - gripper_contact_center()
        open_axis_err = abs(float(np.dot(delta, opening_axis)))
        face_delta = delta - np.dot(delta, opening_axis) * opening_axis
        face_axis_err = float(np.linalg.norm(face_delta))
        center_err = float(np.linalg.norm(delta))
        ok = (
            open_axis_err <= GRIP_LOCK_MAX_OPEN_AXIS_ERR and
            face_axis_err <= GRIP_LOCK_MAX_FACE_AXIS_ERR and
            center_err <= GRIP_LOCK_MAX_ERR
        )
        return {
            "ok": bool(ok),
            "center_err": center_err,
            "open_axis_err": open_axis_err,
            "face_axis_err": face_axis_err,
            "surface_gap": float(center_gap),
        }

    def activate_grip_lock(context: str) -> bool:
        if ball_qpos_adr < 0:
            return False
        cube = current_ball_xyz()
        gc = gripper_contact_center()
        err = float(np.linalg.norm(cube - gc))
        contacts = finger_contact_count()
        left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
        force = gripper_ctrl._max_actuator_force()
        geom = grip_geometry_metrics()
        if (not left_pad_hit or not right_pad_hit or
                force < GRIP_LOCK_MIN_FORCE or err > GRIP_LOCK_MAX_ERR or
                not bool(geom["ok"])):
            print(f"    Physical grasp refused ({context}): contacts={contacts} "
                  f"finger_contacts={pad_contacts} left={left_pad_hit} right={right_pad_hit} "
                  f"force={force:.2f}N err={err*100:.1f}cm "
                  f"geom_center={geom['center_err']*100:.1f}cm "
                  f"open_axis={geom['open_axis_err']*100:.1f}cm "
                  f"face_axis={geom['face_axis_err']*100:.1f}cm "
                  f"surface_gap={geom['surface_gap']*100:.1f}cm")
            return False
        release_cube_static_anchor("physical grasp confirmed")
        print(f"    Physical grasp confirmed ({context}): contacts={contacts} "
              f"finger_contacts={pad_contacts} force={force:.2f}N "
              f"geom_center={geom['center_err']*100:.1f}cm")
        return True

    def release_grip_lock(context: str) -> None:
        nonlocal grip_locked
        if grip_locked:
            grip_locked = False
            zero_cube_velocity()
            mujoco.mj_forward(model, data)
            print(f"    Physical grasp state cleared ({context})")

    def apply_grip_lock() -> None:
        # Real-physics mode: never kinematically attach the cube to the gripper.
        # The cube may only move through MuJoCo contact and friction.
        return

    def stabilize_cube_at_rest() -> None:
        if ball_qpos_adr < 0 or cube_touching_fingers():
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

    def capture_cube_from_rgb(label: str, *, draw: bool = True) -> np.ndarray | None:
        nonlocal detected_opening_hints
        if rgb_window is None:
            print(f"    RGB camera unavailable during {label}; refusing fixed target.")
            return None
        try:
            renderer = rgb_window.renderer
            renderer.update_scene(data, camera=rgb_cam_id)
            rgb_img = renderer.render()
            center_uv, radius = detector.detect(rgb_img, data)
            if center_uv is None:
                if draw and CV2_AVAILABLE:
                    annotated = draw_detection_overlay(rgb_img, False)
                    cv2.imshow("Eye-in-Hand RGB Camera",
                               cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                print(f"    Vision {label}: cube not in RGB view")
                return None
            est_xyz = detector.estimate_3d(
                center_uv, data, rgb_window.width, rgb_window.height,
                target_z=CUBE_REST_Z,
            )
            if est_xyz is None:
                print(f"    Vision {label}: ray-plane miss at uv={center_uv}")
                return None
            in_workspace = (
                workspace_contains_xy(est_xyz[:2]) and
                abs(est_xyz[2] - CUBE_REST_Z) < 0.12
            )
            if not in_workspace:
                print(f"    Vision {label}: rejected estimate {np.round(est_xyz, 3)}")
                return None
            detector._last_estimated_xyz = est_xyz
            detected_opening_hints = detector.last_opening_hints_world(data)
            if draw and CV2_AVAILABLE:
                annotated = draw_detection_overlay(rgb_img, True, est_xyz)
                cv2.imshow("Eye-in-Hand RGB Camera",
                           cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            hint_msg = ""
            if detected_opening_hints:
                hint_msg = f" opening_hint={np.round(detected_opening_hints[0][:2], 3)}"
            print(f"    Vision {label}: uv={center_uv} r={radius}px "
                  f"xyz={np.round(est_xyz, 3)}{hint_msg}")
            return est_xyz
        except Exception as exc:
            print(f"    Vision {label}: ERROR {exc}")
            import traceback; traceback.print_exc()
            return None

    def plan_is_ready(plan: dict[str, IKResult]) -> bool:
        required = {"approach", "grasp", "lift", "place_above", "place_drop"}
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
        gc = gripper_contact_center()
        lin_v, ang_v = cube_velocity_norms()
        print(f"    DIAG [{context}]")
        print(f"      cube={np.round(cube, 4)}  contact_center={np.round(gc, 4)}  "
              f"err={np.linalg.norm(cube-gc)*100:.1f}cm")
        print(f"      cube_v={lin_v:.4f}m/s  cube_w={ang_v:.4f}rad/s  "
              f"contacts={finger_contact_count()}  "
              f"force={gripper_ctrl._max_actuator_force():.2f}N")
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
            orientation_weight=0.22, orientation_tol=ori_tol,
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
                              ball_xyz: np.ndarray) -> float:
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
        pre_min_dist = scratch_cube_robot_min_contact_dist(sim)
        if (pre_contacts > 0 or scratch_nonfinger_robot_contact(sim) or
                pre_min_dist < -EARLY_CONTACT_MAX_PENETRATION):
            penetration = 0.0 if pre_min_dist == float("inf") else max(0.0, -pre_min_dist)
            return 120.0 + penetration * 1000.0 + pre_contacts * 5.0

        grip_sim = SmoothGripperController(
            model, sim, gripper_limits, joint_to_actuator)
        grip_sim.close(160)
        for frame_idx in range(260):
            write_arm(plan["grasp"].angles)
            grip_sim.step()
            step_substeps()
            if grip_sim.done and frame_idx > 160:
                break

        left_hit, right_hit, contact_count = scratch_finger_contact_sides(sim)
        if (not left_hit or not right_hit or scratch_nonfinger_robot_contact(sim)):
            missing = 0
            if not left_hit:
                missing += 1
            if not right_hit:
                missing += 1
            return 80.0 + 10.0 * missing + max(0, 2 - contact_count) * 5.0

        z_before = float(sim.qpos[ball_qpos_adr + 2])
        arm_sim = SmoothArmController(model, sim, arm_joints, joint_to_actuator)
        arm_sim.set_target(plan["lift"].angles,
                           speed=SPEED_SLOW, min_frames=140)
        for frame_idx in range(220):
            arm_sim.step()
            grip_sim.step()
            step_substeps()
            if arm_sim.done and frame_idx > 140:
                break

        cube = sim.qpos[ball_qpos_adr:ball_qpos_adr + 3].copy()
        err = float(np.linalg.norm(cube - scratch_gripper_contact_center(sim)))
        lifted = float(cube[2] - z_before)
        left_hit, right_hit, contacts = scratch_finger_contact_sides(sim)
        if (not left_hit or not right_hit or scratch_nonfinger_robot_contact(sim) or
                lifted < 0.08 or err > CARRY_MAX_ERR):
            return 60.0 + max(0.0, 0.08 - lifted) * 100.0 + err * 20.0

        return err * 2.0

    def compute_dynamic_ik(
        ball_xyz: np.ndarray,
        grasp_z_offs: float = 0.0,
        opening_hints: list[np.ndarray] | None = None,
    ) -> dict[str, IKResult]:
        """Compute a pose-aware IK plan for the detected cube position."""
        nonlocal dynamic_grasp_xmat
        dynamic_grasp_xmat = None
        ball_xyz = np.asarray(ball_xyz, dtype=np.float64).copy()
        ball_xyz[2] = max(float(ball_xyz[2]), CUBE_REST_Z)
        grasp_contact_target = ball_xyz + np.array([0.0, 0.0, grasp_z_offs])
        place_above_contact_target = BOX_POS + np.array([0.0, 0.0, 0.16])
        place_drop_contact_target = np.array([
            BOX_POS[0], BOX_POS[1],
            BOX_POS[2] + BOX_SIZE[2] + CUBE_HALF_SIZE * 0.35,
        ], dtype=np.float64)

        best_plan: dict[str, IKResult] | None = None
        best_xmat: np.ndarray | None = None
        best_score = float("inf")
        candidate_plans: list[tuple[float, dict[str, IKResult], np.ndarray]] = []
        home_rest = np.array(
            [data.qpos[model.jnt_qposadr[j]] for j in arm_joints],
            dtype=np.float64,
        )

        for xmat in generate_grasp_orientations(ball_xyz, opening_hints):
            z_axis = xmat[:, 2]
            grasp_target = gripper_body_target_for_contact(grasp_contact_target, xmat)
            lift_target = gripper_body_target_for_contact(
                grasp_contact_target + np.array([0.0, 0.0, 0.18]),
                xmat,
            )
            place_above_target = gripper_body_target_for_contact(
                place_above_contact_target,
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
                result = solve_oriented_target(
                    scratch, target, xmat, rest=rest,
                    pos_tol=0.005 if name not in ("place_above", "place_drop") else 0.008,
                    ori_tol=0.16 if name not in ("place_above", "place_drop") else 0.22,
                )
                relaxed_place = False
                if not result.success and name in ("place_above", "place_drop"):
                    result = solve_gripper_center_ik(
                        model, scratch, target,
                        pinch_body_id, pinch_body_id, arm_joints,
                        max_iter=700, tol=0.008,
                    )
                    relaxed_place = result.success
                plan[name] = result
                score += result.error_norm * 100.0 + result.orientation_error_norm
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
                candidate_plans.append((score, plan, xmat.copy()))
                if score < best_score:
                    best_score = score
                    best_plan = plan
                    best_xmat = xmat.copy()
            elif best_plan is None and score < best_score:
                best_score = score
                best_plan = plan

        if candidate_plans:
            candidate_plans.sort(key=lambda item: item[0])
            preview_best_plan: dict[str, IKResult] | None = None
            preview_best_xmat: np.ndarray | None = None
            preview_best_score = float("inf")
            for idx, (base_score, plan, xmat) in enumerate(
                    candidate_plans[:PHYSICAL_EVAL_TOP_K], start=1):
                penalty = physical_plan_penalty(plan, ball_xyz)
                total = base_score + penalty
                y_axis = xmat[:, 1]
                z_axis = xmat[:, 2]
                print(f"    Physical preview {idx}: total={total:.2f} "
                      f"penalty={penalty:.2f} "
                      f"open={np.round(y_axis[:2], 2)} "
                      f"approach={np.round(z_axis, 2)}")
                if total < preview_best_score:
                    preview_best_score = total
                    preview_best_plan = plan
                    preview_best_xmat = xmat.copy()
            if preview_best_plan is not None:
                best_plan = preview_best_plan
                best_xmat = preview_best_xmat

        if (best_plan is None or
            set(best_plan) != {"approach", "grasp", "lift", "place_above", "place_drop"} or
            any(not result.success for result in best_plan.values())):
            print("    Pose IK incomplete; falling back to position-only IK.")
            best_plan = {}
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = data.qpos[:]
            scratch.qvel[:] = 0.0
            mujoco.mj_forward(model, scratch)
            fallback_xmat = make_tool_xmat(
                np.array([0.0, 0.0, -1.0], dtype=np.float64),
                opening_hints[0] if opening_hints else _tangent_xy(_radial_xy(ball_xyz[:2])),
            )
            fallback_grasp = gripper_body_target_for_contact(
                grasp_contact_target, fallback_xmat)
            fallback_lift = gripper_body_target_for_contact(
                grasp_contact_target + np.array([0.0, 0.0, 0.20]),
                fallback_xmat,
            )
            fallback_place_above = gripper_body_target_for_contact(
                place_above_contact_target, fallback_xmat)
            fallback_place_drop = gripper_body_target_for_contact(
                place_drop_contact_target, fallback_xmat)
            fallback_targets = {
                "approach": ball_xyz + np.array([0.0, 0.0, 0.15]),
                "grasp": fallback_grasp,
                "lift": fallback_lift,
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

        if best_xmat is not None:
            dynamic_grasp_xmat = best_xmat.copy()
        for name in ("approach", "grasp", "lift", "place_above", "place_drop"):
            result = best_plan[name]
            status = "reachable" if result.success else "UNREACHABLE"
            print(f"  IK {name:8s}: {status}, pos={result.error_norm:.4f}m, "
                  f"ori={result.orientation_error_norm:.3f}, "
                  f"target={np.round(result.target, 3)}")
        return best_plan

    def cube_carry_metrics() -> tuple[float, float]:
        gc = gripper_contact_center()
        cube = current_ball_xyz()
        err = float(np.linalg.norm(cube - gc))
        lifted = float(cube[2] - CUBE_REST_Z)
        return err, lifted

    def cube_is_secured() -> bool:
        err, lifted = cube_carry_metrics()
        left_pad_hit, right_pad_hit, _ = pad_cube_contact_sides()
        return (
            lifted > CARRY_MIN_LIFT and
            err < CARRY_MAX_ERR and
            left_pad_hit and right_pad_hit
        )

    def trusted_cube_xyz() -> np.ndarray:
        if cube_static_anchor_active:
            return cube_static_pos.copy()
        return current_ball_xyz()

    def pre_close_alignment_ok(label: str) -> bool:
        cube = current_ball_xyz()
        pinch = gripper_contact_center()
        xy_err = float(np.linalg.norm((pinch - cube)[:2]))
        z_err = abs(float(pinch[2] - cube[2]))
        left = data.xpos[left_id].copy()
        right = data.xpos[right_id].copy()
        span = right - left
        opening_axis = _unit(span, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        center_delta = cube - pinch
        open_axis_err = abs(float(np.dot(center_delta, opening_axis)))
        center_err = float(np.linalg.norm(center_delta))
        bad_contact, bad_name = cube_nonfinger_robot_contact()
        print(f"    {label}: cube={np.round(cube, 4)} "
              f"pinch={np.round(pinch, 4)} "
              f"xy_err={xy_err*100:.1f}cm z_err={z_err*100:.1f}cm "
              f"open_axis={open_axis_err*100:.1f}cm "
              f"center={center_err*100:.1f}cm")
        if bad_contact:
            print(f"    {label}: rejected, cube is touching non-finger robot part "
                  f"{bad_name}")
            return False
        return (
            xy_err <= PRE_CLOSE_MAX_XY_ERR and
            z_err <= PRE_CLOSE_MAX_Z_ERR and
            open_axis_err <= PRE_CLOSE_MAX_OPEN_AXIS_ERR and
            center_err <= PRE_CLOSE_MAX_CENTER_ERR
        )

    def restart_search(reason: str) -> None:
        nonlocal phase, sub, scan_idx, scan_targets, detected_ball_pos
        nonlocal detected_opening_hints, dynamic_ik_plan, regrasp_count, scan_round, status_msg
        nonlocal pregrasp_replan_count, local_replan_count, dynamic_grasp_xmat
        print(f"    Restarting search: {reason}")
        release_grip_lock("restart search")
        set_cube_static_anchor("restart search")
        detected_ball_pos = None
        detected_opening_hints = []
        dynamic_ik_plan = {}
        dynamic_grasp_xmat = None
        scan_targets = []
        scan_idx = 0
        regrasp_count = 0
        pregrasp_replan_count = 0
        local_replan_count = 0
        scan_round = 0
        phase = 0
        sub = 0
        status_msg = f"Rescanning cube ({reason}) ..."

    def demo_tick() -> None:
        nonlocal phase, sub, finished, scan_idx, scan_targets, regrasp_count
        nonlocal detected_ball_pos, detected_opening_hints
        nonlocal dynamic_ik_plan, status_msg, ball_z_before_lift
        nonlocal scan_round, box_verify_left, pregrasp_replan_count
        nonlocal grip_contact_hold, grip_close_frames, local_replan_count
        nonlocal dynamic_grasp_xmat, rejected_grasp_frames

        if finished:
            return

        # 鈹€鈹€ Phase -1:  idle 鈥?wait for SPACE 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if phase == -1:
            status_msg = "IDLE 鈥?use cube_x/y/z & start_demo sliders"
            return  # do nothing, wait for keyboard

        # 鈹€鈹€ Phase 0:  scanning 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 0:
            if sub == 0:
                if rgb_window is None:
                    print("    ERROR: RGB camera is not available; refusing to use a fixed cube target.")
                    status_msg = "ERROR: RGB camera unavailable"
                    finished = True
                    return
                print(">>> Phase 0 : Camera scanning for cube ...")
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
                    return

                # Capture and detect using the wrist RGB camera. No RGB
                # estimate means no grasp target is accepted.
                est_xyz = capture_cube_from_rgb(
                    f"scan {scan_idx+1}/{len(scan_targets)}"
                )
                if est_xyz is not None:
                    plan_xyz = est_xyz
                    if cube_static_anchor_active:
                        rgb_anchor_err = float(np.linalg.norm(
                            est_xyz[:2] - cube_static_pos[:2]
                        ))
                        print(f"    RGB-anchor check: "
                              f"rgb={np.round(est_xyz, 3)} "
                              f"anchor={np.round(cube_static_pos, 3)} "
                              f"err={rgb_anchor_err*100:.1f}cm")
                        if rgb_anchor_err > RGB_ANCHOR_ACCEPT_TOL:
                            print("    RGB estimate rejected: inconsistent with "
                                  "the stationary cube pose; continuing scan.")
                            scan_idx += 1
                            while scan_idx < len(scan_targets):
                                target = scan_targets[scan_idx]
                                result = solve_gripper_center_ik(
                                    model, data, target.gripper,
                                    pinch_body_id, pinch_body_id, arm_joints,
                                    max_iter=700, tol=SCAN_POS_TOL,
                                    target_xmat=target.xmat, orientation_body_id=tool_id,
                                    orientation_weight=SCAN_ORI_WEIGHT,
                                    orientation_tol=SCAN_ORI_TOL,
                                )
                                if result.success:
                                    print(f"    Moving to scan {scan_idx+1}/{len(scan_targets)} "
                                          f"aim={np.round(target.aim, 3)}")
                                    arm_ctrl.set_target(result.angles, speed=SPEED_SCAN)
                                    status_msg = f"Scanning {scan_idx+1}/{len(scan_targets)} ..."
                                    return
                                scan_idx += 1
                            scan_round += 1
                            if scan_round >= 3:
                                print("    Cube not found after 3 full scan rounds — stopping.")
                                status_msg = "ERROR: cube not found — RGB estimates inconsistent"
                                finished = True
                                return
                            print(f"    All {len(scan_targets)} poses scanned — "
                                  f"restarting scan round {scan_round+1} ...")
                            scan_idx = 0
                            sub = 0
                            status_msg = (f"Rescanning round {scan_round+1} "
                                          f"— RGB estimate inconsistent")
                            return
                        plan_xyz = trusted_cube_xyz()
                    detected_ball_pos = plan_xyz
                    pregrasp_replan_count = 0
                    dynamic_ik_plan = compute_dynamic_ik(
                        detected_ball_pos, opening_hints=detected_opening_hints)
                    if plan_is_ready(dynamic_ik_plan):
                        status_msg = (f"Cube found by RGB! "
                                      f"X={est_xyz[0]:.3f} Y={est_xyz[1]:.3f} Z={est_xyz[2]:.3f}")
                        phase = 2; sub = 0
                        return
                    print("    RGB detection rejected: no complete IK plan for this pose.")
                    diagnose_failure("rgb detection without reachable plan")
                    detected_ball_pos = None
                    dynamic_ik_plan = {}

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
                print(">>> Phase 2 : Approach")
                arm_ctrl.set_target(dynamic_ik_plan["approach"].angles, speed=SPEED_NORMAL)
                sub = 1
                status_msg = "Approaching cube ..."
            if arm_ctrl.done:
                fresh = capture_cube_from_rgb("pre-grasp confirm")
                if fresh is None:
                    if cube_static_anchor_active and detected_ball_pos is not None:
                        print("    Pre-grasp RGB lost cube, likely due to close-range "
                              "occlusion; continuing with anchored cube pose instead "
                              "of restarting scan.")
                        detected_ball_pos = trusted_cube_xyz()
                        dynamic_ik_plan = compute_dynamic_ik(
                            detected_ball_pos, opening_hints=detected_opening_hints)
                        if not plan_is_ready(dynamic_ik_plan):
                            diagnose_failure("anchored plan failed after RGB occlusion")
                            restart_search("no reachable grasp from anchored cube after RGB occlusion")
                            return
                        pregrasp_replan_count = 0
                        phase = 3; sub = 0
                        return
                    diagnose_failure("pre-grasp RGB lost cube")
                    restart_search("cube moved out of RGB view before grasp")
                    return
                previous = detected_ball_pos.copy()
                delta = float(np.linalg.norm(fresh[:2] - previous[:2]))
                if cube_static_anchor_active:
                    anchor_delta = float(np.linalg.norm(fresh[:2] - cube_static_pos[:2]))
                    if delta > VISION_REPLAN_DELTA:
                        print(f"    Vision estimate shifted {delta*100:.1f}cm "
                              f"(anchor error {anchor_delta*100:.1f}cm); "
                              "using static cube state to avoid RGB replan loop.")
                    detected_ball_pos = trusted_cube_xyz()
                    dynamic_ik_plan = compute_dynamic_ik(
                        detected_ball_pos, opening_hints=detected_opening_hints)
                    if not plan_is_ready(dynamic_ik_plan):
                        diagnose_failure("anchored pre-grasp replan failed")
                        restart_search("no reachable grasp from anchored cube state")
                        return
                    approach_err = float(np.linalg.norm(
                        gripper_contact_center() -
                        dynamic_ik_plan["approach"].target
                    ))
                    if (pregrasp_replan_count < MAX_PREGRASP_REPLANS and
                            (delta > VISION_REPLAN_DELTA or
                             approach_err > PREGRASP_REAPPROACH_TOL)):
                        pregrasp_replan_count += 1
                        print(f"    Re-approaching anchored cube "
                              f"({pregrasp_replan_count}/{MAX_PREGRASP_REPLANS}); "
                              f"approach_err={approach_err*100:.1f}cm")
                        sub = 0
                        status_msg = "Re-approaching anchored cube ..."
                        return
                    pregrasp_replan_count = 0
                    phase = 3; sub = 0
                    return
                if delta > VISION_REPLAN_DELTA:
                    if pregrasp_replan_count >= MAX_PREGRASP_REPLANS:
                        print(f"    Vision estimate still shifted {delta*100:.1f}cm "
                              f"after {pregrasp_replan_count} replan(s); "
                              "accepting latest RGB estimate to continue grasp.")
                        detected_ball_pos = trusted_cube_xyz()
                        dynamic_ik_plan = compute_dynamic_ik(
                            detected_ball_pos, opening_hints=detected_opening_hints)
                        if not plan_is_ready(dynamic_ik_plan):
                            diagnose_failure("final pre-grasp RGB plan failed")
                            restart_search("no reachable grasp from final RGB estimate")
                            return
                        pregrasp_replan_count = 0
                        phase = 3; sub = 0
                        return
                    pregrasp_replan_count += 1
                    print(f"    Vision estimate shifted {delta*100:.1f}cm since scan; "
                          f"replanning from RGB "
                          f"({pregrasp_replan_count}/{MAX_PREGRASP_REPLANS}).")
                    detected_ball_pos = trusted_cube_xyz()
                    dynamic_ik_plan = compute_dynamic_ik(
                        detected_ball_pos, opening_hints=detected_opening_hints)
                    if not plan_is_ready(dynamic_ik_plan):
                        diagnose_failure("replan after RGB shift failed")
                        restart_search("no reachable grasp after RGB shift")
                        return
                    sub = 0
                    return
                detected_ball_pos = trusted_cube_xyz()
                pregrasp_replan_count = 0
                phase = 3; sub = 0

        # 鈹€鈹€ Phase 3:  descend 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 3:
            if sub == 0:
                if detected_ball_pos is None:
                    print("    No cube position — restarting search")
                    restart_search("lost cube position before descend")
                    return
                # Align the real jaw center with the cube center.  Retries
                # sample slightly different heights instead of driving below
                # the cube, which caused table/box scraping and side misses.
                offsets = [0.0, 0.004, -0.004, 0.008, -0.008]
                z_offs = offsets[min(regrasp_count, len(offsets) - 1)]
                print(f">>> Phase 3 : Descend  (grasp Z offset = {z_offs*1000:.0f} mm)")
                # Recompute IK with deeper grasp on each retry
                dynamic_ik_plan = compute_dynamic_ik(
                    detected_ball_pos,
                    grasp_z_offs=z_offs,
                    opening_hints=detected_opening_hints,
                )
                if not plan_is_ready(dynamic_ik_plan):
                    diagnose_failure("descend requested with incomplete plan")
                    restart_search("incomplete IK plan before descend")
                    return
                arm_ctrl.set_target(dynamic_ik_plan["grasp"].angles, speed=SPEED_SLOW)
                sub = 1
                status_msg = f"Descending (retry {regrasp_count}) ..."
            if sub == 1 and not arm_ctrl.done and cube_touching_robot():
                min_dist = cube_robot_min_contact_dist()
                print(f"    Early robot-cube contact during descent "
                      f"(min_dist={min_dist*1000:.2f} mm); "
                      "aborting this grasp posture.")
                remember_failed_grasp_frame("early contact during descent")
                release_cube_static_anchor("early contact during descent")
                if plan_is_ready(dynamic_ik_plan):
                    arm_ctrl.set_target(dynamic_ik_plan["approach"].angles,
                                        speed=SPEED_NORMAL, min_frames=80)
                regrasp_count += 1
                if regrasp_count >= MAX_REGRASP:
                    restart_search("repeated early cube collision")
                else:
                    phase = 8; sub = 0
                return
            if arm_ctrl.done:
                if cube_touching_robot():
                    min_dist = cube_robot_min_contact_dist()
                    if min_dist < -EARLY_CONTACT_MAX_PENETRATION:
                        print(f"    Pre-close penetration detected "
                              f"(min_dist={min_dist*1000:.2f} mm); "
                              "rejecting this grasp posture.")
                        remember_failed_grasp_frame("pre-close penetration")
                        diagnose_failure("gripper penetrated cube before close")
                        release_cube_static_anchor("pre-close penetration")
                        if plan_is_ready(dynamic_ik_plan):
                            arm_ctrl.set_target(dynamic_ik_plan["approach"].angles,
                                                speed=SPEED_NORMAL, min_frames=80)
                        regrasp_count += 1
                        if regrasp_count >= MAX_REGRASP:
                            restart_search("pre-close penetration repeated")
                        else:
                            phase = 8; sub = 0
                        return
                offsets = [0.0, 0.004, -0.004, 0.008, -0.008]
                z_offs = offsets[min(regrasp_count, len(offsets) - 1)]
                if not pre_close_alignment_ok("Pre-close alignment"):
                    local_replan_count += 1
                    if local_replan_count <= MAX_LOCAL_REPLAN:
                        remember_failed_grasp_frame("pre-close alignment mismatch")
                        detected_ball_pos = trusted_cube_xyz()
                        print(f"    Refusing to close at wrong location; "
                              f"replanning to actual cube "
                              f"({local_replan_count}/{MAX_LOCAL_REPLAN}).")
                        dynamic_ik_plan = compute_dynamic_ik(
                            detected_ball_pos,
                            grasp_z_offs=z_offs,
                            opening_hints=detected_opening_hints,
                        )
                        if not plan_is_ready(dynamic_ik_plan):
                            diagnose_failure("local replan after misalignment failed")
                            restart_search("cannot align real gripper with cube")
                            return
                        arm_ctrl.set_target(dynamic_ik_plan["grasp"].angles,
                                            speed=SPEED_SLOW)
                        sub = 1
                        status_msg = "Realigning gripper to cube ..."
                        return
                    remember_failed_grasp_frame("persistent pre-close mismatch")
                    diagnose_failure("pre-close gripper/cube mismatch")
                    restart_search("gripper attempted wrong cube location")
                    return
                local_replan_count = 0
                ball_z_before_lift = float(current_ball_xyz()[2])
                phase = 4; sub = 0

        # 鈹€鈹€ Phase 4:  close gripper 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 4:
            if sub == 0:
                gc = gripper_contact_center()
                ball = current_ball_xyz()
                err = np.linalg.norm(ball - gc)
                print(f"    Contact center->cube: {err*100:.1f}cm  "
                      f"(contact Z={gc[2]:.3f}, cube Z={ball[2]:.3f})")
                grip_contact_hold = 0
                grip_close_frames = 0
                gripper_ctrl.close(duration_frames=GRIP_CLOSE_FRAMES)
                sub = 1
                status_msg = "Closing gripper ..."
            if sub == 1:
                grip_close_frames += 1
                contacts = finger_contact_count()
                left_pad_hit, right_pad_hit, pad_contacts = pad_cube_contact_sides()
                bad_contact, bad_name = cube_nonfinger_robot_contact()
                force = gripper_ctrl._max_actuator_force()
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
                        force >= GRIP_LOCK_MIN_FORCE and bool(geom["ok"])):
                    grip_contact_hold += 1
                else:
                    grip_contact_hold = 0

                if grip_contact_hold == 1:
                    print(f"    Two-sided finger contact detected while closing: "
                          f"finger_contacts={pad_contacts} force={force:.2f}N "
                          f"geom_center={geom['center_err']*100:.1f}cm")
                elif grip_close_frames % 45 == 0:
                    print(f"    Closing progress: left={left_pad_hit} right={right_pad_hit} "
                          f"finger_contacts={pad_contacts} force={force:.2f}N "
                          f"hold={grip_contact_hold}/{GRIP_CONTACT_HOLD_FRAMES} "
                          f"geom_ok={bool(geom['ok'])} "
                          f"center={geom['center_err']*100:.1f}cm "
                          f"open_axis={geom['open_axis_err']*100:.1f}cm "
                          f"face_axis={geom['face_axis_err']*100:.1f}cm")

                if grip_contact_hold >= GRIP_CONTACT_HOLD_FRAMES:
                    print(f"    Real two-sided finger contact confirmed: "
                          f"body_contacts={contacts} finger_contacts={pad_contacts} "
                          f"force={force:.2f}N "
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
                          f"force={force:.2f}N "
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
                force = gripper_ctrl._max_actuator_force()
                geom = grip_geometry_metrics()
                if not (left_pad_hit and right_pad_hit and bool(geom["ok"])):
                    print(f"    Grasp destabilized during pre-lift hold: "
                          f"left={left_pad_hit} right={right_pad_hit} "
                          f"finger_contacts={pad_contacts} force={force:.2f}N "
                          f"center={geom['center_err']*100:.1f}cm")
                    remember_failed_grasp_frame("pre-lift hold lost contact")
                    diagnose_failure("pre-lift grip lost contact")
                    regrasp_count += 1
                    if regrasp_count >= MAX_REGRASP:
                        restart_search("pre-lift grip unstable")
                    else:
                        phase = 8; sub = 0
                    return
                if grip_close_frames % 40 == 0:
                    print(f"    Pre-lift grip hold: contacts={contacts} "
                          f"finger_contacts={pad_contacts} force={force:.2f}N "
                          f"center={geom['center_err']*100:.1f}cm")
                grip_close_frames += 1
                if gripper_ctrl.done:
                    phase = 5; sub = 0
                    status_msg = "Grip stable; lifting ..."

        # 鈹€鈹€ Phase 5:  lift + verify 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        elif phase == 5:
            if sub == 0:
                print(">>> Phase 5 : Lift & verify")
                arm_ctrl.set_target(dynamic_ik_plan["lift"].angles,
                                    speed=SPEED_LIFT, min_frames=260)
                sub = 1
                status_msg = "Lifting ..."
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
                    dynamic_grasp_xmat = None
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
            if sub == 0:
                print(">>> Phase 6 : Move above box")
                arm_ctrl.set_target(dynamic_ik_plan["place_above"].angles,
                                    speed=SPEED_CARRY, min_frames=480)
                sub = 1
                status_msg = "Moving above box ..."
            elif sub == 1 and arm_ctrl.done:
                if not cube_is_secured():
                    diagnose_failure("cube dropped while carrying")
                    restart_search("cube dropped while carrying")
                    return
                print(">>> Phase 6b : Lower into box")
                arm_ctrl.set_target(dynamic_ik_plan["place_drop"].angles,
                                    speed=SPEED_PLACE, min_frames=260)
                sub = 2
                status_msg = "Lowering into box ..."
            elif sub == 2 and arm_ctrl.done:
                if not cube_is_secured():
                    diagnose_failure("cube dropped before release")
                    restart_search("cube dropped before release")
                    return
                phase = 7; sub = 0
        elif phase == 7:
            if sub == 0:
                print(">>> Phase 7 : Release into box")
                release_cube_static_anchor("release in box")
                release_grip_lock("release in box")
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
                    # Re-scan to find cube after failed grasp
                    restart_search("cube dropped after failed grasp")
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
        last_ft_reading = ft_sensor.read(data)

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
            if frame % FT_EVERY_N == 0:
                last_ft_reading = ft_sensor.read(data)
                ft_display.update(last_ft_reading)
                ft_display.show()

            if joint_panel and frame % JOINT_EVERY_N == 0:
                idle_hint = "" if phase != -1 else " [slide start_demo to 1]"
                joint_panel.update(model, data, arm_joints, gripper_joints,
                                   status_text=status_msg + idle_hint)
                joint_panel.show()

            if rgb_window and rgb_window.should_update(frame):
                force_mag = float(np.linalg.norm(last_ft_reading.force))
                hint = " | [slide start_demo to 1]" if phase == -1 else ""
                rgb_window.update(data, overlay_text=f"|F|={force_mag:.2f}N{hint}")

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

    ft_display.close()
    if rgb_window:
        rgb_window.close()
    if joint_panel:
        joint_panel.close()
    print("Done.")


if __name__ == "__main__":
    main()
