"""Force/torque and tactile skin sensor utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


@dataclass
class FTReading:
    force: np.ndarray
    torque: np.ndarray
    timestamp: float


@dataclass
class TactileReading:
    left_force: float
    right_force: float
    timestamp: float

    @property
    def total_force(self) -> float:
        return self.left_force + self.right_force

    @property
    def balance(self) -> float:
        total = max(self.total_force, 1e-9)
        return abs(self.left_force - self.right_force) / total


class ForceTorqueSensor:
    """Reads 6-axis F/T from MuJoCo sensor data.

    MuJoCo exposes instantaneous constraint-solver forces, which can contain
    high-frequency contact chatter.  A small low-pass filter makes the display
    behave more like a real sampled F/T sensor without changing simulation
    dynamics or the gripper controller.
    """

    def __init__(self, model, force_name: str = "wrist_force",
                 torque_name: str = "wrist_torque",
                 filter_alpha: float = 0.25):
        self.force_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, force_name)
        self.torque_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, torque_name)
        if self.force_id < 0 or self.torque_id < 0:
            raise RuntimeError(f"Sensors '{force_name}'/'{torque_name}' not found in model.")
        self.force_adr = int(model.sensor_adr[self.force_id])
        self.torque_adr = int(model.sensor_adr[self.torque_id])
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self._force_filtered: np.ndarray | None = None
        self._torque_filtered: np.ndarray | None = None

    def read(self, data) -> FTReading:
        force_raw = data.sensordata[self.force_adr:self.force_adr + 3].copy()
        torque_raw = data.sensordata[self.torque_adr:self.torque_adr + 3].copy()
        if self._force_filtered is None:
            self._force_filtered = force_raw
            self._torque_filtered = torque_raw
        else:
            a = self.filter_alpha
            self._force_filtered += a * (force_raw - self._force_filtered)
            self._torque_filtered += a * (torque_raw - self._torque_filtered)
        return FTReading(
            force=self._force_filtered.copy(),
            torque=self._torque_filtered.copy(),
            timestamp=float(data.time),
        )

class FTDisplay:
    """Real-time scrolling 6-axis force/torque plot using OpenCV."""

    COLORS_FORCE = [(0, 0, 255), (0, 200, 0), (255, 0, 0)]  # Fx=red, Fy=green, Fz=blue (BGR)
    COLORS_TORQUE = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]  # Mx=cyan, My=magenta, Mz=yellow
    LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

    def __init__(self, width: int = 480, height: int = 320, history_len: int = 200):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for FTDisplay.")
        self.width = width
        self.height = height
        self.history_len = history_len
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.force_history = np.zeros((history_len, 3), dtype=np.float64)
        self.torque_history = np.zeros((history_len, 3), dtype=np.float64)
        self.write_idx = 0
        self.filled = False
        self.force_scale = 5.0
        self.torque_scale = 1.0

        cv2.namedWindow("Force/Torque Sensor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Force/Torque Sensor", width, height)

    def update(self, reading: FTReading) -> None:
        self.force_history[self.write_idx] = reading.force
        self.torque_history[self.write_idx] = reading.torque
        self.write_idx = (self.write_idx + 1) % self.history_len
        if self.write_idx == 0:
            self.filled = True

        f_max = max(np.max(np.abs(self.force_history)) + 0.1, 1.0)
        t_max = max(np.max(np.abs(self.torque_history)) + 0.01, 0.1)
        self.force_scale = f_max
        self.torque_scale = t_max

    def show(self) -> None:
        self.canvas[:] = 20
        h, w = self.height, self.width
        text_h = 50
        plot_h = (h - text_h) // 2
        n = self.history_len if self.filled else self.write_idx
        if n < 2:
            cv2.imshow("Force/Torque Sensor", self.canvas)
            return

        indices = np.arange(n)
        ordered = (indices + (self.write_idx if self.filled else 0)) % (self.history_len if self.filled else n)

        force_data = self.force_history[ordered] if self.filled else self.force_history[:n]
        torque_data = self.torque_history[ordered] if self.filled else self.torque_history[:n]

        self._draw_text(force_data[-1], torque_data[-1])
        self._draw_plot(force_data, self.force_scale, text_h, plot_h, self.COLORS_FORCE, "Force (N)")
        self._draw_plot(torque_data, self.torque_scale, text_h + plot_h, plot_h, self.COLORS_TORQUE, "Torque (Nm)")

        cv2.imshow("Force/Torque Sensor", self.canvas)

    def _draw_text(self, force: np.ndarray, torque: np.ndarray) -> None:
        labels = [f"Fx:{force[0]:+.2f}", f"Fy:{force[1]:+.2f}", f"Fz:{force[2]:+.2f}",
                  f"Mx:{torque[0]:+.3f}", f"My:{torque[1]:+.3f}", f"Mz:{torque[2]:+.3f}"]
        colors = self.COLORS_FORCE + self.COLORS_TORQUE
        for i, (lbl, clr) in enumerate(zip(labels, colors)):
            x = 10 + (i % 3) * 155
            y = 18 if i < 3 else 38
            cv2.putText(self.canvas, lbl, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, clr, 1, cv2.LINE_AA)

    def _draw_plot(self, data: np.ndarray, scale: float, y_offset: int, plot_h: int,
                   colors: list, title: str) -> None:
        w = self.width
        n = len(data)
        mid_y = y_offset + plot_h // 2

        cv2.line(self.canvas, (0, mid_y), (w, mid_y), (60, 60, 60), 1)
        cv2.putText(self.canvas, title, (5, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(self.canvas, f"+{scale:.1f}", (w - 50, y_offset + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 120, 120), 1)

        for ch in range(3):
            pts = []
            for i in range(n):
                x = int(i * (w - 1) / max(n - 1, 1))
                val = data[i, ch]
                y = int(mid_y - (val / scale) * (plot_h // 2 - 5))
                y = max(y_offset + 2, min(y_offset + plot_h - 2, y))
                pts.append((x, y))
            if len(pts) > 1:
                cv2.polylines(self.canvas, [np.array(pts, dtype=np.int32)], False, colors[ch], 1, cv2.LINE_AA)

    def close(self) -> None:
        if CV2_AVAILABLE:
            cv2.destroyWindow("Force/Torque Sensor")


class TactileSkinSensor:
    """Reads left/right inner gripper touch sensors as electronic skin."""

    def __init__(self, model, left_name: str = "left_inner_skin_touch",
                 right_name: str = "right_inner_skin_touch",
                 filter_alpha: float = 0.8,
                 force_min: float = 0.0,
                 force_max: float = 20.0,
                 resolution: float = 0.1,
                 recognition_threshold: float = 0.1):
        self.left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, left_name)
        self.right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, right_name)
        if self.left_id < 0 or self.right_id < 0:
            raise RuntimeError(
                f"Tactile sensors '{left_name}'/'{right_name}' not found in model."
            )
        self.left_adr = int(model.sensor_adr[self.left_id])
        self.right_adr = int(model.sensor_adr[self.right_id])
        self.left_dim = int(model.sensor_dim[self.left_id])
        self.right_dim = int(model.sensor_dim[self.right_id])
        self.filter_alpha = float(np.clip(filter_alpha, 0.0, 1.0))
        self.force_min = float(force_min)
        self.force_max = max(float(force_max), self.force_min)
        self.resolution = max(0.0, float(resolution))
        self.recognition_threshold = max(0.0, float(recognition_threshold))
        self._left_filtered: float | None = None
        self._right_filtered: float | None = None

    @staticmethod
    def _force_from_sensor(data, adr: int, dim: int) -> float:
        raw = data.sensordata[adr:adr + dim].astype(np.float64, copy=True)
        if raw.size == 0:
            return 0.0
        if raw.size == 1:
            return max(0.0, float(raw[0]))
        return float(np.linalg.norm(raw))

    def _apply_specs(self, force: float) -> float:
        force = float(np.clip(force, self.force_min, self.force_max))
        if force < self.recognition_threshold:
            return 0.0
        if self.resolution > 0.0:
            force = round(force / self.resolution) * self.resolution
        return float(np.clip(force, self.force_min, self.force_max))

    def read(self, data) -> TactileReading:
        left_raw = self._apply_specs(
            self._force_from_sensor(data, self.left_adr, self.left_dim)
        )
        right_raw = self._apply_specs(
            self._force_from_sensor(data, self.right_adr, self.right_dim)
        )
        if self._left_filtered is None:
            self._left_filtered = left_raw
            self._right_filtered = right_raw
        else:
            a = self.filter_alpha
            self._left_filtered += a * (left_raw - self._left_filtered)
            self._right_filtered += a * (right_raw - self._right_filtered)
        return TactileReading(
            left_force=self._apply_specs(float(self._left_filtered)),
            right_force=self._apply_specs(float(self._right_filtered)),
            timestamp=float(data.time),
        )


class TactileSkinDisplay:
    """Real-time scrolling plot for left/right inner gripper skin force."""

    COLORS = [(0, 220, 255), (255, 180, 0)]  # left/right, BGR

    def __init__(self, width: int = 500, height: int = 260,
                 history_len: int = 180):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for TactileSkinDisplay.")
        self.width = width
        self.height = height
        self.history_len = history_len
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.history = np.zeros((history_len, 2), dtype=np.float64)
        self.write_idx = 0
        self.filled = False
        self.force_scale = 1.0

        cv2.namedWindow("Gripper Tactile Skin", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gripper Tactile Skin", width, height)

    def update(self, reading: TactileReading) -> None:
        self.history[self.write_idx] = [reading.left_force, reading.right_force]
        self.write_idx = (self.write_idx + 1) % self.history_len
        if self.write_idx == 0:
            self.filled = True
        self.force_scale = max(np.max(np.abs(self.history)) + 0.05, 0.5)

    def show(self) -> None:
        self.canvas[:] = 20
        text_h = 64
        plot_h = self.height - text_h - 8
        n = self.history_len if self.filled else self.write_idx
        if n < 2:
            cv2.imshow("Gripper Tactile Skin", self.canvas)
            return

        indices = np.arange(n)
        ordered = (
            indices + (self.write_idx if self.filled else 0)
        ) % (self.history_len if self.filled else n)
        data = self.history[ordered] if self.filled else self.history[:n]
        self._draw_text(data[-1])
        self._draw_plot(data, text_h, plot_h)
        cv2.imshow("Gripper Tactile Skin", self.canvas)

    def _draw_text(self, latest: np.ndarray) -> None:
        left, right = float(latest[0]), float(latest[1])
        total = left + right
        balance = abs(left - right) / max(total, 1e-9)
        labels = [
            f"L skin:{left:.2f}N",
            f"R skin:{right:.2f}N",
            f"sum:{total:.2f}N",
            f"bal:{balance:.2f}",
        ]
        colors = [self.COLORS[0], self.COLORS[1], (220, 220, 220), (160, 220, 160)]
        for idx, (label, color) in enumerate(zip(labels, colors)):
            x = 10 + (idx % 2) * 230
            y = 22 if idx < 2 else 46
            cv2.putText(
                self.canvas, label, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )

    def _draw_plot(self, data: np.ndarray, y_offset: int, plot_h: int) -> None:
        w = self.width
        n = len(data)
        bottom = y_offset + plot_h - 6
        top = y_offset + 8
        cv2.line(self.canvas, (0, bottom), (w, bottom), (60, 60, 60), 1)
        cv2.putText(
            self.canvas, f"+{self.force_scale:.1f}N", (w - 72, top + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 130), 1,
        )
        for ch in range(2):
            pts = []
            for i in range(n):
                x = int(i * (w - 1) / max(n - 1, 1))
                val = np.clip(data[i, ch] / self.force_scale, 0.0, 1.0)
                y = int(bottom - val * (bottom - top))
                pts.append((x, y))
            if len(pts) > 1:
                cv2.polylines(
                    self.canvas, [np.array(pts, dtype=np.int32)],
                    False, self.COLORS[ch], 1, cv2.LINE_AA,
                )

    def close(self) -> None:
        if CV2_AVAILABLE:
            cv2.destroyWindow("Gripper Tactile Skin")
