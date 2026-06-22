"""Eye-in-hand RGB/RGB-D camera windows with frame rate control."""
from __future__ import annotations

import numpy as np
import mujoco

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class RGBCameraWindow:
    """Wrist-mounted RGB camera display with configurable frame decimation."""

    def __init__(
        self,
        model,
        camera_id: int,
        width: int = 640,
        height: int = 360,
        render_every_n: int = 8,
        window_name: str = "Wrist RGB Camera",
    ):
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV is required for RGBCameraWindow.")
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.render_every_n = render_every_n
        self.window_name = window_name
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.enabled = True
        self.frame_count = 0

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)

    def should_update(self, sim_step: int) -> bool:
        return self.enabled and (sim_step % self.render_every_n == 0)

    def update(self, data, overlay_text: str | None = None) -> None:
        if not self.enabled:
            return
        try:
            self.renderer.update_scene(data, camera=self.camera_id)
            rgb = self.renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if overlay_text:
                cv2.putText(
                    bgr, overlay_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )

            cv2.imshow(self.window_name, bgr)
            self.frame_count += 1
        except Exception as exc:
            if self.enabled:
                print(f"RGB camera error: {exc}")
                self.enabled = False

    def close(self) -> None:
        self.enabled = False
        if self.renderer is not None:
            self.renderer.close()
        if CV2_AVAILABLE:
            cv2.destroyWindow(self.window_name)


class RGBDCameraWindow(RGBCameraWindow):
    """Wrist-mounted RGB-D camera display.

    MuJoCo uses one fixed camera for both streams.  RGB-D is produced by
    rendering the color frame, then switching the same renderer to metric depth
    rendering for the aligned depth frame.
    """

    def __init__(
        self,
        model,
        camera_id: int,
        width: int = 640,
        height: int = 360,
        render_every_n: int = 8,
        window_name: str = "Wrist RGB-D Camera",
    ):
        super().__init__(
            model,
            camera_id,
            width=width,
            height=height,
            render_every_n=render_every_n,
            window_name=window_name,
        )
        cv2.resizeWindow(window_name, width * 2, height)

    def render_rgbd(self, data) -> tuple[np.ndarray, np.ndarray]:
        """Return aligned RGB and metric depth images from the wrist camera."""
        self.renderer.update_scene(data, camera=self.camera_id)
        try:
            self.renderer.disable_depth_rendering()
            rgb = self.renderer.render()
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
        finally:
            self.renderer.disable_depth_rendering()
        return rgb, depth

    @staticmethod
    def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 1e-4)
        if not np.any(valid):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        near, far = np.percentile(depth[valid], [3, 97])
        if far <= near + 1e-6:
            far = near + 1e-3
        scaled = np.clip((depth - near) / (far - near), 0.0, 1.0)
        depth_u8 = ((1.0 - scaled) * 255.0).astype(np.uint8)
        depth_u8[~valid] = 0
        return cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    def update(self, data, overlay_text: str | None = None) -> None:
        if not self.enabled:
            return
        try:
            rgb, depth = self.render_rgbd(data)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_bgr = self._depth_to_bgr(depth)

            if overlay_text:
                cv2.putText(
                    bgr, overlay_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )
            cv2.putText(
                depth_bgr, "Depth", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(self.window_name, np.hstack((bgr, depth_bgr)))
            self.frame_count += 1
        except Exception as exc:
            if self.enabled:
                print(f"RGB-D camera error: {exc}")
                self.enabled = False
