from typing import Any

import numpy as np
import torch

from lerobot.cameras.camera import Camera
from lerobot.cameras.realsense.configuration_virtual_depth import VirtualDepthCameraConfig


def compute_depthmap(img: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """
    Placeholder for computing a depth map from a single camera image.

    Args:
        img: Image from the camera.
            Can be a numpy array (e.g., from opencv/record) or a torch Tensor.

    Returns:
        depth_map: The computed depth map, in the same format (numpy array or torch Tensor).
    """
    # TODO: Implement the actual depth map extraction logic here.
    # Returning a dummy depth map filled with zeros with the same height and width.

    if isinstance(img, torch.Tensor):
        # Assuming shape is (B, C, H, W) or (C, H, W)
        shape = list(img.shape)
        # Depth map usually has 1 channel instead of 3
        shape[-3] = 1
        return torch.zeros(shape, dtype=img.dtype, device=img.device)
    else:
        # Assuming numpy array of shape (H, W, C)
        shape = list(img.shape)
        if len(shape) == 3:
            shape[2] = 1
        return np.zeros(shape, dtype=img.dtype)


class VirtualDepthCamera(Camera):
    """
    A virtual camera that computes a depth map by reading from one other physical camera.
    """

    def __init__(self, config: VirtualDepthCameraConfig, robot=None, cameras=None):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        # We need a reference to the robot to access the other cameras
        self.robot = robot
        self.cameras_dict = cameras

    def _get_src_camera(self):
        src_cam = None

        if self.cameras_dict is not None:
            src_cam = self.cameras_dict.get(self.config.source_camera_name)
        elif self.robot is not None:
            src_cam = self.robot.cameras.get(self.config.source_camera_name)

        if not src_cam:
            raise RuntimeError(f"Source camera '{self.config.source_camera_name}' not found.")

        return src_cam

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, warmup: bool = True) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    def read(self):
        if not self._is_connected:
            raise RuntimeError("VirtualDepthCamera is not connected.")

        src_cam = self._get_src_camera()

        if hasattr(src_cam, "read_depth"):
            # RealSense natively supports hardware depth
            try:
                # Add an extra axis to keep things [1, H, W] or [H, W, 1] consistent
                depth_map = src_cam.read_depth()
                if len(depth_map.shape) == 2:
                    depth_map = np.expand_dims(depth_map, axis=-1)
                return depth_map
            except Exception as e:
                # Fallback if depth stream is not enabled on the RealSense camera
                print(f"Warning: Failed to read hardware depth from {self.config.source_camera_name}: {e}")

        # Fallback to computing depth from color image
        img = src_cam.read()
        return compute_depthmap(img)

    def async_read(self, timeout_ms: float = 200.0):
        if not self._is_connected:
            raise RuntimeError("VirtualDepthCamera is not connected.")

        src_cam = self._get_src_camera()

        if hasattr(src_cam, "read_depth"):
            try:
                # Note: RealSenseCamera `read_depth` ignores timeout_ms originally,
                # but we'll call it to get the latest depth map.
                # (You may want to implement async_read_depth internally inside RealSenseCamera)
                depth_map = src_cam.read_depth()
                if len(depth_map.shape) == 2:
                    depth_map = np.expand_dims(depth_map, axis=-1)
                return depth_map
            except Exception as e:
                import logging

                logging.warning(
                    f"Failed to async_read hardware depth from {self.config.source_camera_name}: {e}"
                )

        # Try to read the latest frame asynchronously
        img = src_cam.async_read(timeout_ms)

        depth_map = compute_depthmap(img)
        return depth_map

    def read_latest(self, max_age_ms: int = 1000):
        if not self._is_connected:
            raise RuntimeError("VirtualDepthCamera is not connected.")

        src_cam = self._get_src_camera()

        if hasattr(src_cam, "latest_depth_frame") and src_cam.latest_depth_frame is not None:
            depth_map = src_cam.latest_depth_frame
            if len(depth_map.shape) == 2:
                depth_map = np.expand_dims(depth_map, axis=-1)
            return depth_map

        # Try to peek at the latest frame
        img = src_cam.read_latest(max_age_ms)

        depth_map = compute_depthmap(img)
        return depth_map

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # Virtual cameras aren't "detected" via hardware
        return []
