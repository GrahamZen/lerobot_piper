from dataclasses import dataclass

from ..configs import CameraConfig


@CameraConfig.register_subclass("virtual_depth")
@dataclass(kw_only=True)
class VirtualDepthCameraConfig(CameraConfig):
    # We will need the name of the camera to read from
    source_camera_name: str = "left"
    fps: int | None = 30
    width: int | None = 640
    height: int | None = 480
