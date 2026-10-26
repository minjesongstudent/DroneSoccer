from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "calib.npz"


@dataclass
class Cage:
    length: float = 2.05
    width: float = 2.05
    height: float = 2.05
    margin: float = 0.15
    min_altitude: float = 0.5


@dataclass
class Hoop:
    outer_diameter: float = 0.44
    outer_height: float = 0.48
    diameter_tolerance: float = 0.03
    stretch_tolerance: float = 0.1
    thickness: float = 0.065
    n_points: int = 32
    marker_max_angle_deg: float = 60.0
    hanger_half_width: float = 0.08
    hanger_half_width_range: tuple = (0.025, 0.20)


@dataclass
class Camera:
    width: int = 960
    height: int = 720
    fx: float = 920.0
    fy: float = 920.0
    cx: float = 480.0
    cy: float = 360.0
    pitch_deg: float = 12.8
    offset_forward: float = 0.045

    @property
    def half_extent(self) -> tuple[float, float]:
        return self.width / (2 * self.fx), self.height / (2 * self.fy)


@dataclass
class Drone:
    width: float = 0.17
    height: float = 0.035
    length: float = 0.105
    max_speed: float = 0.4
    max_up_speed: float = 0.4
    up_exponent: float = 1.0
    max_yaw_rate_deg: float = 55.0
    yaw_deadband: float = 0.10
    tau: float = 0.55
    delay_steps: int = 2
    dt: float = 0.1
    takeoff_height: float = 0.8


@dataclass
class Pad:
    inset: float = 0.25
    heading_error_deg: float = 15.0


@dataclass
class Detector:
    min_arc_fraction: float = 0.7
    confirm_frames: int = 2
    marker_miss: float = 0.16
    marker_miss_near: float = 0.09
    near_major: float = 0.56


@dataclass
class Altimeter:
    offset: float = 0.0
    noise: float = 0.02
    resolution: float = 0.10


@dataclass
class Telemetry:
    speed_step: float = 0.1
    speed_noise: float = 0.045
    stale: float = 0.02
    lag_steps: int = 1
    yaw_step_deg: float = 1.0
    yaw_drift_deg_per_s: float = 0.0


@dataclass
class Randomise:
    delay_steps: tuple[int, int] = (1, 4)
    tau: tuple[float, float] = (0.25, 0.7)
    rate_hz: tuple[float, float] = (8.0, 12.0)
    altimeter_resolution: tuple[float, float] = (0.01, 0.10)
    telemetry_lag: tuple[int, int] = (0, 2)
    yaw_drift: tuple[float, float] = (-1 / 60, 1 / 60)
    max_speed: tuple[float, float] = (0.55, 0.8)
    up_exponent: tuple[float, float] = (1.0, 1.6)
    error: dict = field(default_factory=lambda: {
        "cage.length": 0.02, "cage.width": 0.02, "cage.height": 0.02,
        "hoop.thickness": 0.01,
        "hoop.marker_max_angle_deg": 10.0,
        "camera.fx": 30.0, "camera.fy": 30.0, "camera.cx": 20.0, "camera.cy": 20.0,
        "camera.pitch_deg": 2.0, "camera.offset_forward": 0.01,
        "drone.width": 0.01, "drone.length": 0.01, "drone.height": 0.01,
        "drone.takeoff_height": 0.05,
        "drone.max_yaw_rate_deg": 6.0,
        "drone.max_up_speed": 0.08, "drone.yaw_deadband": 0.03,
        "detector.min_arc_fraction": 0.15, "detector.marker_miss": 0.1, "detector.marker_miss_near": 0.08,
        "altimeter.offset": 0.05, "altimeter.noise": 0.02,
    })
