"""Physical measurements of the real setup. Nothing else lives here.

Every value carries one tag:
    MEASURED    taken from the real setup (with date)
    UNMEASURED  a guess; the comment says who replaces it and how
    DESIGN      a choice we made, not a property of the world

Search for UNMEASURED to see what is still guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = ROOT / "calib.npz"


@dataclass
class Cage:
    """Axis-aligned box in metres, origin at a floor corner."""
    length: float = 2.05       # MEASURED 2026-09-24
    width: float = 2.05        # MEASURED 2026-09-24
    height: float = 2.05       # MEASURED 2026-09-24
    margin: float = 0.15       # DESIGN: geofence inset from the net on every side. 0.25 put
                               # the pad start on the fence; the drone still clears the net by ~7 cm
    min_altitude: float = 0.3  # DESIGN: the downward sensor needs this to hold position


@dataclass
class Hoop:
    outer_diameter: float = 0.45      # MEASURED 2026-09-24, outside edge to outside edge
    diameter_tolerance: float = 0.03  # DESIGN: +- per episode and per axis; the noodle is floppy
    thickness: float = 0.065          # UNMEASURED: noodle diameter, measure with the tape
    n_points: int = 32                # DESIGN: points projected around the outer edge
    # The front marker: a disc on the noodle's front face at 6 o'clock. The noodle
    # hides it from behind, which is how the policy tells front from back.
    marker_max_angle_deg: float = 70.0  # UNMEASURED: widest view off the hoop's front normal
                                        # that still shows it; check by eye from the side


@dataclass
class Camera:
    """Intrinsics at the Tello's 960x720 stream. calib.npz replaces these when it exists."""
    width: int = 960
    height: int = 720
    fx: float = 920.0          # UNMEASURED: stage 0 chessboard calibration
    fy: float = 920.0          # UNMEASURED
    cx: float = 480.0          # UNMEASURED
    cy: float = 360.0          # UNMEASURED
    pitch_deg: float = 0.0     # UNMEASURED: camera tilt below horizontal
    offset_forward: float = 0.05  # UNMEASURED: camera ahead of the drone's centre

    @property
    def half_extent(self) -> tuple[float, float]:
        return self.width / (2 * self.fx), self.height / (2 * self.fy)


@dataclass
class Drone:
    # Collision box, prop guards included.
    width: float = 0.17        # MEASURED 2026-09-24, side to side
    height: float = 0.035      # MEASURED 2026-09-24
    length: float = 0.105      # MEASURED 2026-09-24, from the 20 cm diagonal: sqrt(0.20^2 - 0.17^2)
    max_speed: float = 0.4     # DESIGN: m/s at full command
    max_yaw_rate_deg: float = 60.0  # DESIGN
    tau: float = 0.3           # UNMEASURED: velocity response time, B2 step flights
    delay_steps: int = 0       # UNMEASURED: command delay in control steps, B2
    dt: float = 0.1            # DESIGN: 10 Hz control; B2 checks the laptop holds it
    takeoff_height: float = 0.8  # UNMEASURED: hover height after `takeoff`, B2


@dataclass
class Pad:
    """The take-off pad sits in a corner of the cage, facing in."""
    inset: float = 0.25              # DESIGN: pad centre from each of the two walls
    heading_error_deg: float = 15.0  # DESIGN: +- how far off the cage's centre it may face


@dataclass
class Randomise:
    """Ranges drawn per episode when HoopEnv(randomise=True). Stage 3 of the laps plan."""
    delay_steps: tuple[int, int] = (0, 3)       # UNMEASURED: B2, 100-300 ms expected
    tau: tuple[float, float] = (0.15, 0.5)      # UNMEASURED: B2 step flights
    rate_hz: tuple[float, float] = (8.0, 12.0)  # UNMEASURED: drawn every step, WiFi is not steady


def load_camera() -> Camera:
    """Camera from calib.npz if it exists, else the guessed one.

    calib.npz holds K (3x3), dist, and size (width, height) as saved after
    cv2.calibrateCamera. The simulator ignores dist: the detector undistorts,
    so both sides work in the same ideal pinhole coordinates.
    """
    if not CALIB_PATH.exists():
        return Camera()
    data = np.load(CALIB_PATH)
    K = data["K"]
    width, height = (int(x) for x in data["size"])
    return Camera(width=width, height=height,
                  fx=float(K[0, 0]), fy=float(K[1, 1]),
                  cx=float(K[0, 2]), cy=float(K[1, 2]))
