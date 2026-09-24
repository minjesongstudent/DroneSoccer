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
    margin: float = 0.25       # DESIGN: geofence inset from the net on every side
    min_altitude: float = 0.3  # DESIGN: the downward sensor needs this to hold position


@dataclass
class Hoop:
    outer_diameter: float = 0.45      # MEASURED 2026-09-24, outside edge to outside edge
    diameter_tolerance: float = 0.03  # DESIGN: +- per episode and per axis; the noodle is floppy
    thickness: float = 0.065          # UNMEASURED: noodle diameter, measure with the tape
    n_points: int = 32                # DESIGN: points projected around the outer edge


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
