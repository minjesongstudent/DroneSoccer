"""Physical measurements of the real setup. Nothing else lives here.

Every value carries one tag:
    MEASURED    taken from the real setup (with date)
    UNMEASURED  a guess; the comment says who replaces it and how
    DESIGN      a choice we made, not a property of the world

Search src/ for UNMEASURED to see what is still guessed.
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
    min_altitude: float = 0.5  # MEASURED 2026-10-01: holds pad x/y to 1 cm at tof 50 cm; lower untested


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
    # The hanger: the bar on top of the hoop and the straps up to the cage roof.
    # Modelled as a thin strip in the hoop's plane from its top to the ceiling,
    # so the drone cannot go round over the top. Laps only.
    hanger_half_width: float = 0.2    # UNMEASURED: half the top bar's length; assumes the hoop
                                      # hangs from the roof. Measure the bar with the tape


@dataclass
class Camera:
    """Intrinsics at the Tello's 960x720 stream. calib.npz replaces these when it exists."""
    width: int = 960           # MEASURED 2026-10-01: stream frames are 960x720
    height: int = 720          # MEASURED 2026-10-01
    fx: float = 908.9  # MEASURED 2026-10-01: chessboard, 20 views, rms 0.57 px
    fy: float = 909.6  # MEASURED 2026-10-01: chessboard, 20 views, rms 0.57 px
    cx: float = 488.7  # MEASURED 2026-10-01: chessboard, 20 views, rms 0.57 px
    cy: float = 356.0  # MEASURED 2026-10-01: chessboard, 20 views, rms 0.57 px
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
    max_speed: float = 0.4     # DESIGN: m/s at full command. MEASURED 2026-10-01: rc ~50 gives it on
                               # fb and lr (0.32-0.41 m/s at rc 45); ud only reaches 0.26 m/s at rc 60
    max_yaw_rate_deg: float = 60.0  # DESIGN. MEASURED 2026-10-01: rc 50 = 40, rc 100 = 80-92 deg/s, so rc ~70
    tau: float = 0.55          # MEASURED 2026-10-01: fb/lr steps at rc 45 (0.52-0.66 s); ud 0.25 s
    delay_steps: int = 2       # MEASURED 2026-10-01: command to camera 150-200 ms (yaw steps, video)
    dt: float = 0.1            # DESIGN: 10 Hz control. MEASURED 2026-10-01: laptop holds 10.0 Hz
    takeoff_height: float = 0.8  # MEASURED 2026-10-01: tof 79-84 cm after settling


@dataclass
class Pad:
    """The take-off pad sits in a corner of the cage, facing in."""
    inset: float = 0.25              # DESIGN: pad centre from each of the two walls
    heading_error_deg: float = 15.0  # DESIGN: +- how far off the cage's centre it may face


@dataclass
class Randomise:
    """Ranges drawn per episode when HoopEnv(randomise=True). Stage 3 of the laps plan."""
    delay_steps: tuple[int, int] = (1, 4)       # MEASURED 2026-10-01: 2 steps, +-1 plus WiFi spikes
    tau: tuple[float, float] = (0.25, 0.7)      # MEASURED 2026-10-01: 0.25 (ud) to 0.66 (fb)
    rate_hz: tuple[float, float] = (8.0, 12.0)  # MEASURED 2026-10-01: 10.0 Hz mean, worst tick 126 ms; drawn every step


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
