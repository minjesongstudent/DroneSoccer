"""Physical measurements of the real setup. Nothing else lives here.

Every value carries one tag:
    MEASURED    taken from the real setup (with date)
    UNMEASURED  a guess; the comment says who replaces it and how
    DESIGN      a choice we made, not a property of the world

Search src/ for UNMEASURED to see what is still guessed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
    thickness: float = 0.065  # MEASURED 2026-10-01: tape, 6.5 cm
    n_points: int = 32                # DESIGN: points projected around the outer edge
    # The front marker: a disc on the noodle's front face at 6 o'clock. The noodle
    # hides it from behind, which is how the policy tells front from back.
    marker_max_angle_deg: float = 60.0  # MEASURED 2026-10-01: by eye
    # The hanger: the bar on top of the hoop and the straps up to the cage roof.
    # Modelled as a thin strip in the hoop's plane from its top to the ceiling,
    # so the drone cannot go round over the top. Laps only.
    hanger_half_width: float = 0.08  # MEASURED 2026-10-01: tape, 16 cm / 2
    hanger_half_width_range: tuple = (0.025, 0.20)  # DESIGN: per episode, a 5-40 cm bar; the
                                                    # real one varies between hoops


@dataclass
class Camera:
    """Intrinsics at the Tello's 960x720 stream: defaults, not calibrated."""
    width: int = 960           # MEASURED 2026-10-01: stream frames are 960x720
    height: int = 720          # MEASURED 2026-10-01
    fx: float = 920.0     # DEFAULT: typical Tello at 960x720; Randomise.error covers the real one
    fy: float = 920.0     # DEFAULT: typical Tello at 960x720; Randomise.error covers the real one
    cx: float = 480.0     # DEFAULT: typical Tello at 960x720; Randomise.error covers the real one
    cy: float = 360.0     # DEFAULT: typical Tello at 960x720; Randomise.error covers the real one
    pitch_deg: float = 12.8  # MEASURED 2026-10-01: labcheck tilt at 0.8 m
    offset_forward: float = 0.045  # MEASURED 2026-10-01: tape, 4.5 cm

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
    # Measurement error: each value here moves by up to +- this much per episode,
    # so the policy does not lean on any one measurement being exact. DESIGN, sized
    # to how well each was measured (tape about 1 cm, calibration about 3 px).
    error: dict = field(default_factory=lambda: {
        "cage.length": 0.02, "cage.width": 0.02, "cage.height": 0.02,
        "hoop.thickness": 0.01,
        "hoop.marker_max_angle_deg": 10.0,
        "camera.fx": 30.0, "camera.fy": 30.0, "camera.cx": 20.0, "camera.cy": 20.0,
        "camera.pitch_deg": 2.0, "camera.offset_forward": 0.01,
        "drone.width": 0.01, "drone.length": 0.01, "drone.height": 0.01,
        "drone.takeoff_height": 0.05,
        "drone.max_speed": 0.04, "drone.max_yaw_rate_deg": 6.0,
    })
