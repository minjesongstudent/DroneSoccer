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
    margin: float = 0.15       # DESIGN: hoop and way round kept this far in from the net. 0.25
                               # put the pad start outside it
    min_altitude: float = 0.5  # MEASURED 2026-10-01: holds pad x/y to 1 cm at tof 50 cm; lower untested


@dataclass
class Hoop:
    outer_diameter: float = 0.44      # width. MEASURED 2026-10-12: flight video (labcheck 183046, 183807,
                                      # 103 frames), the fitted outer edge with the 6.5 cm noodle as the
                                      # ruler, 0.43-0.46. 0.45 by tape on 2026-09-24, direction not noted
    outer_height: float = 0.48        # MEASURED 2026-10-12: flight video as outer_diameter, 0.46-0.51. The
                                      # tape gave 0.46 on 2026-10-06: the noodle hangs from its top and sags
    diameter_tolerance: float = 0.03  # DESIGN: +- per episode on the width; the noodle is floppy
    # Per episode the height is the width times outer_height / outer_diameter (1.09), times 1 +- this:
    # height/width 0.98 (round) to 1.20. MEASURED 2026-10-12 in the videos: 1.06-1.13, median 1.09,
    # and the top, where the ends are taped to the hanger, is narrower than an ellipse's.
    stretch_tolerance: float = 0.1    # DESIGN
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
                               # fb and lr (0.32-0.41 m/s at rc 45, 1 s steps: may under-read, as ud did)
    max_up_speed: float = 0.4  # DESIGN: m/s at full up or down; labcheck.RC_FULL ud is set so the drone matches.
                               # MEASURED 2026-10-06 (labcheck survey, 3 s holds): 0.4 m/s at rc 51 up, 47 down,
                               # so RC_FULL ud 50. Not linear: rc 30 gives 0.18 up / 0.13 down (sim 0.24)
    up_exponent: float = 1.0   # DESIGN: up/down speed = max_up_speed * |command| ** this, 1 linear.
                               # The real drone is weaker at small commands: Randomise.up_exponent
    max_yaw_rate_deg: float = 55.0  # MEASURED 2026-10-06 (labcheck survey): rc 70 (RC_FULL) turns 52.9-57.6 deg/s
    yaw_deadband: float = 0.10  # MEASURED 2026-10-06 (labcheck survey): rc 5 turns 0-1 deg/s, rc 10 3.4-4.8,
                                # rc 20 11, rc 40 25; the line through them crosses 0 at rc 7 = 0.10 of rc 70
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
class Detector:
    """What the real detector (src/detect.py) reports, which the simulator copies."""
    # Share of the hoop's rim that must be in frame before the detector reports it (half the
    # time). MEASURED 2026-10-06, and it depends on the edge: in flight ~0.8 leaving the left
    # edge (labcheck survey 20261006-184318, slide left: 22% at 0.7-0.8, 71% at 0.8-0.9) and
    # ~0.5-0.65 leaving the top (the policy flights); offline, recorded frames slid off every
    # edge, 0.58. So 0.7, randomised +-0.15. The right and top slides of that survey did not
    # give clean data (UNMEASURED in flight). The detector's own first gate,
    # contract.MIN_ARC_FRACTION (0.25), stays; its ring tests reject most partial arcs above it.
    min_arc_fraction: float = 0.7
    # The hoop must be fitted in this many consecutive frames before it counts. DESIGN, the
    # same as detect.CONFIRM_FRAMES; the live loop runs it at 10 Hz, one frame per step.
    confirm_frames: int = 2
    # Share of frames with the whole hoop in view, front side, where the marker is not
    # reported, with the detector as fixed on 2026-10-06 (detect.MARKER_WHITE_MAX_S 60 then
    # 100; before it, 39% at ~1 m and 98% closer than 0.8 m). MEASURED 2026-10-06 offline
    # only (labcheck markercheck on survey 20261006-184318): 16% at 0.8 m and further (44
    # frames), 9% closer (55 frames). UNMEASURED in flight with the fix, beyond 25 degrees
    # off the axis, and from behind: labcheck survey (and --behind) replaces these.
    marker_miss: float = 0.16
    marker_miss_near: float = 0.09
    near_major: float = 0.56    # hoop this big (0.8 m from a 0.45 m hoop) or bigger: marker_miss_near


@dataclass
class Altimeter:
    """The height the policy gets: the Tello's downward tof after labcheck.fly_policy has
    dropped its glitches. The sim's height is the drone's, and tof is what takeoff_height
    was measured with, so a perfect reading would equal it."""
    offset: float = 0.0       # UNMEASURED: reading minus true height, m. The pad, the sensor's place on
                              # the body and tilt shift it; Randomise.error covers +-5 cm. labcheck holds at
                              # taped heights replace it
    noise: float = 0.02       # UNMEASURED: std of one reading, m. tof "swings wildly when the drone
                              # tilts" (labcheck notes). Randomised 0-4 cm; labcheck hover logs replace it
    resolution: float = 0.10  # UNMEASURED: step between readings, m. The SDK reports whole cm, but the
                              # docs call the height low resolution; Randomise.altimeter_resolution
                              # spans 1-10 cm until labcheck state.csv shows the real step


@dataclass
class Telemetry:
    """The motion the policy gets (contract.MOTION_FIELDS): the Tello's state packet, read
    each tick by labcheck.fly_policy. sim.HoopEnv copies its faults."""
    speed_step: float = 0.1     # MEASURED 2026-10-12: vgx, vgy are whole dm/s; at hover 95% read 0
    speed_noise: float = 0.045  # MEASURED 2026-10-12: std, m/s, on each axis before its step: 5% of
                                # hover packets (rc < 5) read 0.1 or more, which 0.045 gives with
                                # each axis cut towards 0 (2.6% an axis reach 0.1, 5.2% either)
    stale: float = 0.02         # MEASURED 2026-10-12: share of ticks with no new packet (gap > 0.15 s)
    lag_steps: int = 1          # UNMEASURED: steps the packet trails the motion; Randomise.telemetry_lag
    yaw_step_deg: float = 1.0   # MEASURED 2026-10-12: yaw is whole degrees
    yaw_drift_deg_per_s: float = 0.0  # UNMEASURED: IMU heading drift; Randomise.yaw_drift spans +-1/min


@dataclass
class Randomise:
    """Ranges drawn per episode when HoopEnv(randomise=True). Stage 3 of the laps plan."""
    delay_steps: tuple[int, int] = (1, 4)       # MEASURED 2026-10-01: 2 steps, +-1 plus WiFi spikes
    tau: tuple[float, float] = (0.25, 0.7)      # MEASURED 2026-10-01: 0.25 (ud) to 0.66 (fb)
    rate_hz: tuple[float, float] = (8.0, 12.0)  # MEASURED 2026-10-01: 10.0 Hz mean, worst tick 126 ms; drawn every step
    altimeter_resolution: tuple[float, float] = (0.01, 0.10)  # DESIGN: per episode, see Altimeter.resolution
    telemetry_lag: tuple[int, int] = (0, 2)    # DESIGN: per episode, see Telemetry.lag_steps
    yaw_drift: tuple[float, float] = (-1 / 60, 1 / 60)   # DESIGN: deg/s per episode, see Telemetry
    # Top speed and the up/down response, per episode, never beyond what the 2026-10-12 policy
    # flights showed. MEASURED 2026-10-12 (8 flights):
    #   - run ins reached 0.5-0.8 m/s (telemetry) at rc 37-50, against the sim's 0.4 at full
    #     command, and after both passes the drone flew 2.1x as far as the sim does on the same
    #     commands, into the net. 0.55 to 0.8 (the user's choice, 2026-10-12): the run-in speeds
    #     seen, 0.8 the fastest. Drone.max_speed 0.4 (1 s steps, 2026-10-01) is below it.
    #   - up/down over 830 steady, level windows: rc 6-30 gave 0.03-0.11 m/s against the
    #     sim's 0.08-0.18, and rc 2-6 nothing: 0.4 * |command| ** 1.6 fits every bin to ~0.01.
    #     So 1 (the old linear sim) to 1.6. The 2026-10-06 survey agrees: rc 30 gave 0.13-0.18.
    # Both from flight logs, not steps: labcheck steps --axis ud/fb --seconds 3 would pin them.
    max_speed: tuple[float, float] = (0.55, 0.8)
    up_exponent: tuple[float, float] = (1.0, 1.6)
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
        "drone.max_yaw_rate_deg": 6.0,   # max_speed: Randomise.max_speed
        # Up/down is not linear in rc (MEASURED 2026-10-06), so +-20%; deadband 0.07-0.13.
        "drone.max_up_speed": 0.08, "drone.yaw_deadband": 0.03,
        "detector.min_arc_fraction": 0.15, "detector.marker_miss": 0.1, "detector.marker_miss_near": 0.08,
        "altimeter.offset": 0.05, "altimeter.noise": 0.02,
    })
