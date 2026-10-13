"""Gymnasium environment: the drone flying at a hoop, rendering nothing.

The mission: the drone starts above the take-off pad in a corner and must find
the hoop, pass through it front to back, fly round the outside back to the front,
and pass again, for as long as it lasts.

The policy sees only the eight numbers from contract.py (stacked with its own
recent actions). The true hoop position is used for reward and termination only.
"""
from __future__ import annotations

# ----- START LAYOUT (logged into trainingReports/ by every run) -------------
# The reward, the actions, the state, the discount and the episode ends all live
# in contract.py. This file is only the physics.
# The curriculum level blends from easy (0) to real (1).
HOOP_HEIGHT = (1.33, 1.73)     # hoop centre above the floor: hung 10-50 cm below the 2.05 m roof
                               # (top of the hoop to the roof). UNMEASURED by tape: labcheck tape
HOOP_FROM_CENTRE = 0.15        # hoop centre this close to the middle of the cage, as set up for real
PAD_MIN_DISTANCE = 0.6         # pad to hoop centre
LAPS_HOOP_ANGLE_DEG = (30.0, 75.0)  # max hoop angle to the pad, level 0 and 1
CLEARANCE = 0.4                # free space in front of and behind the hoop
SIDE_GAP = 0.35                # side waypoint's distance beyond the rim
# Run-in starts (HoopEnv(run_in_share=...), training only): lined up in front of the
# hoop, at the height that puts it mid-frame, so the last blind half metre of a pass
# is practised far more often than the whole mission reaches it. In the 2026-10-09
# flights the policy stopped 0.55-0.6 m out, lined up, and circled instead.
RUN_IN_DISTANCE = (0.6, 1.0)   # m in front of the hoop centre
RUN_IN_OFFSET = 0.05           # m, +- off the axis
RUN_IN_HEADING_DEG = 5.0       # +- off facing the hoop
RUN_IN_HEIGHT = (-0.3, 0.1)    # m from the height that puts the hoop mid-frame: mostly low,
                               # as the drone came in on 2026-10-09 (hoop high in frame). From
                               # 0.3 m low teacher2-level1 passed 42/60 with the flag on, 1/60 off
# -----------------------------------------------------------------------------

import dataclasses
from collections import deque

import gymnasium as gym
import numpy as np

from src import contract
from src.params import Altimeter, Cage, Camera, Detector, Drone, Hoop, Pad, Randomise, Telemetry

# Collision checks use this many points along the noodle's centreline. At ~5 mm
# spacing, a contact can be missed by at most ~0.1 mm.
CENTRELINE_POINTS = 256

# HoopEnv.privileged pads the commands still in flight to this many. Above
# Randomise.delay_steps' top, so every draw fits.
PENDING_MAX = 6
# The episode's draw of the setup in HoopEnv.privileged(), in order, from DYNAMICS_START
# on: after the hoop-frame position and velocity (3 + 3), facing and yaw rate (3), the
# position over the cage and the cage (3 + 3) and the six layout numbers.
DYNAMICS_FIELDS = ("delay_steps", "tau", "max_speed", "max_up_speed", "up_exponent", "max_yaw_rate_deg",
                   "yaw_deadband", "telemetry_lag", "yaw_drift", "camera_pitch_deg", "altimeter_offset",
                   "height_step")
DYNAMICS_START = 3 + 3 + 3 + 3 + 3 + 6
PRIVILEGED_DIM = DYNAMICS_START + len(DYNAMICS_FIELDS) + PENDING_MAX * contract.ACTION_DIM
# A gated teacher (train.py --privileged-actor gated) reads the true state only as far as
# a student could know it from its own input (HoopEnv.gated). teacher4 read all of it, and
# every seed flew its blind stretches on where the hoop truly was: held at the last
# sighting, seed 0 fell from 6.85 to 1.42 passes a flight and seed 1 (teacher4.zip, which
# never saw the hoop) to 0.02, and seed 1's clone crashed every flight.
#   GATE_HELD: where the drone is and how it moves in the hoop's frame, and the hoop's
#     height and size. Live while the camera reports the hoop, held at the last reported
#     value while it does not, 0 before the first report.
#   GATE_NEVER: where it is over the cage, the cage, and the metres to go round the lap
#     (from the true position). Always 0.
#   The rest (the not-yet-through and front-found flags, the dynamics and the commands
#   in flight) the student can work out from its readings and its own commands: kept.
GATE_HELD = np.r_[0:9, 15:18]
GATE_NEVER = np.r_[9:15, 18]
GATE_FACING = (6, 7)           # sin and cos of the heading off the hoop's axis, in GATE_HELD
GATED_DIM = PRIVILEGED_DIM + 2   # then: reported yet this episode, time since the last report
GATE_SINCE_MAX = contract.HISTORY_STEPS   # steps; the time since saturates at 1 here
# While the hoop is reported, the true numbers are sharper than the camera: seen nearly
# square on, a hoop a few degrees off the axis looks like one dead on, and teacher3's run
# in, which starts on true alignment within 3 cm and 3 degrees, left its clones hovering
# in front of the hoop. So the held numbers get an error of what the policy input leaves
# unresolved, by metres from the hoop centre (GATE_BLUR_DISTANCE, interpolated, flat past
# the ends). Measured 2026-10-13: an MLP from the whole policy input to each number, on
# 24k reported steps of 150 teacher3 flights (noise 0.1, full ranges), held-out RMS error.
# Rows follow GATE_HELD; the facing's error is GATE_BLUR_FACING, in degrees, so its sin
# and cos rows are 0. A draw is kept for the whole sighting, so it cannot be averaged
# away, and made afresh after GATE_REACQUIRE blind steps.
GATE_REACQUIRE = 5
GATE_BLUR_DISTANCE = (0.5, 0.7, 0.9, 1.15)   # m; the hoop was never reported past 1.3
GATE_BLUR = np.array([
    (0.011, 0.010, 0.014, 0.032),   # m along the axis
    (0.023, 0.016, 0.041, 0.173),   # m beside it
    (0.007, 0.006, 0.007, 0.012),   # m above it
    (0.021, 0.016, 0.017, 0.018),   # m/s, the same three ways
    (0.015, 0.014, 0.015, 0.021),
    (0.009, 0.007, 0.008, 0.013),
    (0.0, 0.0, 0.0, 0.0),           # sin and cos of the facing: GATE_BLUR_FACING
    (0.0, 0.0, 0.0, 0.0),
    (0.027, 0.023, 0.022, 0.025),   # rad/s, yaw rate
    (0.017, 0.012, 0.013, 0.020),   # m, hoop centre's height
    (0.003, 0.002, 0.002, 0.004),   # m, hoop's half width and half height
    (0.004, 0.002, 0.003, 0.006),
])
GATE_BLUR_FACING = (1.6, 1.2, 2.7, 10.8)   # degrees

def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def velocity_command(d: Drone, right, forward, up, yaw_cw, heading: float):
    """The velocity (world x, y, z in m/s) and turn rate (rad/s, counter-clockwise) a
    command asks of drone d facing heading: what HoopEnv.step eases towards with tau.
    A function of its own so src.sysid fits real logs with the same equations."""
    c, s = np.cos(heading), np.sin(heading)
    # No faster than max_speed in any direction: on 2026-10-12 full diagonals (rc lr 50, fb
    # -45) flew 0.5-0.65 m/s, slower than straight ahead (0.8), not 1.41 times it.
    across = max(1.0, float(np.hypot(forward, right)))
    forward, right = forward / across, right / across
    # Up and down are weak at small commands (params.Randomise.up_exponent).
    up_speed = d.max_up_speed * np.sign(up) * abs(up) ** d.up_exponent
    v_cmd = np.array([d.max_speed * (forward * c + right * s), d.max_speed * (forward * s - right * c),
                      up_speed])
    # Small yaw commands barely turn the real drone: scale from the deadband up.
    db = float(np.clip(d.yaw_deadband, 0.0, 0.5))
    yaw_cw = np.sign(yaw_cw) * max(0.0, abs(yaw_cw) - db) / (1.0 - db)
    return v_cmd, -yaw_cw * np.deg2rad(d.max_yaw_rate_deg)   # heading is counter-clockwise


class HoopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: float = 1.0, randomise: bool = False, hanger: bool = True,
                 cage: Cage | None = None, hoop: Hoop | None = None,
                 camera: Camera | None = None, drone: Drone | None = None,
                 pad: Pad | None = None, ranges: Randomise | None = None,
                 detector: Detector | None = None, altimeter: Altimeter | None = None,
                 telemetry: Telemetry | None = None, run_in_share: float = 0.0):
        self.cage = cage or Cage()
        self.hoop = hoop or Hoop()
        self.camera = camera or Camera()
        self.drone = drone or Drone()
        self.pad = pad or Pad()
        self.detector = detector or Detector()
        self.altimeter = altimeter or Altimeter()
        self.telemetry = telemetry or Telemetry()
        self.ranges = ranges or Randomise()
        self.randomise = randomise
        self.run_in_share = run_in_share   # share of resets that start lined up (RUN_IN_*)
        self.run_in = False
        self.hanger = hanger   # off replays models trained before it existed
        # The measured setup. With randomise, each episode works on a copy with
        # ranges.error applied; without, on these.
        self._measured = {"cage": self.cage, "hoop": self.hoop, "camera": self.camera, "drone": self.drone,
                          "detector": self.detector, "altimeter": self.altimeter}
        self.set_level(level)

        self.action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (contract.POLICY_INPUT_DIM,), np.float32)

        phi = np.linspace(0.0, 2 * np.pi, self.hoop.n_points, endpoint=False)
        self._cos_phi, self._sin_phi = np.cos(phi), np.sin(phi)
        phi = np.linspace(0.0, 2 * np.pi, CENTRELINE_POINTS, endpoint=False)
        self._cos_line, self._sin_line = np.cos(phi), np.sin(phi)
        self._derive()

    def _derive(self):
        """Values worked out from the setup, redone when measurement error changes it."""
        self._marker_cos = np.cos(np.deg2rad(self.hoop.marker_max_angle_deg))
        pitch = np.deg2rad(self.camera.pitch_deg)
        self._cos_pitch, self._sin_pitch = np.cos(pitch), np.sin(pitch)
        # Half-sizes of the drone's collision box in the body frame (forward, left, up).
        d = self.drone
        self._half_box = np.array([d.length, d.width, d.height]) / 2

    def _apply_measurement_error(self):
        """Per episode, move each measured value by up to +- ranges.error."""
        changes: dict = {name: {} for name in self._measured}
        for key, error in self.ranges.error.items():
            name, attr = key.split(".")
            base = getattr(self._measured[name], attr)
            changes[name][attr] = base + self.np_random.uniform(-error, error)
        for name, measured in self._measured.items():
            setattr(self, name, dataclasses.replace(measured, **changes[name]))
        self._derive()

    def set_level(self, level: float):
        """Curriculum level for starts, 0 (easy) to 1 (the real layout)."""
        self.level = float(np.clip(level, 0.0, 1.0))

    # --- episode setup ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # The gate's blur (gated()) has its own generator, so the sim's own draws are the
        # same with or without a gated teacher.
        if seed is not None or not hasattr(self, "_gate_rng"):
            self._gate_rng = np.random.default_rng([0 if seed is None else seed, 1])
        if self.randomise:
            self._apply_measurement_error()
        self._sample_layout()
        # Only draw when asked, so evaluation layouts stay the same.
        self.run_in = bool(self.run_in_share) and self.np_random.uniform() < self.run_in_share
        if self.run_in:
            self._line_up_start()
        obs = self._start_episode()
        self._found = self.run_in   # nothing to find: it starts looking at it
        return obs, self._info("running")

    def _line_up_start(self):
        """Move the drone onto the hoop's axis RUN_IN_DISTANCE in front, facing it,
        RUN_IN_HEIGHT from the height that keeps the hoop mid-frame. Nearer if that is
        past the fence."""
        rng, n = self.np_random, self.hoop_normal
        out = rng.uniform(*RUN_IN_DISTANCE)
        side = rng.uniform(-RUN_IN_OFFSET, RUN_IN_OFFSET)
        while True:
            p = self.hoop_centre - out * n + side * self._axis_h
            if self._in_fence(p[:2]) or out <= CLEARANCE:
                break
            out -= 0.05
        p[2] = np.clip(self.hoop_centre[2] + np.tan(np.deg2rad(self.camera.pitch_deg)) * out
                       + rng.uniform(*RUN_IN_HEIGHT),
                       self.cage.min_altitude, self.cage.height - self.cage.margin)
        self.pos = p
        self.heading = _wrap(np.arctan2(n[1], n[0])
                             + np.deg2rad(rng.uniform(-RUN_IN_HEADING_DEG, RUN_IN_HEADING_DEG)))

    def place(self, drone_pos, heading, hoop_centre, hoop_normal, hoop_diameter=None,
              hanger_half_width=None):
        """Set an exact layout instead of a random one. For tests and replaying real flights.

        hoop_normal points out of the hoop's back: the front, with the marker,
        faces the other way. hoop_diameter is one outer diameter, or
        (horizontal, vertical) for a squashed hoop. hanger_half_width defaults to
        the measured bar.
        """
        diameter = (self.hoop.outer_diameter, self.hoop.outer_height) if hoop_diameter is None else hoop_diameter
        radius_h, radius_v = np.broadcast_to(np.asarray(diameter, float), 2) / 2
        self._set_layout(np.asarray(drone_pos, float), float(heading),
                         np.asarray(hoop_centre, float), np.asarray(hoop_normal, float),
                         radius_h, radius_v,
                         self.hoop.hanger_half_width if hanger_half_width is None else hanger_half_width)
        self.run_in = False
        return self._start_episode()

    def _in_fence(self, xy) -> bool:
        m = self.cage.margin
        return bool(m <= xy[0] <= self.cage.length - m and m <= xy[1] <= self.cage.width - m)

    def _sample_layout(self):
        """The drone above the pad in a random corner, the hoop near the middle of the
        cage with room to go round on either side. The policy cannot see the net, so
        it cannot tell which side has room: both must."""
        rng, cage, pad, m = self.np_random, self.cage, self.pad, self.cage.margin
        max_angle = np.interp(self.level, [0.0, 1.0], LAPS_HOOP_ANGLE_DEG)
        middle = np.array([cage.length, cage.width]) / 2
        for _ in range(1000):
            start = np.array([rng.choice([pad.inset, cage.length - pad.inset]),
                              rng.choice([pad.inset, cage.width - pad.inset])])
            # Uniform over the disc of radius HOOP_FROM_CENTRE round the middle.
            r, a = HOOP_FROM_CENTRE * np.sqrt(rng.uniform()), rng.uniform(0.0, 2 * np.pi)
            centre = middle + r * np.array([np.cos(a), np.sin(a)])
            offset = centre - start
            if np.linalg.norm(offset) < PAD_MIN_DISTANCE:
                continue
            bearing = np.arctan2(offset[1], offset[0])
            # The normal points away from the pad, so the front faces it.
            angle = bearing + np.deg2rad(rng.uniform(-max_angle, max_angle))
            normal = np.array([np.cos(angle), np.sin(angle)])
            side = np.array([-normal[1], normal[0]])
            # The noodle hangs taller than wide by a varying amount: the width, then the
            # height from the width (params.Hoop.stretch_tolerance).
            wide, tall = rng.uniform(-1, 1, 2)
            width = self.hoop.outer_diameter + wide * self.hoop.diameter_tolerance
            stretch = self.hoop.outer_height / self.hoop.outer_diameter * (1 + tall * self.hoop.stretch_tolerance)
            radii = np.array([width, width * stretch]) / 2
            must_fit = [centre + radii[0] * side, centre - radii[0] * side,
                        centre + CLEARANCE * normal, centre - CLEARANCE * normal]
            gates = [centre + (radii[0] + SIDE_GAP) * side, centre - (radii[0] + SIDE_GAP) * side]
            if all(self._in_fence(p) for p in must_fit + gates):
                break
        else:
            raise RuntimeError("cage too small for the layout ranges in sim.py")

        hoop_z = rng.uniform(*HOOP_HEIGHT)
        drone_z = np.clip(np.interp(self.level, [0.0, 1.0], [hoop_z, self.drone.takeoff_height]),
                          cage.min_altitude, cage.height - m)
        # Level 0 faces the hoop; level 1 faces the cage centre, off by the pad's placement.
        to_centre = middle - start
        facing = (np.arctan2(to_centre[1], to_centre[0])
                  + np.deg2rad(rng.uniform(-pad.heading_error_deg, pad.heading_error_deg)))
        heading = _wrap(bearing + self.level * _wrap(facing - bearing))
        hanger_half = rng.uniform(*self.hoop.hanger_half_width_range)
        self._set_layout(np.array([*start, drone_z]), heading,
                         np.array([*centre, hoop_z]), np.array([*normal, 0.0]), *radii, hanger_half)

    def _set_layout(self, drone_pos, heading, hoop_centre, hoop_normal, radius_h, radius_v,
                    hanger_half_width):
        self.pos = drone_pos.copy()
        self.heading = heading
        self.hoop_centre = hoop_centre
        n = hoop_normal / np.linalg.norm(hoop_normal)
        self.hoop_normal = n
        # Unit axes in the hoop's plane: horizontal, and vertical.
        self._axis_h = np.array([-n[1], n[0], 0.0])
        self._axis_v = np.array([0.0, 0.0, 1.0])
        # Outer-edge ellipse, which the camera sees.
        self._hoop_h = radius_h * self._axis_h
        self._hoop_v = radius_v * self._axis_v
        # Noodle centreline, the same ellipse moved in by half the thickness.
        # Collisions and passes are scored against it.
        half_t = self.hoop.thickness / 2
        self._line_h, self._line_v = radius_h - half_t, radius_v - half_t
        self._centreline = (hoop_centre + np.outer(self._cos_line, self._line_h * self._axis_h)
                            + np.outer(self._sin_line, self._line_v * self._axis_v))
        # The hanger strip runs from the top of the hoop's outer edge to the ceiling.
        self._hanger_bottom = hoop_centre[2] + radius_v
        self.hanger_half_width = hanger_half_width
        # The marker sits on the noodle's front face at 6 o'clock.
        self._marker = hoop_centre - self._line_v * self._axis_v - half_t * n
        # What the camera projects each step: the outer edge, then the marker last.
        outline = (hoop_centre + np.outer(self._cos_phi, self._hoop_h)
                   + np.outer(self._sin_phi, self._hoop_v))
        self._sight_points = np.vstack([outline, self._marker])
        # No centreline point is further than this from the hoop centre, plus the
        # reach of the drone's box and the noodle's half thickness. Beyond it, no hit.
        self._hit_reach = max(self._line_h, self._line_v) + half_t + np.linalg.norm(self._half_box)
        # Side waypoints for the way round, at the hoop's height. Only those inside
        # the geofence count; a layout from place() may have none, so keep both then.
        gates = [hoop_centre + s * (radius_h + SIDE_GAP) * self._axis_h for s in (1, -1)]
        self._gates = [g for g in gates if self._in_fence(g)] or gates
        # The open hole inside the noodle, and the point out front to line up on.
        self._hole_h, self._hole_v = self._line_h - half_t, self._line_v - half_t
        self._approach = hoop_centre - CLEARANCE * n
        # Lined up: flying straight on along the axis, the box would clear the noodle.
        # The hole less the box's half-width beside and half-height above and below.
        self._safe_h = self._hole_h - self._half_box[1]
        self._safe_v = self._hole_v - self._half_box[2]
        # The most clearance any pass here can have: centred, facing straight through.
        self._best_clearance = self._clearance(hoop_centre, np.arctan2(n[1], n[0]))
        # Sideways from the centre, past which the drone clears the rim going round.
        self._beside = radius_h + SIDE_GAP / 2

    def _start_episode(self):
        # Randomised dynamics are drawn once per episode into a copy of the drone.
        self.dyn = self.drone
        self.height_step = self.altimeter.resolution
        self.telemetry_lag, self.yaw_drift = self.telemetry.lag_steps, self.telemetry.yaw_drift_deg_per_s
        if self.randomise:
            r, rng = self.ranges, self.np_random
            self.dyn = dataclasses.replace(
                self.drone,
                delay_steps=int(rng.integers(r.delay_steps[0], r.delay_steps[1] + 1)),
                tau=float(rng.uniform(*r.tau)),
                max_speed=float(rng.uniform(*r.max_speed)),
                up_exponent=float(rng.uniform(*r.up_exponent)))
            self.height_step = float(rng.uniform(*r.altimeter_resolution))
            self.telemetry_lag = int(rng.integers(r.telemetry_lag[0], r.telemetry_lag[1] + 1))
            self.yaw_drift = float(rng.uniform(*r.yaw_drift))
        self.vel = np.zeros(3)
        self.yaw_rate = 0.0
        self.steps = 0
        self.passes = 0
        self.wall_steps = 0
        self.route = None   # which way the drone last went round, for replay.py
        self.centred = None   # how centred this step's pass was (_centred), None without one
        self._pending = deque([np.zeros(contract.ACTION_DIM, np.float32)] * self.dyn.delay_steps)
        # As labcheck.fly_policy starts: no frames yet, so every step of the history
        # is blind with no command, and the height has held at the first reading. The
        # detector has to confirm the hoop before it counts, so a hoop in view at the
        # start is reported from the second step.
        n = contract.HISTORY_STEPS
        self._frames = deque([contract.NOT_VISIBLE.copy()] * n, maxlen=n)
        self._heights = deque([self._height_reading()] * n, maxlen=n)
        # Motion, as labcheck.fly_policy reads it: heading from where the policy started.
        self._start_heading, self._clock = self.heading, 0.0
        self._motion_true = deque([((0.0, 0.0), 0.0)] * (self.telemetry_lag + 1), maxlen=self.telemetry_lag + 1)
        self._last_motion = contract.NO_MOTION.copy()
        self._motion = deque([self._motion_reading()] * n, maxlen=n)
        self._streak = 0
        self._unseen = True
        self._found = False   # the front reported yet this episode, for R_FOUND
        self._actions = deque([np.zeros(contract.ACTION_DIM, np.float32)] * n, maxlen=n)
        # The gate (gated()): nothing reported yet. Tests start episodes without reset().
        if not hasattr(self, "_gate_rng"):
            self._gate_rng = np.random.default_rng([0, 1])
        self._held = np.zeros(len(GATE_HELD), np.float32)
        self._blur = np.zeros(len(GATE_HELD))
        self._reported, self._since = False, GATE_SINCE_MAX
        return self._policy_input()

    # --- stepping --------------------------------------------------------

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).copy()

        # Command delay: the drone acts on the command from delay_steps ago.
        self._pending.append(action)
        right, forward, up, yaw_cw = self._pending.popleft()

        # Commands are body-frame: rotate by heading before integrating.
        d = self.dyn
        dt = 1.0 / self.np_random.uniform(*self.ranges.rate_hz) if self.randomise else d.dt
        v_cmd, yaw_cmd = velocity_command(d, right, forward, up, yaw_cw, self.heading)
        alpha = 1.0 - np.exp(-dt / d.tau)
        self.vel += (v_cmd - self.vel) * alpha
        self.yaw_rate += (yaw_cmd - self.yaw_rate) * alpha

        prev = self.pos.copy()
        self.pos = self.pos + self.vel * dt
        self.heading = (self.heading + self.yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi
        self._clock += dt
        self.steps += 1
        on_wall, face = self._touch_walls()
        self.wall_steps += on_wall

        outcome = self._outcome(prev)
        if face and outcome == "running":
            outcome = face
        if outcome == "passed":
            self.passes += 1
        obs = self._as_detected(self.observe())
        if outcome == "running" and not self._found and obs[contract.VISIBLE] and obs[contract.FIELDS.index("marker")]:
            outcome, self._found = "found", True
        self._frames.append(obs)
        self._heights.append(self._height_reading())
        self._motion.append(self._motion_reading())
        self._unseen = contract.not_yet_through(self._unseen, self._frames)
        self._actions.append(action)
        self._update_gate(obs)

        terminated = outcome in contract.CRASHES
        reward = contract.reward(outcome, self._to_go(prev), self._to_go(self.pos),
                                 on_wall, obs.copy(), action.copy(), self.centred or 0.0)
        truncated = not terminated and self.steps >= contract.LAPS_MAX_STEPS
        if truncated:
            outcome = "timeout"
        return self._policy_input(), float(reward), terminated, truncated, self._info(outcome)

    def _to_go(self, pos) -> float:
        """Metres left round the lap to the next pass, along a route that keeps
        off the noodle. A straight line at the hoop centre from the side, or at
        a side gate from just behind the ring, runs into the rim.

        In front, lined up (a straight run along the axis would clear the noodle):
        straight to the centre.
        In front, not lined up: to the approach point CLEARANCE out front, then in.
        Inside the hole but too near the rim to clear it counts as not lined up, so
        cutting in from the side towards the rim earns nothing. Between the safe edge
        and the hole's edge the two blend, so drifting a little off-centre close to
        the plane costs a little, not the whole detour in one step.
        Behind, still within the rim's width: out to a point CLEARANCE behind a
        side gate, so the drone backs off the rim. Beside the rim: straight to
        the gate, then round to the approach point.
        Crossing the plane outside the ring never raises the distance to go, so
        the only jump up is at a pass, which R_PASS pays for.
        """
        rel = pos - self.hoop_centre
        along = rel @ self.hoop_normal
        h = rel @ self._axis_h
        if along < 0.0:
            v = rel @ self._axis_v
            safe = np.hypot(h / self._safe_h, v / self._safe_v)   # 1 on the safe edge
            if safe <= 1.0:
                return float(np.linalg.norm(rel))
            detour = float(np.linalg.norm(pos - self._approach)) + CLEARANCE
            hole = np.hypot(h / self._hole_h, v / self._hole_v)   # 1 on the hole's edge
            if hole >= 1.0:
                return detour
            # How far out from the safe edge to the hole's edge along this direction, 0 to 1.
            w = (1.0 - 1.0 / safe) / (1.0 / hole - 1.0 / safe)
            return float((1.0 - w) * np.linalg.norm(rel) + w * detour)
        best = np.inf
        for gate in self._gates:
            rest = float(np.linalg.norm(gate - self._approach)) + CLEARANCE
            if h * ((gate - self.hoop_centre) @ self._axis_h) >= self._beside ** 2:
                to_gate = float(np.linalg.norm(pos - gate))
            else:
                back_off = gate + CLEARANCE * self.hoop_normal
                to_gate = float(np.linalg.norm(pos - back_off)) + CLEARANCE
            best = min(best, to_gate + rest)
        return best

    def _crossing(self, prev):
        """None, or (direction, inside, rel) if this step crossed the hoop's plane.

        direction is +1 front to back, -1 back to front; inside means within the
        noodle's centreline where the path met the plane, and rel is that point
        relative to the hoop centre.
        """
        n, c = self.hoop_normal, self.hoop_centre
        along = (self.pos - c) @ n
        before = (prev - c) @ n
        if before < 0.0 <= along:
            direction = 1
        elif along < 0.0 <= before:
            direction = -1
        else:
            return None
        rel = prev + (self.pos - prev) * (-before / (along - before)) - c
        inside = ((rel @ self._axis_h) / self._line_h) ** 2 + ((rel @ self._axis_v) / self._line_v) ** 2 < 1.0
        return direction, inside, rel

    def _hit(self) -> bool:
        if np.linalg.norm(self.pos - self.hoop_centre) > self._hit_reach:
            return False   # too far away to touch: skip the 256-point check
        return bool((self._box_distance(self._centreline) < self.hoop.thickness / 2).any())

    def _hits_hanger(self) -> bool:
        """Drone box against the hanger strip. Both are upright boxes turned only
        in yaw, so a height check plus a 2D separating-axis test is exact."""
        if not self.hanger or self.pos[2] + self.drone.height / 2 < self._hanger_bottom:
            return False
        c, s = np.cos(self.heading), np.sin(self.heading)
        drone_axes = [np.array([c, s]), np.array([-s, c])]
        drone_half = self._half_box[:2]
        strip_axes = [self.hoop_normal[:2], self._axis_h[:2]]
        strip_half = [self.hoop.thickness / 2, self.hanger_half_width]
        gap = self.pos[:2] - self.hoop_centre[:2]
        for axis in drone_axes + strip_axes:
            reach = (sum(h * abs(a @ axis) for h, a in zip(drone_half, drone_axes))
                     + sum(h * abs(a @ axis) for h, a in zip(strip_half, strip_axes)))
            if abs(gap @ axis) > reach:
                return False   # a separating axis: no contact
        return True

    def _touch_walls(self) -> tuple[bool, str]:
        """Stop the drone's box at the net or the floor. Say whether it touched
        either, and which net face it touched, which ends the episode: "roof" for
        the ceiling, "side" for the four walls, "" for neither. The floor is
        min_altitude, with nothing there to catch on.
        """
        c, s = abs(np.cos(self.heading)), abs(np.sin(self.heading))
        forward, left, up = self._half_box
        reach = np.array([c * forward + s * left, s * forward + c * left])
        low = np.array([*reach, self.cage.min_altitude])
        high = np.array([self.cage.length - reach[0], self.cage.width - reach[1],
                         self.cage.height - up])
        inside = np.clip(self.pos, low, high)
        if np.array_equal(inside, self.pos):
            return False, ""
        # How far through each face it went. Down is the floor, which is no crash,
        # so only an overshoot above counts in z. One step can breach two faces at
        # once, and then the face it went furthest through is the one it met.
        over = np.abs(self.pos - inside)
        if self.pos[2] < low[2]:
            over[2] = 0.0
        face = ""
        if over.any():
            face = "roof" if over[2] > max(over[0], over[1]) else "side"
        self.vel = np.where(inside == self.pos, self.vel, 0.0)
        self.pos = inside
        return True, face

    def _outcome(self, prev) -> str:
        """Front to back through the ring scores; back through the ring is wrong_way.
        Crossing the plane outside the ring is how the drone gets round, so it is neutral.
        The hanger counts as part of the hoop."""
        self.route = self.centred = None
        if self._hit() or self._hits_hanger():
            return "hit"
        crossing = self._crossing(prev)
        if crossing is not None and crossing[1]:
            if crossing[0] == 1:
                self.centred = self._centred(crossing[2])
                return "passed"
            return "wrong_way"
        if crossing is not None and crossing[0] == -1:
            self.route = self._route(crossing[2])
        return "running"

    def _route(self, rel) -> str:
        """Which way round, as seen from the front: left, right, over or under."""
        h = (rel @ self._axis_h) / self._line_h   # axis_h points left, seen from the front
        v = (rel @ self._axis_v) / self._line_v
        if abs(v) > abs(h):
            return "over" if v > 0 else "under"
        return "left" if h > 0 else "right"

    def _box_distance(self, pts, pos=None, heading=None):
        """Distance from each world point to the drone's collision box (0 inside it), the
        box where the drone is, or at pos turned to heading."""
        pos = self.pos if pos is None else pos
        heading = self.heading if heading is None else heading
        c, s = np.cos(heading), np.sin(heading)
        body = (pts - pos) @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return np.linalg.norm(np.maximum(np.abs(body) - self._half_box, 0.0), axis=1)

    def _clearance(self, pos, heading) -> float:
        """Gap in metres from the drone's box at pos, turned to heading, to the noodle's
        surface: 0 or less touches it."""
        return float(self._box_distance(self._centreline, pos, heading).min() - self.hoop.thickness / 2)

    def _centred(self, rel) -> float:
        """How centred a pass was, 0 to 1: the box's clearance from the noodle where it
        crossed the plane (rel from the hoop centre), as it was turned, as a share of the
        most this hoop allows. Off to a side, high or low, or crabbing all cut it."""
        gap = self._clearance(self.hoop_centre + rel, self.heading)
        return float(np.clip(gap / self._best_clearance, 0.0, 1.0))

    # --- observation -----------------------------------------------------

    def _as_detected(self, obs: np.ndarray) -> np.ndarray:
        """What the real detector would report from this view: nothing until the hoop
        has been fitted in confirm_frames frames in a row, and the marker missed some
        of the time it is in view."""
        self._streak = self._streak + 1 if obs[contract.VISIBLE] else 0
        if self._streak < self.detector.confirm_frames:
            return contract.NOT_VISIBLE.copy()
        near = obs[contract.FIELDS.index("major")] >= self.detector.near_major
        miss = self.detector.marker_miss_near if near else self.detector.marker_miss
        if obs[contract.FIELDS.index("marker")] and self.np_random.uniform() < miss:
            obs = obs.copy()
            obs[contract.FIELDS.index("marker")] = 0.0
        return obs

    def observe(self) -> np.ndarray:
        """The eight numbers the camera would show in the current state, before
        _as_detected's confirmation and marker misses."""
        # One projection covers the outer edge and the marker, which is last.
        cam_pos, cam_axes = self._camera_pose()
        xn, yn, in_frame = self._project(self._sight_points, cam_pos, cam_axes)
        edge = in_frame[:-1]
        if edge.mean() < max(contract.MIN_ARC_FRACTION, self.detector.min_arc_fraction):
            return contract.NOT_VISIBLE.copy()
        return contract.observe(np.stack([xn[:-1], yn[:-1]], axis=1)[edge], self.camera.half_extent,
                                marker=self._marker_seen(cam_pos, in_frame[-1]))

    def _project(self, pts, cam_pos, cam_axes):
        """Normalised image coordinates of world points, and which of them are in frame."""
        cam = (pts - cam_pos) @ cam_axes
        z = cam[:, 2]
        in_front = z > 0.05
        z_safe = np.where(in_front, z, 1.0)
        xn, yn = cam[:, 0] / z_safe, cam[:, 1] / z_safe
        k = self.camera
        px, py = k.fx * xn + k.cx, k.fy * yn + k.cy
        return xn, yn, in_front & (px >= 0) & (px < k.width) & (py >= 0) & (py < k.height)

    def _marker_seen(self, cam_pos, in_frame) -> bool:
        """In frame, and viewed from within marker_max_angle_deg of the hoop's front."""
        to_camera = cam_pos - self._marker
        facing = (to_camera @ -self.hoop_normal) / np.linalg.norm(to_camera)
        return bool(facing >= self._marker_cos and in_frame)

    def _camera_pose(self):
        """The camera's position, and a matrix taking world offsets to the camera
        frame (x right, y down, z forward): its columns are those axes in world coordinates."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        cp, sp = self._cos_pitch, self._sin_pitch
        pos = self.pos + self.camera.offset_forward * np.array([c, s, 0.0])
        # right, down (-sp*forward - cp*up), optical (cp*forward - sp*up)
        axes = np.array([[s, -sp * c, cp * c],
                         [-c, -sp * s, cp * s],
                         [0.0, -cp, -sp]])
        return pos, axes

    def _height_reading(self) -> float:
        """The height the policy gets: the drone's, off by the episode's offset, with
        noise, in steps of the episode's resolution."""
        a = self.altimeter
        raw = self.pos[2] + a.offset + self.np_random.normal(0.0, max(a.noise, 0.0))
        return float(np.round(raw / self.height_step) * self.height_step)

    def _motion_reading(self) -> np.ndarray:
        """The motion the policy gets (contract.motion_reading), as the Tello's state packet
        gives it: speed from two axes each in whole steps with noise (vgx, vgy, which
        labcheck.motion_from_state combines), heading in whole degrees with drift, both
        telemetry_lag steps late, and now and then no new packet at all."""
        t = self.telemetry
        turned = _wrap(self.heading - self._start_heading + np.deg2rad(self.yaw_drift * self._clock))
        c, s = np.cos(self.heading), np.sin(self.heading)
        body = (float(self.vel[0] * c + self.vel[1] * s), float(self.vel[0] * s - self.vel[1] * c))
        self._motion_true.append((body, turned))
        if self.np_random.uniform() < t.stale:
            return self._last_motion.copy()
        body, turned = self._motion_true[0]
        noisy = np.asarray(body) + self.np_random.normal(0.0, t.speed_noise, 2)
        steps = np.trunc(noisy / t.speed_step) * t.speed_step   # whole steps, towards 0
        speed = float(np.hypot(*steps))
        step = np.deg2rad(t.yaw_step_deg)
        self._last_motion = contract.motion_reading(speed, np.round(turned / step) * step)
        return self._last_motion.copy()

    def _policy_input(self):
        # The deques hold every step; contract.policy_input groups them into slots.
        return contract.policy_input(self._frames, self._heights, self._motion, self._actions, self._unseen)

    def privileged(self) -> np.ndarray:
        """The true state, for src.ppo's critic in training only: never the policy's
        input, and nothing on the drone has it. The policy sees blank frames for
        whole stretches of the lap, and a critic on the same input values every blind
        state alike. With this it knows where the drone is round the lap, how fast it
        flies, which commands are still in flight and the episode's draw of the setup.
        PRIVILEGED_DIM long, the same in every layout."""
        cage = np.array([self.cage.length, self.cage.width, self.cage.height])
        pending = np.zeros((PENDING_MAX, contract.ACTION_DIM))
        queued = np.array(list(self._pending), float).reshape(-1, contract.ACTION_DIM)[:PENDING_MAX]
        pending[:len(queued)] = queued   # none queued at delay_steps 0
        d = self.dyn
        dynamics = {"delay_steps": d.delay_steps, "tau": d.tau, "max_speed": d.max_speed,
                    "max_up_speed": d.max_up_speed, "up_exponent": d.up_exponent,
                    "max_yaw_rate_deg": d.max_yaw_rate_deg, "yaw_deadband": d.yaw_deadband,
                    "telemetry_lag": self.telemetry_lag, "yaw_drift": self.yaw_drift,
                    "camera_pitch_deg": self.camera.pitch_deg, "altimeter_offset": self.altimeter.offset,
                    "height_step": self.height_step}
        relative = self._relative()
        return np.concatenate([
            relative[:9],
            self.pos / cage, cage,
            relative[9:], [self._to_go(self.pos), float(self._unseen), float(self._found)],
            [dynamics[name] for name in DYNAMICS_FIELDS],
            pending.ravel(),
        ]).astype(np.float32)

    def _relative(self) -> np.ndarray:
        """privileged()'s GATE_HELD numbers, in order: the drone's position and velocity in
        the hoop's frame, its facing and yaw rate, the hoop's height and size."""
        n = self.hoop_normal
        to_hoop = np.array([n, self._axis_h, self._axis_v])   # world offsets to the hoop's frame
        facing = _wrap(self.heading - np.arctan2(n[1], n[0]))
        return np.concatenate([
            to_hoop @ (self.pos - self.hoop_centre), to_hoop @ self.vel,
            [np.sin(facing), np.cos(facing), self.yaw_rate],
            [self.hoop_centre[2], self._line_h, self._line_v]])

    def _update_gate(self, frame: np.ndarray):
        """After each step: while the camera reports the hoop (frame, the newest in the
        policy input), the gate's held numbers follow the true ones, blurred by GATE_BLUR."""
        if not frame[contract.VISIBLE]:
            self._since = min(self._since + 1, GATE_SINCE_MAX)
            return
        true = self._relative()
        if self._since >= GATE_REACQUIRE:   # a new sighting: a new error, kept while it lasts
            self._blur = self._gate_rng.standard_normal(len(GATE_HELD))
        distance = float(np.linalg.norm(true[:3]))
        sd = np.array([np.interp(distance, GATE_BLUR_DISTANCE, row) for row in GATE_BLUR])
        held = true + self._blur * sd
        facing = np.arctan2(true[GATE_FACING[0]], true[GATE_FACING[1]])
        facing += np.deg2rad(self._blur[GATE_FACING[0]] * np.interp(distance, GATE_BLUR_DISTANCE, GATE_BLUR_FACING))
        held[list(GATE_FACING)] = np.sin(facing), np.cos(facing)
        self._held = held.astype(np.float32)
        self._reported, self._since = True, 0

    def gated(self, privileged: np.ndarray | None = None) -> np.ndarray:
        """What a gated teacher reads after the policy input: privileged() (pass it in if
        already computed) with the hoop-relative numbers held at the last report and the
        cage hidden (GATE_HELD, GATE_NEVER), then whether the hoop has been reported yet
        and the time since, 0 to 1 over GATE_SINCE_MAX steps. GATED_DIM long."""
        p = (self.privileged() if privileged is None else privileged).copy()
        p[GATE_HELD] = self._held
        p[GATE_NEVER] = 0.0
        return np.concatenate([p, [float(self._reported), self._since / GATE_SINCE_MAX]]).astype(np.float32)

    def _info(self, outcome):
        return {"outcome": outcome, "passes": self.passes, "steps": self.steps,
                "wall_steps": self.wall_steps, "route": self.route, "centred": self.centred,
                "level": self.level, "run_in": self.run_in, "drone_pos": self.pos.copy(),
                "heading": self.heading, "hoop_centre": self.hoop_centre.copy(),
                "hoop_normal": self.hoop_normal.copy()}
