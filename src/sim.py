"""Gymnasium environment: the drone flying at a hoop, rendering nothing.

Two tasks:
    hover   the drone starts hovering near the hoop, facing it; the episode ends
            at the first pass. The original task, kept so its results reproduce.
    laps    the mission. The drone starts above the take-off pad in a corner and
            must find the hoop, pass through it front to back, fly round the
            outside back to the front, and pass again, for as long as it lasts.

The policy sees only the eight numbers from contract.py (stacked with its own
recent actions). The true hoop position is used for reward and termination only.
"""
from __future__ import annotations

import dataclasses
from collections import deque

import gymnasium as gym
import numpy as np

from src import contract
from src.params import Cage, Camera, Drone, Hoop, Pad, Randomise, load_camera

TASKS = ("hover", "laps")

# Hover layout: every reset draws a fresh hoop position and drone start from
# these ranges, so no two episodes look the same.
HOOP_DISTANCE = (0.6, 1.2)     # drone start to hoop centre, metres
HOOP_HEIGHT = (0.6, 1.4)       # hoop centre above the floor
HOOP_YAW_DEG = 30.0            # max hoop angle relative to the drone's line of sight
START_HEIGHT_OFFSET = 0.2      # drone start height relative to the hoop centre
CLEARANCE_BEHIND = 0.4         # free space past the hoop for the drone to stop
MAX_STEPS = 150                # 15 s at 10 Hz

# Laps layout. The hoop goes anywhere with room to fly round it; its front faces
# the pad. The curriculum level blends each start from easy (0) to real (1).
PAD_MIN_DISTANCE = 0.6         # pad to hoop centre
LAPS_HOOP_ANGLE_DEG = (30.0, 75.0)  # max hoop angle to the pad's line of sight, level 0 and 1.
                               # Past ~78 degrees the detector cannot see the hoop at all.
CLEARANCE = 0.4                # free space in front of and behind the hoop
SIDE_GAP = 0.35                # the side waypoint's distance beyond the rim
LAPS_MAX_STEPS = 600           # 60 s; a cut-off, not a failure: the real run goes on

# Collision checks use this many points along the noodle's centreline. At ~5 mm
# spacing, a contact can be missed by at most ~0.1 mm.
CENTRELINE_POINTS = 256

MIN_ARC_FRACTION = 0.25        # UNMEASURED: share of the hoop's edge the detector needs in frame, B3

# Reward weights. MISSED must outweigh the progress a drone can bank by flying
# past the hoop instead of through it (up to 12 over a 1.2 m approach).
# R_PROGRESS pays for closing the straight-line distance to the hoop centre, not
# for closing the axial gap. Axial progress rated the rim and the centre alike,
# so SAC charged the plane and hit. Distance to the centre falls fastest on the
# line that passes through, which is the behaviour we want.
# In laps, behind the hoop the target is a side waypoint instead: distance to the
# centre from behind would pay for flying back through the hole.
PROGRESS_UNIT = 0.10           # metres
R_PROGRESS = 1.0               # per PROGRESS_UNIT closer to the target
R_HIDDEN = 0.1                 # per step the hoop is not visible
# Per step, so hovering until the timeout costs 7.5. Without it SAC learned to
# stop short of the hoop and hover, which scored better than risking the pass.
R_TIME = 0.05
# left_cage is crossing the geofence, Cage.margin inside the net.
# PASSED is well above HIT so that an attempt with even odds is worth making.
# In laps a pass pays and the episode goes on; wrong_way is back through the hole.
R_TERMINAL = {"passed": 25.0, "hit": -10.0, "missed": -15.0, "left_cage": -10.0,
              "wrong_way": -10.0}
LAPS_TERMINAL = ("hit", "wrong_way", "left_cage")


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class HoopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task: str = "hover", level: float = 1.0, randomise: bool = False,
                 yaw_enabled: bool = False, start_angle_deg: float = 10.0,
                 cage: Cage | None = None, hoop: Hoop | None = None,
                 camera: Camera | None = None, drone: Drone | None = None,
                 pad: Pad | None = None, ranges: Randomise | None = None):
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, not {task!r}")
        self.task = task
        self.cage = cage or Cage()
        self.hoop = hoop or Hoop()
        self.camera = camera or load_camera()
        self.drone = drone or Drone()
        self.pad = pad or Pad()
        self.ranges = ranges or Randomise()
        self.randomise = randomise
        # Laps needs yaw: the drone turns to find the hoop and to fly round it.
        self.yaw_enabled = yaw_enabled or task == "laps"
        self.max_steps = LAPS_MAX_STEPS if task == "laps" else MAX_STEPS
        # Hover only: max heading offset from the hoop at the start.
        self.start_angle_deg = start_angle_deg
        self.set_level(level)

        self.action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (contract.POLICY_INPUT_DIM,), np.float32)

        phi = np.linspace(0.0, 2 * np.pi, self.hoop.n_points, endpoint=False)
        self._cos_phi, self._sin_phi = np.cos(phi), np.sin(phi)
        phi = np.linspace(0.0, 2 * np.pi, CENTRELINE_POINTS, endpoint=False)
        self._cos_line, self._sin_line = np.cos(phi), np.sin(phi)
        self._marker_cos = np.cos(np.deg2rad(self.hoop.marker_max_angle_deg))

        # Half-sizes of the drone's collision box in the body frame (forward, left, up).
        d = self.drone
        self._half_box = np.array([d.length, d.width, d.height]) / 2

    def set_level(self, level: float):
        """Curriculum level for laps starts, 0 (easy) to 1 (the real layout)."""
        self.level = float(np.clip(level, 0.0, 1.0))

    # --- episode setup ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.task == "laps":
            self._sample_pad_layout()
        else:
            self._sample_layout()
        return self._start_episode(), self._info("running")

    def place(self, drone_pos, heading, hoop_centre, hoop_normal, hoop_diameter=None):
        """Set an exact layout instead of a random one. For tests and replaying real flights.

        hoop_normal points out of the hoop's back: the front, with the marker,
        faces the other way. hoop_diameter is one outer diameter, or
        (horizontal, vertical) for a squashed hoop.
        """
        diameter = self.hoop.outer_diameter if hoop_diameter is None else hoop_diameter
        radius_h, radius_v = np.broadcast_to(np.asarray(diameter, float), 2) / 2
        self._set_layout(np.asarray(drone_pos, float), float(heading),
                         np.asarray(hoop_centre, float), np.asarray(hoop_normal, float),
                         radius_h, radius_v)
        return self._start_episode()

    def _in_fence(self, xy) -> bool:
        m = self.cage.margin
        return bool(m <= xy[0] <= self.cage.length - m and m <= xy[1] <= self.cage.width - m)

    def _sample_layout(self):
        rng, cage, m = self.np_random, self.cage, self.cage.margin
        lo = np.array([m, m])
        hi = np.array([cage.length - m, cage.width - m])
        for _ in range(1000):
            start = rng.uniform(lo, hi)
            bearing = rng.uniform(-np.pi, np.pi)
            centre = start + rng.uniform(*HOOP_DISTANCE) * np.array([np.cos(bearing), np.sin(bearing)])
            angle = bearing + np.deg2rad(rng.uniform(-HOOP_YAW_DEG, HOOP_YAW_DEG))
            normal = np.array([np.cos(angle), np.sin(angle)])
            side = np.array([-normal[1], normal[0]])
            radii = (self.hoop.outer_diameter + rng.uniform(-1, 1, 2) * self.hoop.diameter_tolerance) / 2
            corners = [centre + radii[0] * side, centre - radii[0] * side,
                       centre + CLEARANCE_BEHIND * normal]
            if all(((lo <= c) & (c <= hi)).all() for c in corners):
                break
        else:
            raise RuntimeError("cage too small for the episode layout ranges in sim.py")

        hoop_z = rng.uniform(*HOOP_HEIGHT)
        drone_z = np.clip(hoop_z + rng.uniform(-START_HEIGHT_OFFSET, START_HEIGHT_OFFSET),
                          cage.min_altitude, cage.height - m)
        heading = bearing + np.deg2rad(rng.uniform(-self.start_angle_deg, self.start_angle_deg))
        self._set_layout(np.array([*start, drone_z]), heading,
                         np.array([*centre, hoop_z]), np.array([*normal, 0.0]), *radii)

    def _sample_pad_layout(self):
        """The drone above the pad in a random corner, the hoop anywhere it can be lapped."""
        rng, cage, pad, m = self.np_random, self.cage, self.pad, self.cage.margin
        max_angle = np.interp(self.level, [0.0, 1.0], LAPS_HOOP_ANGLE_DEG)
        for _ in range(1000):
            start = np.array([rng.choice([pad.inset, cage.length - pad.inset]),
                              rng.choice([pad.inset, cage.width - pad.inset])])
            centre = rng.uniform([m, m], [cage.length - m, cage.width - m])
            offset = centre - start
            if np.linalg.norm(offset) < PAD_MIN_DISTANCE:
                continue
            bearing = np.arctan2(offset[1], offset[0])
            # The normal points away from the pad, so the front faces it.
            angle = bearing + np.deg2rad(rng.uniform(-max_angle, max_angle))
            normal = np.array([np.cos(angle), np.sin(angle)])
            side = np.array([-normal[1], normal[0]])
            radii = (self.hoop.outer_diameter + rng.uniform(-1, 1, 2) * self.hoop.diameter_tolerance) / 2
            must_fit = [centre + radii[0] * side, centre - radii[0] * side,
                        centre + CLEARANCE * normal, centre - CLEARANCE * normal]
            gates = [centre + (radii[0] + SIDE_GAP) * side, centre - (radii[0] + SIDE_GAP) * side]
            if all(self._in_fence(p) for p in must_fit) and any(self._in_fence(g) for g in gates):
                break
        else:
            raise RuntimeError("cage too small for the laps layout ranges in sim.py")

        hoop_z = rng.uniform(*HOOP_HEIGHT)
        drone_z = np.clip(np.interp(self.level, [0.0, 1.0], [hoop_z, self.drone.takeoff_height]),
                          cage.min_altitude, cage.height - m)
        # Level 0 faces the hoop; level 1 faces the cage centre, off by the pad's placement.
        to_centre = np.array([cage.length, cage.width]) / 2 - start
        facing = (np.arctan2(to_centre[1], to_centre[0])
                  + np.deg2rad(rng.uniform(-pad.heading_error_deg, pad.heading_error_deg)))
        heading = _wrap(bearing + self.level * _wrap(facing - bearing))
        self._set_layout(np.array([*start, drone_z]), heading,
                         np.array([*centre, hoop_z]), np.array([*normal, 0.0]), *radii)

    def _set_layout(self, drone_pos, heading, hoop_centre, hoop_normal, radius_h, radius_v):
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
        # The marker sits on the noodle's front face at 6 o'clock.
        self._marker = hoop_centre - self._line_v * self._axis_v - half_t * n
        # Side waypoints for the way round, at the hoop's height. Only those inside
        # the geofence count; a layout from place() may have none, so keep both then.
        gates = [hoop_centre + s * (radius_h + SIDE_GAP) * self._axis_h for s in (1, -1)]
        self._gates = [g for g in gates if self._in_fence(g)] or gates

    def _start_episode(self):
        # Randomised dynamics are drawn once per episode into a copy of the drone.
        self.dyn = self.drone
        if self.randomise:
            r, rng = self.ranges, self.np_random
            self.dyn = dataclasses.replace(
                self.drone,
                delay_steps=int(rng.integers(r.delay_steps[0], r.delay_steps[1] + 1)),
                tau=float(rng.uniform(*r.tau)))
        self.vel = np.zeros(3)
        self.yaw_rate = 0.0
        self.steps = 0
        self.passes = 0
        self._pending = deque([np.zeros(contract.ACTION_DIM, np.float32)] * self.dyn.delay_steps)
        self._frames = deque([self.observe()] * contract.FRAME_HISTORY,
                             maxlen=contract.FRAME_HISTORY)
        self._actions = deque([np.zeros(contract.ACTION_DIM, np.float32)] * contract.ACTION_HISTORY,
                              maxlen=contract.ACTION_HISTORY)
        return self._policy_input()

    # --- stepping --------------------------------------------------------

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0).copy()
        if not self.yaw_enabled:
            action[3] = 0.0

        # Command delay: the drone acts on the command from delay_steps ago.
        self._pending.append(action)
        right, forward, up, yaw_cw = self._pending.popleft()

        # Commands are body-frame: rotate by heading before integrating.
        d = self.dyn
        dt = 1.0 / self.np_random.uniform(*self.ranges.rate_hz) if self.randomise else d.dt
        c, s = np.cos(self.heading), np.sin(self.heading)
        v_cmd = d.max_speed * np.array([forward * c + right * s, forward * s - right * c, up])
        yaw_cmd = -yaw_cw * np.deg2rad(d.max_yaw_rate_deg)  # heading is counter-clockwise
        alpha = 1.0 - np.exp(-dt / d.tau)
        self.vel += (v_cmd - self.vel) * alpha
        self.yaw_rate += (yaw_cmd - self.yaw_rate) * alpha

        prev = self.pos.copy()
        target = self._progress_target(prev)
        self.pos = self.pos + self.vel * dt
        self.heading = (self.heading + self.yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1

        outcome = self._lap_outcome(prev) if self.task == "laps" else self._outcome(prev)
        if outcome == "passed":
            self.passes += 1
        obs = self.observe()
        self._frames.append(obs)
        self._actions.append(action)

        progress = float(np.linalg.norm(prev - target) - np.linalg.norm(self.pos - target))
        reward = (R_PROGRESS * progress / PROGRESS_UNIT - R_HIDDEN * (1.0 - obs[6])
                  - R_TIME + R_TERMINAL.get(outcome, 0.0))
        if self.task == "laps":
            terminated = outcome in LAPS_TERMINAL
        else:
            terminated = outcome != "running"
        truncated = not terminated and self.steps >= self.max_steps
        if truncated:
            outcome = "timeout"
        return self._policy_input(), float(reward), terminated, truncated, self._info(outcome)

    def _progress_target(self, pos):
        """The hoop centre, or in laps from behind the hoop, the nearest side waypoint."""
        if self.task == "laps" and (pos - self.hoop_centre) @ self.hoop_normal >= 0.0:
            return min(self._gates, key=lambda g: np.linalg.norm(pos - g))
        return self.hoop_centre

    def _crossing(self, prev):
        """None, or (direction, inside) if this step crossed the hoop's plane.

        direction is +1 front to back, -1 back to front; inside means within the
        noodle's centreline where the path met the plane.
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
        return direction, inside

    def _hit(self) -> bool:
        return bool((self._box_distance(self._centreline) < self.hoop.thickness / 2).any())

    def _left_cage(self) -> bool:
        z = self.pos[2]
        return not (self._in_fence(self.pos) and self.cage.min_altitude <= z <= self.cage.height - self.cage.margin)

    def _outcome(self, prev) -> str:
        if self._hit():
            return "hit"
        crossing = self._crossing(prev)
        if crossing is not None and crossing[0] == 1:
            return "passed" if crossing[1] else "missed"
        if self._left_cage():
            return "left_cage"
        return "running"

    def _lap_outcome(self, prev) -> str:
        """Front to back through the ring scores; back through the ring is wrong_way.
        Crossing the plane outside the ring is how the drone gets round, so it is neutral."""
        if self._hit():
            return "hit"
        crossing = self._crossing(prev)
        if crossing is not None and crossing[1]:
            return "passed" if crossing[0] == 1 else "wrong_way"
        if self._left_cage():
            return "left_cage"
        return "running"

    def _box_distance(self, pts):
        """Distance from each world point to the drone's collision box (0 inside it)."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        rel = pts - self.pos
        body = np.stack([rel @ [c, s, 0.0], rel @ [-s, c, 0.0], rel[:, 2]], axis=1)
        return np.linalg.norm(np.maximum(np.abs(body) - self._half_box, 0.0), axis=1)

    # --- observation -----------------------------------------------------

    def observe(self) -> np.ndarray:
        """The eight numbers the camera would report in the current state."""
        pts = (self.hoop_centre + np.outer(self._cos_phi, self._hoop_h)
               + np.outer(self._sin_phi, self._hoop_v))
        xn, yn, in_frame = self._project(pts)
        if in_frame.mean() < MIN_ARC_FRACTION:
            return contract.NOT_VISIBLE.copy()
        return contract.observe(np.stack([xn, yn], axis=1)[in_frame], self.camera.half_extent,
                                marker=self._marker_seen())

    def _project(self, pts):
        """Normalised image coordinates of world points, and which of them are in frame."""
        cam = self._to_camera(pts)
        z = cam[:, 2]
        in_front = z > 0.05
        z_safe = np.where(in_front, z, 1.0)
        xn, yn = cam[:, 0] / z_safe, cam[:, 1] / z_safe
        k = self.camera
        px, py = k.fx * xn + k.cx, k.fy * yn + k.cy
        return xn, yn, in_front & (px >= 0) & (px < k.width) & (py >= 0) & (py < k.height)

    def _marker_seen(self) -> bool:
        """In frame, and viewed from within marker_max_angle_deg of the hoop's front."""
        to_camera = self._camera_pos() - self._marker
        facing = (to_camera @ -self.hoop_normal) / np.linalg.norm(to_camera)
        return bool(facing >= self._marker_cos and self._project(self._marker[None])[2][0])

    def _camera_pos(self):
        fwd = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        return self.pos + self.camera.offset_forward * fwd

    def _to_camera(self, pts):
        """World points to the camera frame: x right, y down, z forward."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        fwd = np.array([c, s, 0.0])
        right = np.array([s, -c, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        p = np.deg2rad(self.camera.pitch_deg)
        optical = np.cos(p) * fwd - np.sin(p) * up
        down = -np.sin(p) * fwd - np.cos(p) * up
        rel = pts - self._camera_pos()
        return np.stack([rel @ right, rel @ down, rel @ optical], axis=1)

    def _policy_input(self):
        # _frames holds every step. OBS_LAGS picks the strided subset, oldest
        # first, so the newest frame lands last.
        frames = [self._frames[-1 - lag] for lag in contract.OBS_LAGS]
        return np.concatenate([*frames, *self._actions]).astype(np.float32)

    def _info(self, outcome):
        return {"outcome": outcome, "passes": self.passes, "steps": self.steps,
                "level": self.level, "drone_pos": self.pos.copy(),
                "heading": self.heading, "hoop_centre": self.hoop_centre.copy(),
                "hoop_normal": self.hoop_normal.copy()}
