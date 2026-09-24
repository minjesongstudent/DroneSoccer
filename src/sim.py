"""Gymnasium environment: the drone flying at a hoop, rendering nothing.

The policy sees only the seven numbers from contract.py (stacked with its own
recent actions). The true hoop position is used for reward and termination only.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np

from src import contract
from src.params import Cage, Camera, Drone, Hoop, load_camera

# Episode layout: every reset draws a fresh hoop position and drone start
# from these ranges, so no two episodes look the same.
HOOP_DISTANCE = (0.6, 1.2)     # drone start to hoop centre, metres
HOOP_HEIGHT = (0.6, 1.4)       # hoop centre above the floor
HOOP_YAW_DEG = 30.0            # max hoop angle relative to the drone's line of sight
START_HEIGHT_OFFSET = 0.2      # drone start height relative to the hoop centre
CLEARANCE_BEHIND = 0.4         # free space past the hoop for the drone to stop
MAX_STEPS = 150                # 15 s at 10 Hz

# Collision checks use this many points along the noodle's centreline. At ~5 mm
# spacing, a contact can be missed by at most ~0.1 mm.
CENTRELINE_POINTS = 256

# The real detector needs a decent arc in frame. Stand-in until B3 logs dropout.
MIN_ARC_FRACTION = 0.25

# Reward weights. MISSED must outweigh the progress a drone can bank by flying
# past the hoop instead of through it (up to 12 over a 1.2 m approach).
PROGRESS_UNIT = 0.10           # metres
R_PROGRESS = 1.0               # per PROGRESS_UNIT along the hoop's axis
R_JERK = 0.05                  # times squared change in action
R_HIDDEN = 0.1                 # per step the hoop is not visible
# left_cage is crossing the geofence, Cage.margin inside the net.
R_TERMINAL = {"passed": 10.0, "hit": -10.0, "missed": -15.0, "left_cage": -10.0}

_ZERO_ACTION = np.zeros(contract.ACTION_DIM, dtype=np.float32)


class HoopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, yaw_enabled: bool = False, start_angle_deg: float = 10.0,
                 cage: Cage | None = None, hoop: Hoop | None = None,
                 camera: Camera | None = None, drone: Drone | None = None):
        self.cage = cage or Cage()
        self.hoop = hoop or Hoop()
        self.camera = camera or load_camera()
        self.drone = drone or Drone()
        self.yaw_enabled = yaw_enabled
        # Max heading offset from the hoop at the start. The A4 curriculum widens it.
        self.start_angle_deg = start_angle_deg

        self.action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (contract.POLICY_INPUT_DIM,), np.float32)

        phi = np.linspace(0.0, 2 * np.pi, self.hoop.n_points, endpoint=False)
        self._cos_phi, self._sin_phi = np.cos(phi), np.sin(phi)
        phi = np.linspace(0.0, 2 * np.pi, CENTRELINE_POINTS, endpoint=False)
        self._cos_line, self._sin_line = np.cos(phi), np.sin(phi)

        # Half-sizes of the drone's collision box in the body frame (forward, left, up).
        d = self.drone
        self._half_box = np.array([d.length, d.width, d.height]) / 2

    # --- episode setup ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_layout()
        return self._start_episode(), self._info("running")

    def place(self, drone_pos, heading, hoop_centre, hoop_normal, hoop_diameter=None):
        """Set an exact layout instead of a random one. For tests and replaying real flights.

        hoop_diameter is one outer diameter, or (horizontal, vertical) for a squashed hoop.
        """
        diameter = self.hoop.outer_diameter if hoop_diameter is None else hoop_diameter
        radius_h, radius_v = np.broadcast_to(np.asarray(diameter, float), 2) / 2
        self._set_layout(np.asarray(drone_pos, float), float(heading),
                         np.asarray(hoop_centre, float), np.asarray(hoop_normal, float),
                         radius_h, radius_v)
        return self._start_episode()

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

    def _start_episode(self):
        self.vel = np.zeros(3)
        self.yaw_rate = 0.0
        self.steps = 0
        self._pending = deque([_ZERO_ACTION] * self.drone.delay_steps)
        self._frames = deque([self.observe()] * contract.OBS_STACK, maxlen=contract.OBS_STACK)
        self._actions = deque([_ZERO_ACTION] * contract.ACTION_HISTORY, maxlen=contract.ACTION_HISTORY)
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
        d = self.drone
        c, s = np.cos(self.heading), np.sin(self.heading)
        v_cmd = d.max_speed * np.array([forward * c + right * s, forward * s - right * c, up])
        yaw_cmd = -yaw_cw * np.deg2rad(d.max_yaw_rate_deg)  # heading is counter-clockwise
        alpha = 1.0 - np.exp(-d.dt / d.tau)
        self.vel += (v_cmd - self.vel) * alpha
        self.yaw_rate += (yaw_cmd - self.yaw_rate) * alpha

        prev = self.pos.copy()
        self.pos = self.pos + self.vel * d.dt
        self.heading = (self.heading + self.yaw_rate * d.dt + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1

        outcome = self._outcome(prev)
        obs = self.observe()
        jerk = float(np.sum((action - self._actions[-1]) ** 2))
        self._frames.append(obs)
        self._actions.append(action)

        progress = float((self.pos - prev) @ self.hoop_normal)
        reward = (R_PROGRESS * progress / PROGRESS_UNIT - R_JERK * jerk - R_HIDDEN * (1.0 - obs[6])
                  + R_TERMINAL.get(outcome, 0.0))
        terminated = outcome != "running"
        truncated = not terminated and self.steps >= MAX_STEPS
        if truncated:
            outcome = "timeout"
        return self._policy_input(), float(reward), terminated, truncated, self._info(outcome)

    def _outcome(self, prev) -> str:
        if (self._box_distance(self._centreline) < self.hoop.thickness / 2).any():
            return "hit"

        n, c = self.hoop_normal, self.hoop_centre
        along = (self.pos - c) @ n
        before = (prev - c) @ n
        if before < 0.0 <= along:
            rel = prev + (self.pos - prev) * (-before / (along - before)) - c
            inside = ((rel @ self._axis_h) / self._line_h) ** 2 + ((rel @ self._axis_v) / self._line_v) ** 2 < 1.0
            return "passed" if inside else "missed"

        cage, m = self.cage, self.cage.margin
        x, y, z = self.pos
        if not (m <= x <= cage.length - m and m <= y <= cage.width - m
                and cage.min_altitude <= z <= cage.height - m):
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
        """The seven numbers the camera would report in the current state."""
        pts = (self.hoop_centre + np.outer(self._cos_phi, self._hoop_h)
               + np.outer(self._sin_phi, self._hoop_v))
        cam = self._to_camera(pts)
        z = cam[:, 2]
        in_front = z > 0.05
        z_safe = np.where(in_front, z, 1.0)
        xn, yn = cam[:, 0] / z_safe, cam[:, 1] / z_safe

        k = self.camera
        px, py = k.fx * xn + k.cx, k.fy * yn + k.cy
        in_frame = in_front & (px >= 0) & (px < k.width) & (py >= 0) & (py < k.height)
        if in_frame.mean() < MIN_ARC_FRACTION:
            return contract.NOT_VISIBLE.copy()
        return contract.observe(np.stack([xn, yn], axis=1)[in_frame], k.half_extent)

    def _to_camera(self, pts):
        """World points to the camera frame: x right, y down, z forward."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        fwd = np.array([c, s, 0.0])
        right = np.array([s, -c, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        p = np.deg2rad(self.camera.pitch_deg)
        optical = np.cos(p) * fwd - np.sin(p) * up
        down = -np.sin(p) * fwd - np.cos(p) * up
        rel = pts - (self.pos + self.camera.offset_forward * fwd)
        return np.stack([rel @ right, rel @ down, rel @ optical], axis=1)

    def _policy_input(self):
        return np.concatenate([*self._frames, *self._actions]).astype(np.float32)

    def _info(self, outcome):
        return {"outcome": outcome, "drone_pos": self.pos.copy(), "heading": self.heading,
                "hoop_centre": self.hoop_centre.copy(), "hoop_normal": self.hoop_normal.copy()}
