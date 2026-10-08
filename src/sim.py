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
# -----------------------------------------------------------------------------

import dataclasses
from collections import deque

import gymnasium as gym
import numpy as np

from src import contract
from src.params import Cage, Camera, Detector, Drone, Hoop, Pad, Randomise

# Collision checks use this many points along the noodle's centreline. At ~5 mm
# spacing, a contact can be missed by at most ~0.1 mm.
CENTRELINE_POINTS = 256

def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class HoopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: float = 1.0, randomise: bool = False, hanger: bool = True,
                 cage: Cage | None = None, hoop: Hoop | None = None,
                 camera: Camera | None = None, drone: Drone | None = None,
                 pad: Pad | None = None, ranges: Randomise | None = None,
                 detector: Detector | None = None):
        self.cage = cage or Cage()
        self.hoop = hoop or Hoop()
        self.camera = camera or Camera()
        self.drone = drone or Drone()
        self.pad = pad or Pad()
        self.detector = detector or Detector()
        self.ranges = ranges or Randomise()
        self.randomise = randomise
        self.hanger = hanger   # off replays models trained before it existed
        # The measured setup. With randomise, each episode works on a copy with
        # ranges.error applied; without, on these.
        self._measured = {"cage": self.cage, "hoop": self.hoop, "camera": self.camera, "drone": self.drone,
                          "detector": self.detector}
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
        if self.randomise:
            self._apply_measurement_error()
        self._sample_layout()
        return self._start_episode(), self._info("running")

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
            radii = (np.array([self.hoop.outer_diameter, self.hoop.outer_height])
                     + rng.uniform(-1, 1, 2) * self.hoop.diameter_tolerance) / 2
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
        # Sideways from the centre, past which the drone clears the rim going round.
        self._beside = radius_h + SIDE_GAP / 2

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
        self.wall_steps = 0
        self.route = None   # which way the drone last went round, for replay.py
        self._pending = deque([np.zeros(contract.ACTION_DIM, np.float32)] * self.dyn.delay_steps)
        # As labcheck.fly_policy starts: no frames yet, so every lag is blind, and the
        # detector has to confirm the hoop before it counts. A hoop in view at the
        # start is reported from the second step.
        self._frames = deque([contract.NOT_VISIBLE.copy()] * contract.FRAME_HISTORY,
                             maxlen=contract.FRAME_HISTORY)
        self._streak = 0
        self._unseen = True
        self._actions = deque([np.zeros(contract.ACTION_DIM, np.float32)] * contract.ACTION_HISTORY,
                              maxlen=contract.ACTION_HISTORY)
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
        c, s = np.cos(self.heading), np.sin(self.heading)
        v_cmd = np.array([d.max_speed * (forward * c + right * s), d.max_speed * (forward * s - right * c),
                          d.max_up_speed * up])
        # Small yaw commands barely turn the real drone: scale from the deadband up.
        db = float(np.clip(d.yaw_deadband, 0.0, 0.5))
        yaw_cw = np.sign(yaw_cw) * max(0.0, abs(yaw_cw) - db) / (1.0 - db)
        yaw_cmd = -yaw_cw * np.deg2rad(d.max_yaw_rate_deg)  # heading is counter-clockwise
        alpha = 1.0 - np.exp(-dt / d.tau)
        self.vel += (v_cmd - self.vel) * alpha
        self.yaw_rate += (yaw_cmd - self.yaw_rate) * alpha

        prev = self.pos.copy()
        self.pos = self.pos + self.vel * dt
        self.heading = (self.heading + self.yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1
        on_wall, face = self._touch_walls()
        self.wall_steps += on_wall

        outcome = self._outcome(prev)
        if face and outcome == "running":
            outcome = face
        if outcome == "passed":
            self.passes += 1
        obs = self._as_detected(self.observe())
        self._frames.append(obs)
        self._unseen = contract.not_yet_through(self._unseen, self._frames)
        self._actions.append(action)

        terminated = outcome in contract.CRASHES
        reward = contract.reward(outcome, self._to_go(prev), self._to_go(self.pos),
                                 on_wall, obs.copy(), action.copy())
        truncated = not terminated and self.steps >= contract.LAPS_MAX_STEPS
        if truncated:
            outcome = "timeout"
        return self._policy_input(), float(reward), terminated, truncated, self._info(outcome)

    def _to_go(self, pos) -> float:
        """Metres left round the lap to the next pass, along a route that keeps
        off the noodle. A straight line at the hoop centre from the side, or at
        a side gate from just behind the ring, runs into the rim.

        In front, lined up with the hole: straight to the centre.
        In front, not lined up: to the approach point CLEARANCE out front, then in.
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
            if (h / self._hole_h) ** 2 + (v / self._hole_v) ** 2 < 1.0:
                return float(np.linalg.norm(rel))
            return float(np.linalg.norm(pos - self._approach)) + CLEARANCE
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
        self.route = None
        if self._hit() or self._hits_hanger():
            return "hit"
        crossing = self._crossing(prev)
        if crossing is not None and crossing[1]:
            return "passed" if crossing[0] == 1 else "wrong_way"
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

    def _box_distance(self, pts):
        """Distance from each world point to the drone's collision box (0 inside it)."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        body = (pts - self.pos) @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return np.linalg.norm(np.maximum(np.abs(body) - self._half_box, 0.0), axis=1)

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

    def _policy_input(self):
        # _frames holds every step. OBS_LAGS picks the strided subset, oldest
        # first, so the newest frame lands last.
        return contract.policy_input(self._frames, self._actions, self._unseen)

    def _info(self, outcome):
        return {"outcome": outcome, "passes": self.passes, "steps": self.steps,
                "wall_steps": self.wall_steps, "route": self.route,
                "level": self.level, "drone_pos": self.pos.copy(),
                "heading": self.heading, "hoop_centre": self.hoop_centre.copy(),
                "hoop_normal": self.hoop_normal.copy()}
