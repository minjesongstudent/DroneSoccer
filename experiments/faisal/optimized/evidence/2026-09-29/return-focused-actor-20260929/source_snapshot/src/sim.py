"""Gymnasium environment: the drone flying at a hoop, rendering nothing.

The mission: the drone starts above the take-off pad in a corner and must find
the hoop, pass through it front to back, fly round the outside back to the front,
and pass again, for as long as it lasts.

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

# Layout. The hoop goes anywhere with room to fly round it; its front faces
# the pad. The curriculum level blends each start from easy (0) to real (1).
HOOP_HEIGHT = (0.6, 1.4)       # hoop centre above the floor
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

# Reward weights.
# R_PROGRESS pays for closing the straight-line distance to the hoop centre, not
# for closing the axial gap. Axial progress rated the rim and the centre alike,
# so SAC charged the plane and hit. Distance to the centre falls fastest on the
# line that passes through, which is the behaviour we want.
# Targets advance only after clearance/return milestones. Progress is the
# decrease in the entire remaining route length, including subsequent legs.
# Reaching a waypoint therefore does not suddenly add the next leg as a cost.
# A credited pass resets the route; that reset cost is included in progress.
PROGRESS_UNIT = 0.10           # metres
R_PROGRESS = 1.0               # per PROGRESS_UNIT closer to the target
# Per step the hoop is not visible. Kept small: going round the outside hides the
# hoop, and at 0.1 a minute of lapping cost more than crashing out.
R_HIDDEN = 0.02
# There is no per-step time cost. With one, a 60 s episode cost -30 and a crash
# only -10, so ending the episode early was the cheap way out. Passes (+25) are
# what keep the drone from hovering, and the curriculum needs 2+ per minute.
# A pass pays and the episode goes on. The crash endings (a hit, including the
# hanger; back through the hole; crossing the geofence, Cage.margin inside the net)
# cost more than a pass pays, so trading the rest of the minute for a quick pass
# does not pay.
R_PASS = 25.0
R_CRASH = 40.0
CRASHES = ("hit", "wrong_way", "left_cage")


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class HoopEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, level: float = 1.0, randomise: bool = False, hanger: bool = True,
                 cage: Cage | None = None, hoop: Hoop | None = None,
                 camera: Camera | None = None, drone: Drone | None = None,
                 pad: Pad | None = None, ranges: Randomise | None = None):
        self.cage = cage or Cage()
        self.hoop = hoop or Hoop()
        self.camera = camera or load_camera()
        self.drone = drone or Drone()
        self.pad = pad or Pad()
        self.ranges = ranges or Randomise()
        self.randomise = randomise
        self.hanger = hanger   # off replays models trained before it existed
        self.set_level(level)

        self.action_space = gym.spaces.Box(-1.0, 1.0, (contract.ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (contract.POLICY_INPUT_DIM,), np.float32)

        phi = np.linspace(0.0, 2 * np.pi, self.hoop.n_points, endpoint=False)
        self._cos_phi, self._sin_phi = np.cos(phi), np.sin(phi)
        phi = np.linspace(0.0, 2 * np.pi, CENTRELINE_POINTS, endpoint=False)
        self._cos_line, self._sin_line = np.cos(phi), np.sin(phi)
        self._marker_cos = np.cos(np.deg2rad(self.hoop.marker_max_angle_deg))
        pitch = np.deg2rad(self.camera.pitch_deg)
        self._cos_pitch, self._sin_pitch = np.cos(pitch), np.sin(pitch)

        # Half-sizes of the drone's collision box in the body frame (forward, left, up).
        d = self.drone
        self._half_box = np.array([d.length, d.width, d.height]) / 2

    def set_level(self, level: float):
        """Curriculum level for starts, 0 (easy) to 1 (the real layout)."""
        self.level = float(np.clip(level, 0.0, 1.0))

    # --- episode setup ---------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
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
            offset = self._clearance_target_offset()*normal
            if (all(self._in_fence(p) for p in must_fit)
                    and any(all(self._in_fence(g+s*offset) for s in (-1, 0, 1)) for g in gates)):
                break
        else:
            raise RuntimeError("cage too small for the layout ranges in sim.py")

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
        # The hanger strip runs from the top of the hoop's outer edge to the ceiling.
        self._hanger_bottom = hoop_centre[2] + radius_v
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
        offset = self._clearance_target_offset()*n
        self._gates = [g for g in gates
                       if all(self._in_fence(g+s*offset) for s in (-1, 0, 1))] or gates

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
        self.route = None   # which way the drone last went round, for replay.py
        self._crossed_pending = False
        self._lap_phase = 'approach'
        # Either feasible side is available until its back waypoint is reached.
        self._return_gate = None
        if (self.pos-self.hoop_centre) @ self.hoop_normal > self._clearance_reach():
            self._lap_phase = 'return_side'
        self._pending = deque([np.zeros(contract.ACTION_DIM, np.float32)] * self.dyn.delay_steps)
        self._frames = deque([self.observe()] * contract.FRAME_HISTORY,
                             maxlen=contract.FRAME_HISTORY)
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
        v_cmd = d.max_speed * np.array([forward * c + right * s, forward * s - right * c, up])
        yaw_cmd = -yaw_cw * np.deg2rad(d.max_yaw_rate_deg)  # heading is counter-clockwise
        alpha = 1.0 - np.exp(-dt / d.tau)
        self.vel += (v_cmd - self.vel) * alpha
        self.yaw_rate += (yaw_cmd - self.yaw_rate) * alpha

        prev = self.pos.copy()
        target = self._progress_target(prev)
        remaining_before = self._progress_distance(prev)
        self.pos = self.pos + self.vel * dt
        self.heading = (self.heading + self.yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi
        self.steps += 1

        outcome = self._outcome(prev)
        if outcome == "passed":
            self.passes += 1
            self._return_gate = None
            self._lap_phase = 'return_side'
        elif outcome not in CRASHES:
            if self._lap_phase == 'return_side':
                gate = self._return_side_gate(self.pos)
                back = gate+self._clearance_target_offset()*self.hoop_normal
                if np.linalg.norm(self.pos-back) < .05:
                    self._return_gate = gate.copy()
                    self._lap_phase = 'return_front'
            elif (self._lap_phase == 'return_front'
                  and np.linalg.norm(self.pos-target) < .05
                  and (self.pos-self.hoop_centre) @ self.hoop_normal < -self._clearance_reach()):
                self._lap_phase = 'align_front'
            elif self._lap_phase == 'align_front' and np.linalg.norm(self.pos-target) < .05:
                self._lap_phase = 'approach'
                self._return_gate = None
        obs = self.observe()
        self._frames.append(obs)
        self._actions.append(action)

        # One scalar distance is used on both sides of every transition,
        # including a pass's route reset. Undiscounted shaping telescopes to
        # initial minus final remaining distance. The 5 cm waypoint tolerance
        # permits a small corner shortcut, but phases cannot be repeated without
        # another credited pass and its corresponding route reset cost.
        progress = float(remaining_before-self._progress_distance(self.pos))
        terminated = outcome in CRASHES
        reward = (R_PROGRESS * progress / PROGRESS_UNIT - R_HIDDEN * (1.0 - obs[6])
                  + R_PASS * (outcome == "passed") - R_CRASH * terminated)
        truncated = not terminated and self.steps >= LAPS_MAX_STEPS
        if truncated:
            outcome = "timeout"
        return self._policy_input(), float(reward), terminated, truncated, self._info(outcome)

    def _progress_target(self, pos):
        """Next route point; crossing the centre plane alone never changes phase."""
        return self._progress_route(pos)[0]

    def _return_side_gate(self, pos):
        """Choose the shortest feasible return route without latching a side.

        Both gates have identical remaining leg lengths: they are equal-radius
        offsets on opposite sides of the same ring. Only the first distance
        differs. Taking this minimum is continuous at the side-choice boundary.
        """
        offset = self._clearance_target_offset()*self.hoop_normal
        return min(self._gates, key=lambda gate: np.linalg.norm(pos-(gate+offset)))

    def _progress_route(self, pos):
        offset = self._clearance_target_offset()*self.hoop_normal
        front = self.hoop_centre-offset
        exit_point = self.hoop_centre+offset
        if self._lap_phase == 'return_side':
            gate = self._return_side_gate(pos)
            return (gate+offset, gate-offset, front, exit_point)
        if self._lap_phase == 'return_front':
            return (self._return_gate-offset, front, exit_point)
        if self._lap_phase == 'align_front':
            return (front, exit_point)
        return (exit_point,)

    def _progress_distance(self, pos):
        """Remaining polyline length, continuous at exact route milestones.

        All later legs are already counted before a waypoint is reached. This
        supplies forward progress through a return without rewarding a repeatable
        waypoint cycle. A lap's distance resets only when a pass is credited.
        Geometry determines rewards only; none of this chooses policy actions.
        """
        route = self._progress_route(pos)
        return float(np.linalg.norm(pos-route[0])
                     + sum(np.linalg.norm(end-start) for start, end in zip(route, route[1:])))

    def _clearance_target_offset(self):
        # Static across yaw: rotating cannot move a reward target.
        return np.hypot(self.drone.length, self.drone.width)/2 + self.hoop.thickness/2 + .10

    def _clearance_reach(self):
        forward = np.array([np.cos(self.heading), np.sin(self.heading), 0.])
        side = np.array([-forward[1], forward[0], 0.])
        return (self.drone.length/2*abs(forward @ self.hoop_normal)
                + self.drone.width/2*abs(side @ self.hoop_normal)
                + self.hoop.thickness/2 + .01)

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
        strip_half = [self.hoop.thickness / 2, self.hoop.hanger_half_width]
        gap = self.pos[:2] - self.hoop_centre[:2]
        for axis in drone_axes + strip_axes:
            reach = (sum(h * abs(a @ axis) for h, a in zip(drone_half, drone_axes))
                     + sum(h * abs(a @ axis) for h, a in zip(strip_half, strip_axes)))
            if abs(gap @ axis) > reach:
                return False   # a separating axis: no contact
        return True

    def _left_cage(self) -> bool:
        z = self.pos[2]
        return not (self._in_fence(self.pos) and self.cage.min_altitude <= z <= self.cage.height - self.cage.margin)

    def _outcome(self, prev) -> str:
        """Front to back scores only after the entire box clears the ring.
        Crossing the plane outside the ring is how the drone gets round, so it is neutral.
        The hanger counts as part of the hoop."""
        self.route = None
        if self._hit() or self._hits_hanger():
            return "hit"
        if self._left_cage():
            return "left_cage"
        crossing = self._crossing(prev)
        if crossing is not None and crossing[1]:
            if crossing[0] == -1:
                self._crossed_pending = False
                return "wrong_way"
            self._crossed_pending = True
        if crossing is not None and crossing[0] == -1:
            self.route = self._route(crossing[2])
        if (self._crossed_pending
                and (self.pos-self.hoop_centre) @ self.hoop_normal > self._clearance_reach()):
            self._crossed_pending = False
            return "passed"
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

    def observe(self) -> np.ndarray:
        """The eight numbers the camera would report in the current state."""
        # One projection covers the outer edge and the marker, which is last.
        cam_pos, cam_axes = self._camera_pose()
        xn, yn, in_frame = self._project(self._sight_points, cam_pos, cam_axes)
        edge = in_frame[:-1]
        if edge.mean() < MIN_ARC_FRACTION:
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
        frames = [self._frames[-1 - lag] for lag in contract.OBS_LAGS]
        return np.concatenate([*frames, *self._actions]).astype(np.float32)

    def _info(self, outcome):
        return {"outcome": outcome, "passes": self.passes, "steps": self.steps, "route": self.route,
                "lap_phase": self._lap_phase,
                "level": self.level, "drone_pos": self.pos.copy(),
                "heading": self.heading, "hoop_centre": self.hoop_centre.copy(),
                "hoop_normal": self.hoop_normal.copy()}
