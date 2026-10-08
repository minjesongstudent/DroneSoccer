"""A demonstrator that flies the whole mission in the simulator, for train.py --demos.

It flies like the real flight should: it takes off blind and climbs until the detector
reports the hoop, lines up in front of it, flies through, goes round a side picked at
random each lap, and lines up again until the minute is up. Its flights only go in the
replay buffer as examples. The policy still learns its own search and lap, and nothing
here runs on the drone.

The policy can only learn from a step whose reason it can see. The first teacher
steered on the simulator's true positions while the camera saw nothing (its line-up 79%
blind, its run in 89%), so the buffer said "no hoop in view, fly forward" and the
policies trained on it flew into the rim. This one uses true positions only to get
roughly into place and to go round; every correction near the hoop is made while the
detector reports it, and the run in is steered on the camera alone:
  - It lines up STAGE out on the axis, raised so the hoop sits in the middle of the
    frame. Only there is it reported close in: on the axis at 0.6 m the detector sees
    it from 0.06-0.24 m above its centre (0.14-0.16 m for the strictest draw of
    Detector.min_arc_fraction), never at its height.
  - The run in flies down the camera's line of sight, forward and down along its
    pitch, which ends at the hoop's centre. While the hoop is reported it strafes and
    climbs to keep it centred; once the hoop fills the frame it flies straight on.

Run: python -m src.teacher   (how well it flies, at level 1 with randomise on)
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

from src import contract
from src.sim import CLEARANCE, HoopEnv

U, V = contract.FIELDS.index("u"), contract.FIELDS.index("v")

# Search: climb at half speed. The hoop is reported a few steps after it comes into
# view (2-frame confirm, then up to 4 steps of command delay), and hoops hang as little
# as 10 cm below the roof net, so a full-speed climb overshoots into it.
SEARCH_UP = 0.5
MAX_CLIMB = 1.75     # m: still blind here, so turn on the spot until it is seen
SEARCH_YAW = 0.5

# Lining up: the furthest of these that fits 0.15 m inside the fence. 0.6 m fits 99%
# of layouts, 0.7 m 77%.
STAGES = (0.7, 0.6, 0.5)
STAGE_TOL = 0.06     # m from the stage point
LINED_UP = 0.03      # m off the axis
FACING = 6.0         # degrees off the axis
STILL = 0.08         # m/s
CENTRED = 0.15       # |u| with the hoop reported, to start the run in
LOST = 5             # steps without the hoop while lining up before it backs off
BACK_OFF = 0.25      # m further out than it is, until the hoop is back
GIVE_UP = 150        # steps lining up before it runs in anyway (a stage too close to see from)

# The run in, down the line of sight: forward at RUN_SPEED and down along the pitch.
RUN_SPEED = 0.3      # m/s
K_U = 1.5            # strafe command per unit of u while the hoop is reported
K_V = 1.0            # up command per unit of v
RUN_BLIND = 40       # steps without the hoop and without passing: line up again

K_POS = 0.9          # velocity command per metre of position error
K_YAW = 1.5          # yaw command per radian of heading error
WAYPOINT = 0.1       # m: close enough to a way-round point to head for the next
ROOM = 0.15          # m inside the fence for every point it flies to: lag and delay overshoot
ROUND_SPEED = 0.3    # m/s going round


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Teacher:
    """Call reset() after env.reset(), then act(env) each step. rng picks the side
    it goes round each lap."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def reset(self, env: HoopEnv):
        self.phase = "search"
        self.route: list[np.ndarray] = []
        self.blind = 0      # steps since the hoop was last reported
        self.lining = 0     # steps in this line-up
        self.backing = None
        self.run_ins = self.run_ins_seen = 0

    # --- geometry ----------------------------------------------------------

    def _slope(self, env):
        return np.tan(np.deg2rad(env.camera.pitch_deg))

    def _sight_z(self, env, pos):
        """The height at which the hoop sits in the middle of the frame from pos."""
        return env.hoop_centre[2] + self._slope(env) * np.linalg.norm((pos - env.hoop_centre)[:2])

    def _fence(self, env):
        lo = env.cage.margin + ROOM
        hi = np.array([env.cage.length, env.cage.width]) - env.cage.margin - ROOM
        return lo, hi

    def _inside(self, env, p):
        lo, hi = self._fence(env)
        return np.array([*np.clip(p[:2], lo, hi), p[2]])

    def _stage(self, env):
        lo, hi = self._fence(env)
        for out in STAGES:
            p = env.hoop_centre - out * env.hoop_normal
            if (p[:2] >= lo).all() and (p[:2] <= hi).all():
                break
        return np.array([*p[:2], env.hoop_centre[2] + self._slope(env) * out])

    def _axis_heading(self, env):
        return float(np.arctan2(env.hoop_normal[1], env.hoop_normal[0]))

    def _facing_hoop(self, env):
        to = env.hoop_centre[:2] - env.pos[:2]
        if np.linalg.norm(to) < 0.3:
            return self._axis_heading(env)
        return float(np.arctan2(to[1], to[0]))

    # --- commands ----------------------------------------------------------

    def _command(self, env, v_world, heading):
        """Body-frame action for a world velocity (m/s) and a wanted heading."""
        d = env.drone
        v = np.asarray(v_world, float)
        flat = np.linalg.norm(v[:2])
        if flat > d.max_speed:
            v[:2] *= d.max_speed / flat
        c, s = np.cos(env.heading), np.sin(env.heading)
        forward = (v[0] * c + v[1] * s) / d.max_speed
        right = (v[0] * s - v[1] * c) / d.max_speed
        up = v[2] / d.max_up_speed
        yaw_cw = -K_YAW * _wrap(heading - env.heading)
        return np.clip([right, forward, up, yaw_cw], -1.0, 1.0).astype(np.float32)

    def _towards(self, env, target, heading, speed=None):
        v = K_POS * (target - env.pos)
        flat = np.linalg.norm(v[:2])
        if speed is not None and flat > speed:
            v[:2] *= speed / flat
        return self._command(env, v, heading)

    # --- the mission -------------------------------------------------------

    def act(self, env: HoopEnv) -> np.ndarray:
        frame = env._frames[-1]
        seen = bool(frame[contract.VISIBLE])
        self.blind = 0 if seen else self.blind + 1
        along = (env.pos - env.hoop_centre) @ env.hoop_normal

        if self.phase == "search":
            if not seen:
                if env.pos[2] < MAX_CLIMB:
                    return np.array([0.0, 0.0, SEARCH_UP, 0.0], np.float32)
                return np.array([0.0, 0.0, 0.0, SEARCH_YAW], np.float32)
            self._line_up()

        if self.phase == "line_up":
            self.lining += 1
            stage = self._stage(env)
            ready = (np.linalg.norm(env.pos - stage) < STAGE_TOL
                     and abs((env.pos - env.hoop_centre) @ env._axis_h) < LINED_UP
                     and np.linalg.norm(env.vel) < STILL
                     and abs(_wrap(self._axis_heading(env) - env.heading)) < np.deg2rad(FACING))
            if ready and ((seen and abs(frame[U]) < CENTRED) or self.lining > GIVE_UP):
                self.phase, self.blind_run = "run_in", 0
                self.run_ins += 1
                self.run_ins_seen += seen
            else:
                if self.blind > LOST and self.backing is None:
                    # Lost it before lining up: back away from the hoop until it is back.
                    away = env.pos[:2] - env.hoop_centre[:2]
                    away = away / (np.linalg.norm(away) or 1.0)
                    p = self._inside(env, np.array([*(env.pos[:2] + BACK_OFF * away), 0.0]))
                    p[2] = self._sight_z(env, p)
                    self.backing = p
                if seen:
                    self.backing = None
                target = self.backing if self.backing is not None else stage
                if self.backing is None:
                    # On the way, hold the height that keeps the hoop mid-frame.
                    target = np.array([*target[:2], max(target[2], self._sight_z(env, env.pos))])
                return self._towards(env, target, self._facing_hoop(env))

        if self.phase == "run_in":
            if along > CLEARANCE:
                self._plan_round(env)
                self.phase = "round"
            elif not seen and self.blind > RUN_BLIND:
                self._line_up()
                return self.act(env)
            else:
                # Down the line of sight, steered on the camera while it shows the hoop.
                d = env.drone
                forward = RUN_SPEED / d.max_speed
                up = -RUN_SPEED * self._slope(env) / d.max_up_speed
                right = 0.0
                if seen:
                    right = K_U * frame[U]
                    up -= K_V * frame[V]
                return np.clip([right, forward, up, 0.0], -1.0, 1.0).astype(np.float32)

        # Round the side: behind the gate, the gate, in front of it, then line up.
        while self.route and np.linalg.norm(env.pos - self.route[0]) < WAYPOINT:
            self.route.pop(0)
        if not self.route:
            self._line_up()
            return self.act(env)
        return self._towards(env, self.route[0], self._facing_hoop(env), ROUND_SPEED)

    def _line_up(self):
        self.phase, self.lining, self.backing = "line_up", 0, None

    def _plan_round(self, env):
        """The way round that _to_go rewards, past a gate picked at random each lap.
        Both have room: sim._sample_layout only makes layouts where they do."""
        gate = env._gates[self.rng.integers(len(env._gates))]
        n = env.hoop_normal
        front = gate - CLEARANCE * n
        front[2] = self._sight_z(env, front)   # where the hoop will be in view again
        self.route = [self._inside(env, p) for p in (gate + CLEARANCE * n, gate, front)]


def run(episodes: int, level: float = 1.0, seed: int = 0, noise: float = 0.0) -> collections.Counter:
    """How the teacher's flights end, over `episodes` minutes at `level`."""
    env, out = HoopEnv(level=level, randomise=True), collections.Counter()
    rng = np.random.default_rng(seed)
    teacher = Teacher(rng)
    for i in range(episodes):
        env.reset(seed=seed + i)
        teacher.reset(env)
        done = False
        while not done:
            action = np.clip(teacher.act(env) + rng.normal(0.0, noise, contract.ACTION_DIM), -1.0, 1.0)
            _, _, terminated, truncated, info = env.step(action.astype(np.float32))
            done = terminated or truncated
        out[info["outcome"]] += 1
        out["passes"] += info["passes"]
        out["passed"] += info["passes"] > 0
        out["run_ins"] += teacher.run_ins
        out["run_ins_seen"] += teacher.run_ins_seen
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.0)
    args = parser.parse_args()
    c = run(args.episodes, args.level, noise=args.noise)
    n = args.episodes
    print(f"passes/flight {c['passes'] / n:.2f}  passed {c['passed'] / n:.0%}  "
          f"run ins with the hoop in view {c['run_ins_seen'] / max(c['run_ins'], 1):.0%}  "
          + "  ".join(f"{k} {c[k] / n:.0%}" for k in (*contract.CRASHES, "timeout")))
