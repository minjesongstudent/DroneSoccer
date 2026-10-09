"""A demonstrator that flies the whole mission in the simulator, for train.py --demos.

It flies like the real flight should: it takes off blind and climbs until the detector
reports the hoop, lines up in front of it, flies through, goes round, and lines up again
until the minute is up. Its flights only go in the replay buffer as examples. The policy
still learns its own search and lap, and nothing here runs on the drone.

The policy can only learn from a step whose reason it can see, so whenever the hoop is
not reported the teacher acts on the policy's own input and nothing else: blind_action
takes that input and no simulator. teacher2 steered its blind stretches on true state (a
climb cut off at a true height, a way round on true waypoints, a back-off kept inside
the net), which the policy saw only as identical blank frames. Its students learned the
average of all of them, one fixed drift held until it hit the roof, the net or the rim.

Blind, from the policy input:
  - Never seen it: climb until the height reads MAX_CLIMB, then turn on the spot.
  - Lost it while lining up: turn or climb towards the edge it left by, without moving
    across the cage; lost mid-frame, wait, then turn.
  - Lost it running in, big and centred: the pass and the way round, flown on the
    commands since that last sighting (contract.policy_input sums them per slot): on
    through to BEHIND past the hoop, out to GATE on the side it is already on, back past
    the plane to CLEAR in front, across to FRONT out on the axis, then turn round to face
    it. The hoop hangs within 0.15 m of the cage's middle, so every point of the way
    stays within 0.75 m of its centre and clear of the net in any layout.
With the hoop in view it may use true state, as teacher2 did, because the ellipse shows
roughly where the hoop is: it lines up STAGE out on the axis, raised so the hoop sits
mid-frame, and runs in down the camera's line of sight, steered on u and v.

Run: python -m src.teacher   (how well it flies, at level 1 with randomise on)
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

from src import contract
from src.params import Camera, Drone, Hoop
from src.sim import HoopEnv

U, V, MAJOR = (contract.FIELDS.index(k) for k in ("u", "v", "major"))
VISIBLE = contract.VISIBLE
RIGHT, FORWARD, UP, YAW = range(contract.ACTION_DIM)

# Blind, the teacher knows only what the policy's designer knows: the measured setup,
# never the episode's randomised draw of it.
_CAMERA, _DRONE, _HOOP = Camera(), Drone(), Hoop()
PITCH = np.deg2rad(_CAMERA.pitch_deg)
SLOPE = np.tan(PITCH)
HALF_X, HALF_Y = _CAMERA.half_extent
DIAMETER = (_HOOP.outer_diameter + _HOOP.outer_height) / 2
STEP = _DRONE.max_speed * _DRONE.dt         # m flown in a step at full command
STEP_UP = _DRONE.max_up_speed * _DRONE.dt
WIDTHS = np.array(contract.SLOT_STEPS, float)

# Search: climb at half speed. The hoop is reported a few steps after it comes into
# view (2-frame confirm, then up to 4 steps of command delay), and hoops hang as little
# as 10 cm below the roof net, so a full-speed climb overshoots into it. A straight
# climb from the pad brings the hoop into view in 200 of 200 eval layouts.
SEARCH_UP = 0.5
MAX_CLIMB = 1.75     # m by the height reading: still blind here, so turn on the spot
MIN_HEIGHT = 0.6     # m: never descend below this blind
TURN_YAW = 0.7       # every blind turn: 37 deg/s, 180 degrees in ~5 s

# Lost while lining up.
LOST_EDGE = 0.3      # |u| or |v| of the last sighting past which it left by that edge
LOST_UP = 0.4
LOST_WAIT = 20       # steps to hold still after losing it mid-frame before turning

# The pass and the way round, blind. RAN_AT_IT tells a run in from a line-up that lost a
# big, centred hoop: the run in commands 0.75 forward, the line-up near its stage little.
RAN_AT_IT = 0.4      # mean forward command in the slot of the last sighting
# Within 0.775 m of the hoop centre the drone's box clears the net in any layout: the
# hoop is within HOOP_FROM_CENTRE 0.15 of the middle of the 2.05 m cage, and the box
# reaches 0.1 m. The commands overstate the distance flown by up to ~10% (max_speed
# randomised), so the way round keeps inside 0.7 m.
BEHIND = 0.3         # m past the hoop's plane
GATE = 0.45          # m beside its centre: the rim and the drone's half-width reach 0.32
CLEAR = 0.3          # m in front of the plane, still at GATE, before heading in
FRONT = 0.7          # m out on the axis, where it turns round: the hoop is reported there
ARRIVED = 0.04       # m from a corner of the way round, by the commands, to head for the next
K_BOX = 2.0          # velocity per metre from the corner, before the speed cap
RUN_SPEED = 0.3      # m/s, as the run in
ROUND_SPEED = 0.35   # m/s round the box
K_DZ = 1.0           # climb speed per metre of height error, by the commands

# Lining up, with the hoop in view: the furthest of these that fits 0.15 m inside the
# fence. 0.6 m fits 99% of layouts, 0.7 m 77%.
STAGES = (0.7, 0.6, 0.5)
STAGE_TOL = 0.06     # m from the stage point
LINED_UP = 0.03      # m off the axis
FACING = 6.0         # degrees off the axis
STILL = 0.08         # m/s
CENTRED = 0.15       # |u| to start the run in
GIVE_UP = 150        # steps lining up before it runs in anyway (a stage too close to see from)
RESET_BLIND = 5      # blind steps after which a sighting starts a fresh line-up

# The run in, down the line of sight: forward at RUN_SPEED and down along the pitch.
K_U = 1.5            # strafe command per unit of u
K_V = 1.0            # up command per unit of v

K_POS = 0.9          # velocity command per metre of position error
K_YAW = 1.5          # yaw command per radian of heading error
ROOM = 0.15          # m inside the fence for the stage point


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _action(right=0.0, forward=0.0, up=0.0, yaw=0.0) -> np.ndarray:
    return np.clip([right, forward, up, yaw], -1.0, 1.0).astype(np.float32)


def _safe_up(up: float, height: float) -> float:
    """Never climb past MAX_CLIMB or sink below MIN_HEIGHT by the height reading."""
    if height >= MAX_CLIMB:
        up = min(up, 0.0)
    if height <= MIN_HEIGHT:
        up = max(up, 0.0)
    return up


# --- blind: the policy input and nothing else ---------------------------------------

def blind_action(x: np.ndarray) -> np.ndarray:
    """What the teacher does while the hoop is not reported, from the policy input x alone."""
    frames, heights, actions, unseen = contract.split_input(x)
    height = float(heights[-1])
    seen = np.flatnonzero(frames[:, VISIBLE])
    if not len(seen):
        if unseen and height < MAX_CLIMB:
            return _action(up=SEARCH_UP)
        return _action(up=_safe_up(0.0, height), yaw=TURN_YAW)
    k = int(seen[-1])
    last = frames[k]
    # Steps in slot k after its last sighting, taking its sightings to come first.
    after = (1.0 - last[VISIBLE]) * WIDTHS[k]
    if (last[MAJOR] >= contract.PASS_MAJOR and abs(last[U]) < contract.PASS_U
            and actions[k, FORWARD] >= RAN_AT_IT):
        return _through_and_round(last, actions, k, after, height)
    return _lost(last, after + WIDTHS[k + 1:].sum(), height)


def _lost(last: np.ndarray, age: float, height: float) -> np.ndarray:
    """Lost while lining up: turn or climb towards the edge it left by, holding position."""
    yaw = TURN_YAW * np.sign(last[U]) if abs(last[U]) > LOST_EDGE else 0.0
    up = -LOST_UP * np.sign(last[V]) if abs(last[V]) > LOST_EDGE else 0.0   # v is down
    if not yaw and not up and age > LOST_WAIT:
        yaw = TURN_YAW
    return _action(up=_safe_up(up, height), yaw=yaw)


def _through_and_round(last, actions, k, after, height) -> np.ndarray:
    """On through the hoop and round the box, on the commands since the last sighting."""
    distance = DIAMETER / last[MAJOR]
    hoop_right = distance * last[U] * HALF_X
    hoop_below = distance * np.tan(PITCH + np.arctan(last[V] * HALF_Y))
    side = 1.0 if hoop_right <= 0 else -1.0   # the side of the axis it is already on
    # Where the commands have taken it since the sighting, at the end of each slot:
    # forward and right of where it was then, and how far it climbed.
    steps = np.concatenate([[after], WIDTHS[k + 1:]])
    moved = actions[k:] * steps[:, None]
    path = np.cumsum(moved[:, [FORWARD, RIGHT]], axis=0) * STEP
    along = path[:, 0] - distance        # + behind the hoop's plane
    beside = path[:, 1] - hoop_right     # + right of its centre
    climbed = moved[:, UP].sum() * STEP_UP
    corners = [(BEHIND, 0.0), (BEHIND, side * GATE), (-CLEAR, side * GATE), (-FRONT, 0.0)]
    leg = 0
    for a, b in zip(along, beside):
        if leg == 0 and a >= BEHIND - ARRIVED:
            leg = 1
        if leg == 1 and side * b >= GATE - ARRIVED:
            leg = 2
        if leg == 2 and a <= -CLEAR + ARRIVED:
            leg = 3
        if leg == 3 and np.hypot(a + FRONT, b) <= ARRIVED:
            leg = 4
    # Through and round at the hoop's height, then up to see it mid-frame from FRONT.
    dz = -hoop_below + (FRONT * SLOPE if leg >= 3 else 0.0) - climbed
    if leg == 0 and along[-1] < 0:
        # Not through yet: down the line of sight, as the run in flew.
        up = -RUN_SPEED * SLOPE / _DRONE.max_up_speed
    else:
        up = K_DZ * dz / _DRONE.max_up_speed
    up = _safe_up(up, height)
    if leg == 4:
        return _action(up=up, yaw=TURN_YAW)
    v = K_BOX * (np.array(corners[leg]) - [along[-1], beside[-1]])
    speed = RUN_SPEED if leg == 0 else ROUND_SPEED
    v *= min(1.0, speed / max(np.linalg.norm(v), 1e-9))
    return _action(right=v[1] / _DRONE.max_speed, forward=v[0] / _DRONE.max_speed, up=up)


# --- with the hoop in view: true state allowed ---------------------------------------

class Teacher:
    """Call reset() after env.reset(), then act(env, obs) each step with the policy
    input env last returned."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()   # kept for train.add_demos' call

    def reset(self, env: HoopEnv | None = None):
        self.phase = "line_up"
        self.blind = 0      # blind steps in a row
        self.lining = 0     # steps in this line-up
        self.run_ins = 0

    def act(self, env: HoopEnv, obs: np.ndarray) -> np.ndarray:
        x = np.array(obs, np.float32)
        x.flags.writeable = False
        frame = contract.split_input(x)[0][-1]
        if not frame[VISIBLE]:
            self.blind += 1
            return blind_action(x)
        if self.blind >= RESET_BLIND:
            self.phase, self.lining = "line_up", 0
        self.blind = 0
        if self.phase == "line_up":
            self.lining += 1
            stage = self._stage(env)
            ready = (np.linalg.norm(env.pos - stage) < STAGE_TOL
                     and abs((env.pos - env.hoop_centre) @ env._axis_h) < LINED_UP
                     and np.linalg.norm(env.vel) < STILL
                     and abs(_wrap(self._axis_heading(env) - env.heading)) < np.deg2rad(FACING))
            if not (ready and (abs(frame[U]) < CENTRED or self.lining > GIVE_UP)):
                # On the way, hold the height that keeps the hoop mid-frame.
                target = np.array([*stage[:2], max(stage[2], self._sight_z(env, env.pos))])
                return self._towards(env, target, self._facing_hoop(env))
            self.phase = "run_in"
            self.run_ins += 1
        # Down the line of sight, steered on the camera.
        d = env.drone
        forward = RUN_SPEED / d.max_speed
        up = -RUN_SPEED * self._slope(env) / d.max_up_speed - K_V * frame[V]
        return _action(right=K_U * frame[U], forward=forward, up=up)

    # --- geometry ----------------------------------------------------------

    def _slope(self, env):
        return np.tan(np.deg2rad(env.camera.pitch_deg))

    def _sight_z(self, env, pos):
        """The height at which the hoop sits in the middle of the frame from pos."""
        return env.hoop_centre[2] + self._slope(env) * np.linalg.norm((pos - env.hoop_centre)[:2])

    def _stage(self, env):
        lo = env.cage.margin + ROOM
        hi = np.array([env.cage.length, env.cage.width]) - lo
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

    def _towards(self, env, target, heading):
        """Body-frame command towards a point, facing a heading."""
        d = env.drone
        v = K_POS * (target - env.pos)
        flat = np.linalg.norm(v[:2])
        if flat > d.max_speed:
            v[:2] *= d.max_speed / flat
        c, s = np.cos(env.heading), np.sin(env.heading)
        return _action(right=(v[0] * s - v[1] * c) / d.max_speed, forward=(v[0] * c + v[1] * s) / d.max_speed,
                       up=v[2] / d.max_up_speed, yaw=-K_YAW * _wrap(heading - env.heading))


def run(episodes: int, level: float = 1.0, seed: int = 0, noise: float = 0.0) -> collections.Counter:
    """How the teacher's flights end, over `episodes` minutes at `level`."""
    env, out = HoopEnv(level=level, randomise=True), collections.Counter()
    rng = np.random.default_rng(seed)
    teacher = Teacher(rng)
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        teacher.reset(env)
        done = False
        while not done:
            action = np.clip(teacher.act(env, obs) + rng.normal(0.0, noise, contract.ACTION_DIM), -1.0, 1.0)
            obs, _, terminated, truncated, info = env.step(action.astype(np.float32))
            done = terminated or truncated
            out["steps"] += 1
            out["blind_steps"] += not contract.split_input(obs)[0][-1][VISIBLE]
        out[info["outcome"]] += 1
        out["passes"] += info["passes"]
        out["passed"] += info["passes"] > 0
        out["run_ins"] += teacher.run_ins
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
          f"passes per run in {c['passes'] / max(c['run_ins'], 1):.0%}  blind {c['blind_steps'] / c['steps']:.0%}  "
          + "  ".join(f"{k} {c[k] / n:.0%}" for k in (*contract.CRASHES, "timeout")))
