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
  - Never seen it, or not for the whole 12 s: climb until the height reads MAX_CLIMB,
    then sweep down to MIN_HEIGHT and back, turning, the way the last command went.
  - Lost it while lining up: turn or climb towards the edge it left by, without moving
    across the cage; lost big and mid-frame, it is too close, so back straight off to
    BACK_TO, then wait, then turn.
  - Lost it running in, big and centred: the pass and the way round, flown on the
    commands since that last sighting (contract.policy_input sums them per slot): on
    through to BEHIND past the hoop, out to GATE on the side it is already on, back past
    the plane to CLEAR in front, across to FRONT out on the axis, then turn round to face
    it. The hoop hangs within 0.15 m of the cage's middle, so every point of the way
    stays clear of the net in any layout (see FRONT).
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
MAX_CLIMB = 1.75     # m by the height reading: still blind here, so sweep back down
MIN_HEIGHT = 0.6     # m: never descend below this blind
TURN_YAW = 0.7       # every blind turn: 37 deg/s, 180 degrees in ~5 s
SWEEPING = 0.25      # |up command| that says a height sweep is under way

# Lost while lining up.
LOST_EDGE = 0.3      # |u| or |v| of the last sighting past which it left by that edge
LOST_UP = 0.4
LOST_WAIT = 20       # steps to hold still after losing it mid-frame before turning
# Lost big and mid-frame: it is too close to be reported, not off to one side. Over 60
# flights teacher3 spun on the spot 35% of the time 0.45-0.66 m in front of the hoop
# at the height to see it, where no heading brings it back. So it backs straight off,
# by the commands since the sighting, until the hoop's size puts it BACK_TO out.
BACK_TO = 0.75       # m from the hoop by the last sighting's size and the commands since
BACK_SPEED = 0.2     # m/s

# The pass and the way round, blind. RAN_AT_IT tells a run in from a line-up that lost a
# big, centred hoop: the run in commands 0.75 forward, the line-up near its stage little.
RAN_AT_IT = 0.4      # mean forward command in the slot of the last sighting
# Within 0.775 m of the hoop centre the drone's box clears the net in any layout: the
# hoop is within HOOP_FROM_CENTRE 0.15 of the middle of the 2.05 m cage, and the box
# reaches 0.1 m. The commands overstate the distance flown by up to ~10% (max_speed
# randomised), so the way round was kept inside 0.7 m. In the sim it falls short of the
# commands instead: aimed at FRONT 0.7 it turned round 0.55 m out (median, 200 flights),
# where the hoop is too big to be reported, and searched there blind. FRONT 0.85 turns
# round 0.59 m out (max 0.86) and passes 2.5 a flight, twice 0.7's, with no net crash.
BEHIND = 0.3         # m past the hoop's plane
GATE = 0.45          # m beside its centre: the rim and the drone's half-width reach 0.32
CLEAR = 0.3          # m in front of the plane, still at GATE, before heading in
FRONT = 0.85         # m out on the axis by the commands, where it turns round to see the hoop
ARRIVED = 0.04       # m from a corner of the way round, by the commands, to head for the next
K_BOX = 2.0          # velocity per metre from the corner, before the speed cap
RUN_SPEED = 0.3      # m/s, as the run in
ROUND_SPEED = 0.35   # m/s round the box
K_DZ = 1.0           # climb speed per metre of height error, by the commands

# Lining up, with the hoop in view: the furthest of these that fits ROOM inside the
# fence. Facing the hoop on its axis, the detector reports it from 0.65-0.9 m within
# ~0.1 m of the height that puts it mid-frame, from 0.55 m only at that height, and
# from 0.5 m not at all. Stages of 0.7, 0.6, 0.5 m lost it while lining up and never
# passed in 29% of flights; 0.9, 0.8, 0.7 m with ROOM 0.05 and a slower approach never
# passed in 5% (600 flights).
STAGES = (0.9, 0.8, 0.7)
LINE_UP_SPEED = 0.2  # m/s on the way to the stage point. At 0.4 it closed in fast with the
                     # hoop near the frame's edge, and on losing it coasted 15-20 cm nearer
STAGE_TOL = 0.06     # m from the stage point
LINED_UP = 0.03      # m off the axis
FACING = 3.0         # degrees off the axis. Run ins that started 3-6 degrees off hit the rim
                     # twice as often under demo noise: 7.2% of flights at 6, 3.8% at 3 (800)
STILL = 0.08         # m/s
TURNING = 2.0        # deg/s: still turning faster than this, it is not ready. Run ins that
                     # started still turning carried on 5-6 degrees (command delay and yaw lag)
                     # and drifted ~7 cm sideways in the blind last half metre, onto the rim:
                     # 13% of flights hit it, 0.5% with this (800 flights)
CENTRED = 0.15       # |u| to start the run in
GIVE_UP = 150        # steps lining up before it runs in anyway (a stage too close to see from)
RESET_BLIND = 5      # blind steps after which a sighting starts a fresh line-up

# The run in, down the line of sight: forward at RUN_SPEED and down along the pitch.
K_U = 1.5            # strafe command per unit of u
K_V = 1.0            # up command per unit of v

K_POS = 0.9          # velocity command per metre of position error
K_YAW = 1.5          # yaw command per radian of heading error
ROOM = 0.05          # m inside the fence (cage.margin in from the net) for the stage point


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
        # Never seen it, or not for 12 s: sweep the heights between MIN_HEIGHT and
        # MAX_CLIMB, carrying on the way the last command went. Never seen, it first
        # climbs straight; after that it turns as it sweeps.
        # Lost after flying round, it ends at about the height to see it from, turned
        # away (105 degrees median): turn a full circle there before sweeping off it.
        turned = np.abs(actions[:, YAW] * WIDTHS).sum() * _DRONE.max_yaw_rate_deg * _DRONE.dt
        if not unseen and turned < 360.0:
            return _action(up=_safe_up(0.0, height), yaw=TURN_YAW)
        going = actions[-1, UP]   # with demo noise on it, so read its sign past SWEEPING
        if abs(going) < SWEEPING:
            going = SEARCH_UP if unseen else -SEARCH_UP
        up = SEARCH_UP if going > 0 else -SEARCH_UP
        if height >= MAX_CLIMB:
            up = -SEARCH_UP
        elif height <= MIN_HEIGHT:
            up = SEARCH_UP
        first_climb = unseen and up > 0 and not (actions[:, UP] < -SWEEPING).any()
        return _action(up=up, yaw=0.0 if first_climb else TURN_YAW)
    k = int(seen[-1])
    last = frames[k]
    # Steps in slot k after its last sighting, taking its sightings to come first.
    after = (1.0 - last[VISIBLE]) * WIDTHS[k]
    if (last[MAJOR] >= contract.PASS_MAJOR and abs(last[U]) < contract.PASS_U
            and actions[k, FORWARD] >= RAN_AT_IT):
        return _through_and_round(last, actions, k, after, height)
    backed = -(actions[k:, FORWARD] * np.concatenate([[after], WIDTHS[k + 1:]])).sum() * STEP
    return _lost(last, after + WIDTHS[k + 1:].sum(), backed, height)


def _lost(last: np.ndarray, age: float, backed: float, height: float) -> np.ndarray:
    """Lost while lining up: turn or climb towards the edge it left by, holding position.
    Lost big and mid-frame, back straight off to BACK_TO first."""
    yaw = TURN_YAW * np.sign(last[U]) if abs(last[U]) > LOST_EDGE else 0.0
    # v is down. At a height limit the climb is cut to nothing, so it turns instead.
    up = _safe_up(-LOST_UP * np.sign(last[V]) if abs(last[V]) > LOST_EDGE else 0.0, height)
    if not yaw and not up:
        if DIAMETER / last[MAJOR] + backed < BACK_TO:
            return _action(forward=-BACK_SPEED / _DRONE.max_speed, up=_safe_up(0.0, height))
        if age > LOST_WAIT:
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
    if leg == 0 and along[-1] < 0:
        # Not through yet: straight on. The hoop was half out of frame at the last
        # sighting, and the offset beside it read from that had no correlation with the
        # true one (r -0.02 over 2000 run ins), so steering on it only added error.
        v[1] = 0.0
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
                     and abs(np.degrees(env.yaw_rate)) < TURNING
                     and abs(_wrap(self._axis_heading(env) - env.heading)) < np.deg2rad(FACING))
            if not (ready and (abs(frame[U]) < CENTRED or self.lining > GIVE_UP)):
                # On the way, hold the height that keeps the hoop mid-frame.
                # Below the roof net by cage.margin: from 0.9 m a high hoop's sight height
                # is up to 1.94 m, and with demo noise it touched the roof (1% of flights).
                z = min(max(stage[2], self._sight_z(env, env.pos)), env.cage.height - env.cage.margin)
                target = np.array([*stage[:2], z])
                return self._towards(env, target, self._facing_hoop(env), LINE_UP_SPEED)
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

    def _towards(self, env, target, heading, speed=None):
        """Body-frame command towards a point, facing a heading, at up to `speed` m/s."""
        d = env.drone
        speed = d.max_speed if speed is None else min(speed, d.max_speed)
        v = K_POS * (target - env.pos)
        flat = np.linalg.norm(v[:2])
        if flat > speed:
            v[:2] *= speed / flat
        c, s = np.cos(env.heading), np.sin(env.heading)
        # Past the measured yaw deadband, so the turn is K_YAW times the error however
        # small: K_YAW alone asks 0.08 at 3 degrees off, which does not turn the drone.
        yaw = -K_YAW * _wrap(heading - env.heading)
        db = _DRONE.yaw_deadband
        yaw = np.sign(yaw) * (db + (1.0 - db) * abs(yaw)) if abs(yaw) > 1e-3 else 0.0
        return _action(right=(v[0] * s - v[1] * c) / d.max_speed, forward=(v[0] * c + v[1] * s) / d.max_speed,
                       up=v[2] / d.max_up_speed, yaw=yaw)


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
