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
    commands since that last sighting (contract.policy_input sums them per slot),
    scaled to this flight's top speed by the speed readings (_flown): down to the hoop's
    centre in full-scale pulses, on through to BEHIND past the hoop, holding the line it
    was on, out to GATE on the side it is already on, back past the plane to CLEAR in
    front, across to FRONT out on the axis, then turn round to face it. The leg of the
    way round it has reached is kept until the hoop is seen again (Teacher.leg). The
    hoop hangs within 0.15 m of the cage's middle, so every point of the way stays clear
    of the net in any layout (see FRONT).
Its commands assume the middle of the randomised top speed and up/down response (SPEED,
UP_EXPONENT), which the policy cannot see either.
With the hoop in view it may use true state, as teacher2 did, because the ellipse shows
roughly where the hoop is: it lines up STAGE out on the axis, raised so the hoop sits
mid-frame, and runs in down the camera's line of sight, steered on v, held on the axis.

2026-10-13 on the full ranges (150 flights a corner, top speed 0.55/0.8 x exponent
1/1.6): 2.7-3.1 passes a flight, 1-5% crashed, passes 1.7 cm from the centre's height
and 0.6 cm beside it (median). With noise 0.1 on its commands: 1.6-1.8, 14-21% crashed.

Run: python -m src.teacher   (how well it flies, at level 1 with randomise on)
"""
from __future__ import annotations

import argparse
import collections

import numpy as np

from src import contract
from src.params import Camera, Drone, Hoop, Randomise
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
WIDTHS = np.array(contract.SLOT_STEPS, float)
# The episode's top speed and up/down response are drawn per flight (Randomise.max_speed
# 0.55-0.8, up_exponent 1-1.6) and the policy never sees the draw, so commands assume the
# middle of each range. Speeds then come out within +-20% of what was asked. Assuming
# Drone.max_speed 0.4 made teacher3 fly 1.4-2x what it planned: it overran the hoop into
# the net after passes and reached the rim before it had descended.
SPEED = float(np.mean(Randomise().max_speed))
UP_EXPONENT = float(np.mean(Randomise().up_exponent))
# The speed readings (contract.MOTION_FIELDS) are whole dm/s cut towards 0 on each axis,
# so they read ~0.05 m/s low while moving.
TELEMETRY_BIAS = 0.05
# The drone's response at the middle of the randomised ranges (Randomise.delay_steps,
# Randomise.tau), for _speed_scale.
DELAY = int(round(np.mean(Randomise().delay_steps)))
TAU = float(np.mean(Randomise().tau))
SCALE_RANGE = (0.7, 1.4)   # the speed readings' scale on the modelled path: the speed range's ends
SCALE_MIN = 0.5            # sum of squared modelled speeds over the window, (m/s)^2, before the readings count


def _up(speed: float) -> float:
    """The up command for a climb (+) or descent (-) in m/s. Small commands are weak
    (|command| ** up_exponent), so a descent of a few cm/s needs a bigger command than
    a linear drone would."""
    c = speed / _DRONE.max_up_speed
    return float(np.sign(c) * min(abs(c), 1.0) ** (1.0 / UP_EXPONENT))


def _pulse(dz: float) -> float:
    """The up command that moves the drone dz metres up as one full-scale pulse: 1 or -1
    until less than a step's worth is left, which goes as a part command."""
    return float(np.clip(dz / (_DRONE.max_up_speed * _DRONE.dt), -1.0, 1.0))


def _speed_scale(asked: np.ndarray, speeds: np.ndarray) -> float:
    """This flight's top speed over SPEED, from the policy input: each slot's speed reading
    against the speed the commands would give by then, through the sim's response at the
    middle of the randomised delay and lag, weighted by that speed, so slow stretches
    (which read 0 below 0.1 m/s) count for little and nothing switches in or out. Over
    teacher flights on the way round: 0.80 for 0.81 (p10-p90 0.74-0.87), 1.18 for 1.19."""
    per_step = np.repeat(asked, WIDTHS.astype(int), axis=0)
    per_step = np.vstack([np.zeros((DELAY, 2)), per_step[:len(per_step) - DELAY]])
    alpha = 1.0 - np.exp(-_DRONE.dt / TAU)
    v, model = np.zeros(2), np.zeros(len(per_step))
    for i, c in enumerate(per_step):
        v += (c * SPEED - v) * alpha
        model[i] = np.linalg.norm(v)
    model = model[np.cumsum(WIDTHS.astype(int)) - 1]
    weight = (model ** 2).sum()
    if weight < SCALE_MIN:
        return 1.0
    read = speeds + TELEMETRY_BIAS * (speeds > 0)
    return float(np.clip((read * model).sum() / weight, *SCALE_RANGE))


def _flown(actions: np.ndarray, speeds: np.ndarray, steps: np.ndarray, k: int) -> np.ndarray:
    """(forward, right) metres in each slot from slot k on: where the commands will have
    taken the drone once it settles, at this flight's speed (_speed_scale).

    Settled, not where it is now: steering on the drone's present place (its commands
    through the delay and lag) left 0.25 m of momentum out, and a fast drone braked late,
    0.5 m behind the hoop and 0.66 m beside it, towards the net. A drone answering a command
    through a lag ends up where the command's sum says, so this is where it will stop.
    The top speed is drawn per flight (+-19% of SPEED). On SPEED alone a slow drone slid
    sideways 0.06 m behind the rim and a fast one went 0.7 m behind, where some layouts
    leave 0.55 m to the net. Readings paired with each slot's own command were unstable:
    which way a slot went flipped with every reversal of the commands."""
    asked = actions[:, [FORWARD, RIGHT]]
    asked = asked / np.maximum(1.0, np.linalg.norm(asked, axis=1))[:, None]   # sim.velocity_command
    return asked[k:] * (_speed_scale(asked, speeds) * SPEED * _DRONE.dt * steps)[:, None]


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
RAN_AT_IT = 0.4      # mean forward command in the slot before the last sighting's. The sighting's own slot
                     # also holds the blind steps after it, whose back-off commands, under DAgger's noise,
                     # flipped this test each step: the drone crept into the hoop and out of it, wrong_way
# Within 0.775 m of the hoop centre the drone's box clears the net in any layout: the
# hoop is within HOOP_FROM_CENTRE 0.15 of the middle of the 2.05 m cage, and the box
# reaches 0.1 m. The commands overstate the distance flown by up to ~10% (max_speed
# randomised), so the way round was kept inside 0.7 m. In the sim it falls short of the
# commands instead: aimed at FRONT 0.7 it turned round 0.55 m out (median, 200 flights),
# where the hoop is too big to be reported, and searched there blind. FRONT 0.85 turns
# round 0.59 m out (max 0.86) and passes 2.5 a flight, twice 0.7's, with no net crash.
BEHIND = 0.3         # m past the hoop's plane by the commands. 0.25 left a 0.55 m/s drone too near the
                     # back of the rim when it slid sideways (9% of flights hit it)
GATE = 0.52          # m beside its centre by the commands: the rim and the drone's half-width reach
                     # 0.32. At 0.45 a 0.55 m/s drone, flying 0.81 of what SPEED plans, came back past
                     # the rim 0.30 m out and hit it after 6-8% of passes
CLEAR = 0.3          # m in front of the plane, still at GATE, before heading in
FRONT = 0.85         # m out on the axis by the commands, where it turns round to see the hoop
ARRIVED = 0.04       # m from a corner of the way round, by the commands, to head for the next
K_BOX = 1.5          # velocity per metre from the corner, before the speed cap: 2 braked too late for
                     # up to 0.7 s of lag and 4 steps of delay
RUN_SPEED = 0.3      # m/s, as the run in
ROUND_SPEED = 0.35   # m/s round the box
K_HOLD = 4.0         # sideways velocity per metre drifted off the line, blind on the way in

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
FACING = 3.0         # degrees off the axis. Run ins that started 3-6 degrees off hit the rim twice as
                     # often under demo noise: 7.2% of flights at 6, 3.8% at 3 (800). The run in now
                     # also holds the axis heading itself (Teacher.act)
STILL = 0.08         # m/s
TURNING = 2.0        # deg/s: still turning faster than this, it is not ready. Run ins that
                     # started still turning carried on 5-6 degrees (command delay and yaw lag)
                     # and drifted ~7 cm sideways in the blind last half metre, onto the rim:
                     # 13% of flights hit it, 0.5% with this (800 flights)
CENTRED = 0.15       # |u| to start the run in
GIVE_UP = 150        # steps lining up before it runs in anyway (a stage too close to see from)
RESET_BLIND = 5      # blind steps after which a sighting starts a fresh line-up

# The run in, down the line of sight: forward at RUN_SPEED and down along the pitch.
K_LINE = 2.0         # sideways velocity per metre off the hoop's axis on the run in
K_V = 1.0            # up command per unit of v

K_POS = 0.9          # velocity command per metre of position error
K_YAW = 1.5          # yaw command per radian of heading error
ROOM = 0.15          # m inside the fence (cage.margin in from the net) for the stage point. At 0.05 a
                     # 0.8 m/s drone overran it into the net in front on 17% of flights


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

def blind_action(x: np.ndarray, reached: int = 0) -> np.ndarray:
    """What the teacher does while the hoop is not reported, from the policy input x alone.
    reached: the leg of the way round (_through_and_round) it has already reached since
    the last sighting; the leg it is on now is in blind_action.leg afterwards."""
    blind_action.leg = reached
    frames, heights, actions, unseen = contract.split_input(x)
    speeds = contract.split_motion(x)[:, 0]
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
            and actions[max(k - 1, 0), FORWARD] >= RAN_AT_IT):
        return _through_and_round(last, actions, speeds, heights, k, after, reached)
    steps = np.concatenate([[after], WIDTHS[k + 1:]])
    backed = -_flown(actions, speeds, steps, k)[:, 0].sum()
    return _lost(last, after + WIDTHS[k + 1:].sum(), backed, height)


def _lost(last: np.ndarray, age: float, backed: float, height: float) -> np.ndarray:
    """Lost while lining up: turn or climb towards the edge it left by, holding position.
    Lost big and mid-frame, back straight off to BACK_TO first."""
    yaw = TURN_YAW * np.sign(last[U]) if abs(last[U]) > LOST_EDGE else 0.0
    # v is down. At a height limit the climb is cut to nothing, so it turns instead.
    up = _safe_up(-LOST_UP * np.sign(last[V]) if abs(last[V]) > LOST_EDGE else 0.0, height)
    if not yaw and not up:
        if DIAMETER / last[MAJOR] + backed < BACK_TO:
            return _action(forward=-BACK_SPEED / SPEED, up=_safe_up(0.0, height))
        if age > LOST_WAIT:
            yaw = TURN_YAW
    return _action(up=_safe_up(up, height), yaw=yaw)


def _through_and_round(last, actions, speeds, heights, k, after, reached=0) -> np.ndarray:
    """On through the hoop and round the box, on the commands and heights since the
    last sighting."""
    distance = DIAMETER / last[MAJOR]
    hoop_right = distance * last[U] * HALF_X
    hoop_below = distance * np.tan(PITCH + np.arctan(last[V] * HALF_Y))
    side = 1.0 if hoop_right <= 0 else -1.0   # the side of the axis it is already on
    # Where it has flown since the sighting, at the end of each slot: forward and right
    # of where it was then, and how far it climbed by the height readings.
    height = float(heights[-1])
    steps = np.concatenate([[after], WIDTHS[k + 1:]])
    path = np.cumsum(_flown(actions, speeds, steps, k), axis=0)
    along = path[:, 0] - distance        # + behind the hoop's plane
    beside = path[:, 1] - hoop_right     # + right of its centre
    # How far it climbed since, by its own up commands: blind, they are full-scale pulses
    # (_pulse), which move the drone max_up_speed a step whatever up_exponent is, and a
    # pulse moves it the same distance however long the drone takes to respond. The
    # height readings come in steps of up to 10 cm, and steering on them left passes
    # 6-7 cm off the centre (p90).
    climbed = float((actions[k:, UP] * steps).sum()) * _DRONE.max_up_speed * _DRONE.dt
    corners = [(BEHIND, 0.0), (BEHIND, side * GATE), (-CLEAR, side * GATE), (-FRONT, 0.0)]
    # Never back to an earlier leg: rebuilt every step from the slots, the estimate moves a
    # few cm as they regroup, and near a corner that flipped the leg each step and the drone
    # hung there, alternating its commands, until the sighting left the window.
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
    # The scan runs from leg 0, as the start of the path (in front, before the pass) would
    # satisfy the later legs' tests.
    leg = max(leg, reached)
    blind_action.leg = leg
    # Down to the hoop's centre at once, and through and round at its height, then up to
    # see it mid-frame from FRONT.
    up = _safe_up(_pulse(-hoop_below + (FRONT * SLOPE if leg >= 3 else 0.0) - climbed), height)
    if leg == 4:
        return _action(up=up, yaw=TURN_YAW)
    v = K_BOX * (np.array(corners[leg]) - [along[-1], beside[-1]])
    if leg == 0 and along[-1] < 0:
        # Not through yet: straight on from where the last sighting left it. The hoop was
        # half out of frame then, and the offset beside it read from that had no
        # correlation with the true one (r -0.02 over 2000 run ins), so steering on it only
        # added error. But it holds the line it was on: any drift beside it since, from
        # the commands actually sent, is flown back. With DAgger's 0.1 noise on the
        # commands, at 0.55-0.8 m/s, the drift hit the rim on 17% of flights.
        v[0] = min(v[0], RUN_SPEED)
        v[1] = -K_HOLD * path[-1, 1]
    else:
        speed = RUN_SPEED if leg == 0 else ROUND_SPEED
        v *= min(1.0, speed / max(np.linalg.norm(v), 1e-9))
    return _action(right=v[1] / SPEED, forward=v[0] / SPEED, up=up)


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
        self.leg = 0        # leg of the way round reached since the last sighting

    def act(self, env: HoopEnv, obs: np.ndarray) -> np.ndarray:
        x = np.array(obs, np.float32)
        x.flags.writeable = False
        frame = contract.split_input(x)[0][-1]
        if not frame[VISIBLE]:
            self.blind += 1
            action = blind_action(x, self.leg)
            self.leg = blind_action.leg
            return action
        if self.blind >= RESET_BLIND:
            self.phase, self.lining = "line_up", 0
        self.blind = 0
        self.leg = 0
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
        forward = RUN_SPEED / SPEED
        up = _up(-RUN_SPEED * self._slope(env)) - K_V * frame[V]
        # Held on the axis: the last ~0.58 m are blind and flown straight along the heading,
        # and 1.5 degrees off there put passes 1.5 cm to the side, which cost most of the
        # centring (r -0.69 between heading at the start and offset at the pass).
        # Held on the axis by the true offset, as the line-up was: the image's u is off by
        # the heading and the cut-off arc near the end, and strafing on it left passes 1.1 cm
        # to the side (median) against 0.5 without. Without any, under DAgger's noise the
        # run ins started off the axis stayed off it and hit the rim (17 of 255).
        beside = float((env.pos - env.hoop_centre) @ env._axis_h)   # + left of the axis, from the front
        right = K_LINE * beside / SPEED
        return _action(right=right, forward=forward, up=up, yaw=self._yaw_to(env, self._axis_heading(env)))

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
        speed = SPEED if speed is None else min(speed, SPEED)
        v = K_POS * (target - env.pos)
        flat = np.linalg.norm(v[:2])
        if flat > speed:
            v[:2] *= speed / flat
        c, s = np.cos(env.heading), np.sin(env.heading)
        return _action(right=(v[0] * s - v[1] * c) / SPEED, forward=(v[0] * c + v[1] * s) / SPEED,
                       up=_up(v[2]), yaw=self._yaw_to(env, heading))

    def _yaw_to(self, env, heading):
        """The turn towards a heading. Past the measured yaw deadband, so the turn is K_YAW
        times the error however small: K_YAW alone asks 0.08 at 3 degrees off, which does
        not turn the drone."""
        yaw = -K_YAW * _wrap(heading - env.heading)
        db = _DRONE.yaw_deadband
        return np.sign(yaw) * (db + (1.0 - db) * abs(yaw)) if abs(yaw) > 1e-3 else 0.0


class LearnedTeachers:
    """A privileged src.ppo actor (train.py --privileged-actor) teaching n simulators at
    once: one actor, one batched step for all of them, each with its own memory. It
    reads the true state after the policy input, env.privileged() or, when gated,
    env.gated(), so act on every step of every episode, as src.ppo's labels do."""

    def __init__(self, path, n: int):
        from src import actor
        self.actor, privileged = actor.load(path)
        if not privileged:
            raise ValueError(f"{path} is not a privileged actor (train.py --privileged-actor)")
        self.gated = privileged == "gated"
        self.h = self.actor.initial(n)

    def reset(self, k: int):
        """Simulator k starts a new episode: its memory clears."""
        self.h[k] = 0.0

    def act(self, obs: np.ndarray, priv: np.ndarray) -> np.ndarray:
        """Commands for every simulator from its policy input (n, ...) and true state (n, ...):
        env.gated()'s when self.gated, else env.privileged()'s."""
        import torch
        x = torch.as_tensor(np.concatenate([obs, priv], 1), dtype=torch.float32)
        with torch.no_grad():
            mean, self.h = self.actor.step(x, self.h)
        return mean.numpy().clip(-1.0, 1.0).astype(np.float32)


class LearnedTeacher:
    """LearnedTeachers for one simulator, with Teacher's reset/act."""

    def __init__(self, path):
        self.teachers = LearnedTeachers(path, 1)
        self.run_ins = 0   # Teacher counts its run ins; this one has none to count

    def reset(self, env: HoopEnv | None = None):
        self.teachers.reset(0)

    def act(self, env: HoopEnv, obs: np.ndarray) -> np.ndarray:
        view = env.gated() if self.teachers.gated else env.privileged()
        return self.teachers.act(np.asarray(obs, np.float32)[None], view[None])[0]


def run(episodes: int, level: float = 1.0, seed: int = 0, noise: float = 0.0,
        ranges: Randomise | None = None, teacher=None) -> collections.Counter:
    """How the teacher's flights end, over `episodes` minutes at `level`, on ranges
    (default the full training ranges)."""
    env, out = HoopEnv(level=level, randomise=True, ranges=ranges or Randomise()), collections.Counter()
    rng = np.random.default_rng(seed)
    teacher = teacher or Teacher(rng)
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
