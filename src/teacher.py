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

_CAMERA, _DRONE, _HOOP = Camera(), Drone(), Hoop()
PITCH = np.deg2rad(_CAMERA.pitch_deg)
SLOPE = np.tan(PITCH)
HALF_X, HALF_Y = _CAMERA.half_extent
DIAMETER = (_HOOP.outer_diameter + _HOOP.outer_height) / 2
WIDTHS = np.array(contract.SLOT_STEPS, float)
SPEED = float(np.mean(Randomise().max_speed))
UP_EXPONENT = float(np.mean(Randomise().up_exponent))
TELEMETRY_BIAS = 0.05
DELAY = int(round(np.mean(Randomise().delay_steps)))
TAU = float(np.mean(Randomise().tau))
SCALE_RANGE = (0.7, 1.4)
SCALE_MIN = 0.5


def _up(speed: float) -> float:
    c = speed / _DRONE.max_up_speed
    return float(np.sign(c) * min(abs(c), 1.0) ** (1.0 / UP_EXPONENT))


def _pulse(dz: float) -> float:
    return float(np.clip(dz / (_DRONE.max_up_speed * _DRONE.dt), -1.0, 1.0))


def _speed_scale(asked: np.ndarray, speeds: np.ndarray) -> float:
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
    asked = actions[:, [FORWARD, RIGHT]]
    asked = asked / np.maximum(1.0, np.linalg.norm(asked, axis=1))[:, None]
    return asked[k:] * (_speed_scale(asked, speeds) * SPEED * _DRONE.dt * steps)[:, None]


SEARCH_UP = 0.5
MAX_CLIMB = 1.75
MIN_HEIGHT = 0.6
TURN_YAW = 0.7
SWEEPING = 0.25

LOST_EDGE = 0.3
LOST_UP = 0.4
LOST_WAIT = 20
BACK_TO = 0.75
BACK_SPEED = 0.2

RAN_AT_IT = 0.4
BEHIND = 0.3
GATE = 0.52
CLEAR = 0.3
FRONT = 0.85
ARRIVED = 0.04
K_BOX = 1.5
RUN_SPEED = 0.3
ROUND_SPEED = 0.35
K_HOLD = 4.0

STAGES = (0.9, 0.8, 0.7)
LINE_UP_SPEED = 0.2
STAGE_TOL = 0.06
LINED_UP = 0.03
FACING = 3.0
STILL = 0.08
TURNING = 2.0
CENTRED = 0.15
GIVE_UP = 150
RESET_BLIND = 5

K_LINE = 2.0
K_V = 1.0

K_POS = 0.9
K_YAW = 1.5
ROOM = 0.15


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _action(right=0.0, forward=0.0, up=0.0, yaw=0.0) -> np.ndarray:
    return np.clip([right, forward, up, yaw], -1.0, 1.0).astype(np.float32)


def _safe_up(up: float, height: float) -> float:
    if height >= MAX_CLIMB:
        up = min(up, 0.0)
    if height <= MIN_HEIGHT:
        up = max(up, 0.0)
    return up



def blind_action(x: np.ndarray, reached: int = 0) -> np.ndarray:
    blind_action.leg = reached
    frames, heights, actions, unseen = contract.split_input(x)
    speeds = contract.split_motion(x)[:, 0]
    height = float(heights[-1])
    seen = np.flatnonzero(frames[:, VISIBLE])
    if not len(seen):
        turned = np.abs(actions[:, YAW] * WIDTHS).sum() * _DRONE.max_yaw_rate_deg * _DRONE.dt
        if not unseen and turned < 360.0:
            return _action(up=_safe_up(0.0, height), yaw=TURN_YAW)
        going = actions[-1, UP]
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
    after = (1.0 - last[VISIBLE]) * WIDTHS[k]
    if (last[MAJOR] >= contract.PASS_MAJOR and abs(last[U]) < contract.PASS_U
            and actions[max(k - 1, 0), FORWARD] >= RAN_AT_IT):
        return _through_and_round(last, actions, speeds, heights, k, after, reached)
    steps = np.concatenate([[after], WIDTHS[k + 1:]])
    backed = -_flown(actions, speeds, steps, k)[:, 0].sum()
    return _lost(last, after + WIDTHS[k + 1:].sum(), backed, height)


def _lost(last: np.ndarray, age: float, backed: float, height: float) -> np.ndarray:
    yaw = TURN_YAW * np.sign(last[U]) if abs(last[U]) > LOST_EDGE else 0.0
    up = _safe_up(-LOST_UP * np.sign(last[V]) if abs(last[V]) > LOST_EDGE else 0.0, height)
    if not yaw and not up:
        if DIAMETER / last[MAJOR] + backed < BACK_TO:
            return _action(forward=-BACK_SPEED / SPEED, up=_safe_up(0.0, height))
        if age > LOST_WAIT:
            yaw = TURN_YAW
    return _action(up=_safe_up(up, height), yaw=yaw)


def _through_and_round(last, actions, speeds, heights, k, after, reached=0) -> np.ndarray:
    distance = DIAMETER / last[MAJOR]
    hoop_right = distance * last[U] * HALF_X
    hoop_below = distance * np.tan(PITCH + np.arctan(last[V] * HALF_Y))
    side = 1.0 if hoop_right <= 0 else -1.0
    height = float(heights[-1])
    steps = np.concatenate([[after], WIDTHS[k + 1:]])
    path = np.cumsum(_flown(actions, speeds, steps, k), axis=0)
    along = path[:, 0] - distance
    beside = path[:, 1] - hoop_right
    climbed = float((actions[k:, UP] * steps).sum()) * _DRONE.max_up_speed * _DRONE.dt
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
    leg = max(leg, reached)
    blind_action.leg = leg
    up = _safe_up(_pulse(-hoop_below + (FRONT * SLOPE if leg >= 3 else 0.0) - climbed), height)
    if leg == 4:
        return _action(up=up, yaw=TURN_YAW)
    v = K_BOX * (np.array(corners[leg]) - [along[-1], beside[-1]])
    if leg == 0 and along[-1] < 0:
        v[0] = min(v[0], RUN_SPEED)
        v[1] = -K_HOLD * path[-1, 1]
    else:
        speed = RUN_SPEED if leg == 0 else ROUND_SPEED
        v *= min(1.0, speed / max(np.linalg.norm(v), 1e-9))
    return _action(right=v[1] / SPEED, forward=v[0] / SPEED, up=up)



class Teacher:

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def reset(self, env: HoopEnv | None = None):
        self.phase = "line_up"
        self.blind = 0
        self.lining = 0
        self.run_ins = 0
        self.leg = 0

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
                z = min(max(stage[2], self._sight_z(env, env.pos)), env.cage.height - env.cage.margin)
                target = np.array([*stage[:2], z])
                return self._towards(env, target, self._facing_hoop(env), LINE_UP_SPEED)
            self.phase = "run_in"
            self.run_ins += 1
        forward = RUN_SPEED / SPEED
        up = _up(-RUN_SPEED * self._slope(env)) - K_V * frame[V]
        beside = float((env.pos - env.hoop_centre) @ env._axis_h)
        right = K_LINE * beside / SPEED
        return _action(right=right, forward=forward, up=up, yaw=self._yaw_to(env, self._axis_heading(env)))


    def _slope(self, env):
        return np.tan(np.deg2rad(env.camera.pitch_deg))

    def _sight_z(self, env, pos):
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
        speed = SPEED if speed is None else min(speed, SPEED)
        v = K_POS * (target - env.pos)
        flat = np.linalg.norm(v[:2])
        if flat > speed:
            v[:2] *= speed / flat
        c, s = np.cos(env.heading), np.sin(env.heading)
        return _action(right=(v[0] * s - v[1] * c) / SPEED, forward=(v[0] * c + v[1] * s) / SPEED,
                       up=_up(v[2]), yaw=self._yaw_to(env, heading))

    def _yaw_to(self, env, heading):
        yaw = -K_YAW * _wrap(heading - env.heading)
        db = _DRONE.yaw_deadband
        return np.sign(yaw) * (db + (1.0 - db) * abs(yaw)) if abs(yaw) > 1e-3 else 0.0


class LearnedTeachers:

    def __init__(self, path, n: int):
        from src import actor
        self.actor, privileged = actor.load(path)
        if not privileged:
            raise ValueError(f"{path} is not a privileged actor (train.py --privileged-actor)")
        self.gated = privileged == "gated"
        self.h = self.actor.initial(n)

    def reset(self, k: int):
        self.h[k] = 0.0

    def act(self, obs: np.ndarray, priv: np.ndarray) -> np.ndarray:
        import torch
        x = torch.as_tensor(np.concatenate([obs, priv], 1), dtype=torch.float32)
        with torch.no_grad():
            mean, self.h = self.actor.step(x, self.h)
        return mean.numpy().clip(-1.0, 1.0).astype(np.float32)


class LearnedTeacher:

    def __init__(self, path):
        self.teachers = LearnedTeachers(path, 1)
        self.run_ins = 0

    def reset(self, env: HoopEnv | None = None):
        self.teachers.reset(0)

    def act(self, env: HoopEnv, obs: np.ndarray) -> np.ndarray:
        view = env.gated() if self.teachers.gated else env.privileged()
        return self.teachers.act(np.asarray(obs, np.float32)[None], view[None])[0]


def run(episodes: int, level: float = 1.0, seed: int = 0, noise: float = 0.0,
        ranges: Randomise | None = None, teacher=None) -> collections.Counter:
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
