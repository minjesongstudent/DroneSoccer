from __future__ import annotations

import numpy as np

from src import contract
from src.params import Camera, Drone

K_CENTRE = 1.5
COMMIT = 0.5
V_LEVEL = -np.tan(np.deg2rad(Camera().pitch_deg)) / Camera().half_extent[1]

ROUND = 0.95
K_ALIGN = 0.8
K_YAW = 1.0
FLATTER = 0.02
NOW_LAGS = (0, 1, 2)
THEN_LAGS = (4, 8)
MARKER_LAG = 60
HOLD_STEPS = 10

SEARCH_YAW = 0.5
TURN_STEPS = round(360 / (SEARCH_YAW * Drone().max_yaw_rate_deg) / Drone().dt)
CLIMB_STEPS = 10
MAX_CLIMBS = 2
GO_ROUND = [
    (15, (0.0, 1.0, 0.0, 0.0)),
    (18, (1.0, 0.0, 0.0, 0.0)),
    (22, (0.0, -1.0, 0.0, 0.0)),
    (18, (-1.0, 0.0, 0.0, 0.0)),
    (35, (0.0, 0.0, 0.0, 1.0)),
]


def _split(policy_input: np.ndarray):
    frames, _heights, actions, _unseen = contract.split_input(policy_input)
    return frames, actions


def act(policy_input: np.ndarray) -> np.ndarray:
    frames, actions = _split(policy_input)
    u, v, _major, _ratio, _s, _c, visible, _marker = frames[-1]
    last_forward = actions[-1][1]

    if not visible:
        forward = last_forward if last_forward > COMMIT else 0.0
        return np.array([0.0, forward, 0.0, 0.0], dtype=np.float32)

    right = np.clip(K_CENTRE * u, -1, 1)
    up = np.clip(-K_CENTRE * (v - V_LEVEL), -1, 1)
    forward = np.clip(1.0 - 2.0 * max(abs(u), abs(v - V_LEVEL)), 0.0, 1.0)
    return np.array([right, forward, up, 0.0], dtype=np.float32)


class Lapper:

    def __init__(self):
        self.reset()

    def reset(self):
        self.script = []
        self.strafe = 1.0
        self.since_flip = HOLD_STEPS
        self.searching = 0

    def __call__(self, policy_input: np.ndarray) -> np.ndarray:
        frames, actions = _split(policy_input)
        visible = frames[-1][6]

        if not self.script and not visible and actions[-1][1] > COMMIT:
            self.script = [list(leg) for leg in GO_ROUND]
        if self.script:
            leg = self.script[0]
            leg[0] -= 1
            if leg[0] <= 0:
                self.script.pop(0)
            return np.array(leg[1], dtype=np.float32)
        if not visible:
            return self._search()
        self.searching = 0
        action = self._line_up(frames, actions, act(policy_input))
        if not frames[contract.slot_of(MARKER_LAG):, 7].any():
            action[1] = 0.0
        return action

    def _search(self) -> np.ndarray:
        turns, into = divmod(self.searching, TURN_STEPS + CLIMB_STEPS)
        self.searching += 1
        climbing = 1 <= turns + 1 <= MAX_CLIMBS and into >= TURN_STEPS
        return np.array([0.0, 0.0, 1.0 if climbing else 0.0, SEARCH_YAW], dtype=np.float32)

    def _line_up(self, frames, actions, action):
        u, ratio = frames[-1][0], frames[-1][3]
        now, then = ([frames[contract.slot_of(lag)] for lag in lags] for lags in (NOW_LAGS, THEN_LAGS))
        self.since_flip += 1
        if (all(f[6] for f in now + then) and self.since_flip >= HOLD_STEPS
                and np.mean([f[3] for f in now]) < np.mean([f[3] for f in then]) - FLATTER):
            self.strafe, self.since_flip = -self.strafe, 0
        off = np.clip((ROUND - ratio) / (1 - ROUND), 0.0, 1.0)
        action[0] = np.clip(self.strafe * K_ALIGN * off + 0.3 * K_CENTRE * u, -1, 1)
        action[1] *= 1 - off
        action[3] = np.clip(K_YAW * u, -1, 1)
        return action
