"""Hand-written controllers on the same eight numbers the policy sees.

They prove the mission is solvable and are the comparison line for the RL results.
    act     a proportional controller that flies at the hoop and through it.
    Lapper  act for each pass, plus a scripted search and go-round.
Run: python -m src.evaluate --baseline
"""
from __future__ import annotations

import numpy as np

from src import contract

K_CENTRE = 1.5     # command per unit of u or v
COMMIT = 0.5       # forward command above which a lost hoop means "keep going"

# Lapper. After a pass the drone is only ~0.3 m past the hoop, too close to see
# it, so the way round is a fixed box, flown blind on timing: sidestep, reverse
# past the hoop's plane, sidestep back, turn to face it. Right is a guess; the
# baseline cannot know which side has room, so some laps end at the geofence.
SEARCH_YAW = 0.5       # turning speed while the hoop is out of view
GO_ROUND = [           # (steps, action): ~0.6 m right, ~0.8 m back, ~0.6 m left, ~180 degrees
    (15, (0.0, 1.0, 0.0, 0.0)),    # finish carrying the body through
    (18, (1.0, 0.0, 0.0, 0.0)),
    (22, (0.0, -1.0, 0.0, 0.0)),
    (18, (-1.0, 0.0, 0.0, 0.0)),
    (35, (0.0, 0.0, 0.0, 1.0)),
]


def _split(policy_input: np.ndarray):
    n_obs = contract.OBS_STACK * contract.OBS_DIM
    frames = policy_input[:n_obs].reshape(contract.OBS_STACK, contract.OBS_DIM)
    actions = policy_input[n_obs:].reshape(contract.ACTION_HISTORY, contract.ACTION_DIM)
    return frames, actions


def act(policy_input: np.ndarray) -> np.ndarray:
    frames, actions = _split(policy_input)
    u, v, _major, _ratio, _s, _c, visible, _marker = frames[-1]
    last_forward = actions[-1][1]

    if not visible:
        # The hoop leaves the frame in the last ~40 cm. Keep going if we were committed.
        forward = last_forward if last_forward > COMMIT else 0.0
        return np.array([0.0, forward, 0.0, 0.0], dtype=np.float32)

    right = np.clip(K_CENTRE * u, -1, 1)
    up = np.clip(-K_CENTRE * v, -1, 1)   # v is positive downwards in the image
    forward = np.clip(1.0 - 2.0 * max(abs(u), abs(v)), 0.0, 1.0)
    return np.array([right, forward, up, 0.0], dtype=np.float32)


class Lapper:
    """Search, pass with `act`, fly the GO_ROUND box, repeat.

    Needs memory across a lap, so it is a small state machine rather than a function.
    Call reset() at the start of every episode.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.script = []   # remaining (steps, action) of the go-round, empty when approaching

    def __call__(self, policy_input: np.ndarray) -> np.ndarray:
        frames, actions = _split(policy_input)
        visible = frames[-1][6]

        if not self.script and not visible and actions[-1][1] > COMMIT:
            self.script = [list(leg) for leg in GO_ROUND]   # committed and blind: the pass
        if self.script:
            leg = self.script[0]
            leg[0] -= 1
            if leg[0] <= 0:
                self.script.pop(0)
            return np.array(leg[1], dtype=np.float32)
        if not visible:
            return np.array([0.0, 0.0, 0.0, SEARCH_YAW], dtype=np.float32)
        return act(policy_input)
