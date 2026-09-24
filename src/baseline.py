"""Proportional controller on the same seven numbers the policy sees.

Proves the task is solvable and is the comparison line for the RL results.
Run: python -m src.evaluate --baseline
"""
from __future__ import annotations

import numpy as np

from src import contract

K_CENTRE = 1.5     # command per unit of u or v
COMMIT = 0.5       # forward command above which a lost hoop means "keep going"


def act(policy_input: np.ndarray) -> np.ndarray:
    n_obs = contract.OBS_STACK * contract.OBS_DIM
    frames = policy_input[:n_obs].reshape(contract.OBS_STACK, contract.OBS_DIM)
    actions = policy_input[n_obs:].reshape(contract.ACTION_HISTORY, contract.ACTION_DIM)
    u, v, _major, _ratio, _s, _c, visible = frames[-1]
    last_forward = actions[-1][1]

    if not visible:
        # The hoop leaves the frame in the last ~40 cm. Keep going if we were committed.
        forward = last_forward if last_forward > COMMIT else 0.0
        return np.array([0.0, forward, 0.0, 0.0], dtype=np.float32)

    right = np.clip(K_CENTRE * u, -1, 1)
    up = np.clip(-K_CENTRE * v, -1, 1)   # v is positive downwards in the image
    forward = np.clip(1.0 - 2.0 * max(abs(u), abs(v)), 0.0, 1.0)
    return np.array([right, forward, up, 0.0], dtype=np.float32)
