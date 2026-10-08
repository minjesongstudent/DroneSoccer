"""Current group policy inputs from real detections and actually issued actions."""
from collections import deque

import numpy as np

from src import contract


class CameraHistory:
    def __init__(self, lags=contract.OBS_LAGS, action_history=contract.ACTION_HISTORY):
        if type(action_history) is not int or action_history not in (3,4):
            raise ValueError('Only explicit three or four issued-action histories supported')
        self.action_history = action_history
        self.lags = tuple(lags)
        if (not self.lags or self.lags[-1] != 0
                or any(type(v) is not int or v < 0 for v in self.lags)
                or any(a <= b for a, b in zip(self.lags, self.lags[1:]))):
            raise ValueError('Explicit decreasing integer lags ending in zero required')
        self.frames = None

    def update(self, detection, issued_action):
        obs = np.asarray(detection, dtype=np.float32)
        action = np.asarray(issued_action, dtype=np.float32)
        if obs.shape != (contract.OBS_DIM,) or not np.isfinite(obs).all():
            raise ValueError('Eight finite detector features required')
        if action.shape != (contract.ACTION_DIM,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('Actual issued normalized action required')
        if self.frames is None:
            self.frames = deque([obs.copy() for _ in range(max(self.lags)+1)],
                                maxlen=max(self.lags)+1)
            self.actions = deque([np.zeros(contract.ACTION_DIM,np.float32)
                                  for _ in range(self.action_history)],
                                 maxlen=self.action_history)
            self.unseen = not bool(obs[contract.VISIBLE])
        else:
            self.frames.append(obs.copy())
            self.actions.append(action.copy())
            self.unseen = self.unseen and not bool(obs[contract.VISIBLE])
        return np.concatenate([*(self.frames[-1-lag] for lag in self.lags),
                               *self.actions, [float(self.unseen)]]).astype(np.float32)
