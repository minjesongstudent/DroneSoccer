"""Current group policy inputs from real detections and actually issued actions."""
from collections import deque

import numpy as np

from src import contract


class CameraHistory:
    def __init__(self):
        self.frames = None

    def update(self, detection, issued_action):
        obs = np.asarray(detection, dtype=np.float32)
        action = np.asarray(issued_action, dtype=np.float32)
        if obs.shape != (contract.OBS_DIM,) or not np.isfinite(obs).all():
            raise ValueError('Eight finite detector features required')
        if action.shape != (contract.ACTION_DIM,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('Actual issued normalized action required')
        if self.frames is None:
            self.frames = deque([obs.copy() for _ in range(contract.FRAME_HISTORY)],
                                maxlen=contract.FRAME_HISTORY)
            self.actions = deque([np.zeros(contract.ACTION_DIM,np.float32)
                                  for _ in range(contract.ACTION_HISTORY)],
                                 maxlen=contract.ACTION_HISTORY)
            self.unseen = not bool(obs[contract.VISIBLE])
        else:
            self.frames.append(obs.copy())
            self.actions.append(action.copy())
            self.unseen = self.unseen and not bool(obs[contract.VISIBLE])
        return contract.policy_input(self.frames, self.actions, self.unseen)
