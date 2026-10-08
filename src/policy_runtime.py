"""Named, hash-bound learned policies and their documented camera histories.

Never infer lag spacing from an input dimension or change the simulator contract.
Both supported models use the same eight camera features, three issued actions,
and the seen-hoop flag; only the documented observation lags differ.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.camera_history import CameraHistory


@dataclass(frozen=True)
class PolicySpec:
    name: str
    sha256: str
    lags: tuple[int, ...]

    @property
    def inputs(self):
        return len(self.lags)*8+13


SPECS = {
    'lags60-scratch': PolicySpec('lags60-scratch',
        '832e75fcf52998aa2b4b5dd1fa67ec0573d389816db5c5875e85099c5391ffe7',
        (60, 50, 40, 32, 24, 16, 12, 8, 4, 2, 1, 0)),
    'seenflag-s1-level1': PolicySpec('seenflag-s1-level1',
        '8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef',
        (24, 20, 16, 12, 8, 4, 2, 1, 0)),
}


class LearnedPolicy:
    def __init__(self, path: Path, spec: PolicySpec):
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != spec.sha256:
            raise ValueError('Model checksum does not match the selected profile')
        self.model = SAC.load(str(path), device='cpu')
        if (self.model.observation_space.shape != (spec.inputs,)
                or self.model.action_space.shape != (4,)):
            raise ValueError('Model spaces do not match the selected profile')
        self.spec = spec
        self.reset()

    def reset(self):
        self.history = CameraHistory(self.spec.lags)

    def predict(self, detection, previous_issued_action):
        obs = self.history.update(detection, previous_issued_action)
        action, _ = self.model.predict(obs, deterministic=True)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('Invalid learned action')
        return obs, action
