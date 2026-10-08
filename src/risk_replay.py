"""FIT-only collision-approach sampling; original rewards and transitions unchanged."""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


def collision_approaches(dones, timeouts, window=16):
    done = np.asarray(dones).reshape(-1).astype(bool)
    timeout = np.asarray(timeouts).reshape(-1).astype(bool)
    indices = []
    start = 0
    for end in np.flatnonzero(done):
        if not timeout[end]:
            indices.extend(range(max(start, end-window+1), end+1))
        start = end+1
    return np.asarray(indices, dtype=np.int64)


class RiskReplayBuffer(ReplayBuffer):
    """Half uniform, half fixed FIT collision approaches; never use holdout rows.

    This bounded experiment never fills the buffer, so saved FIT indices cannot
    be overwritten. Fail closed if used beyond that budget.
    """
    def sample(self, batch_size, env=None):
        if self.full:
            raise RuntimeError('Risk replay requires preserved, non-overwritten FIT indices')
        risk = getattr(self, 'risk_indices', np.array([], dtype=np.int64))
        if not len(risk):
            return super().sample(batch_size, env)
        count = batch_size//2
        indices = np.concatenate([np.random.randint(0, self.pos, batch_size-count),
                                  np.random.choice(risk, count, replace=True)])
        np.random.shuffle(indices)
        return self._get_samples(indices, env=env)
