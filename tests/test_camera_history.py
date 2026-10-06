import numpy as np
import pytest
from src.camera_history import CameraHistory
from src.sim import HoopEnv
from src import contract


def test_matches_simulator_public_history():
    env = HoopEnv(level=0)
    policy_obs, _ = env.reset(seed=9876)
    bridge = CameraHistory()
    assert np.array_equal(bridge.update(env.observe(), np.zeros(4)), policy_obs)
    rng = np.random.default_rng(9)
    for _ in range(90):
        action = rng.uniform(-.2,.2,4).astype(np.float32)
        policy_obs, _, end, cutoff, _ = env.step(action)
        assert np.array_equal(bridge.update(env.observe(), action), policy_obs)
        if end or cutoff:
            break


def test_seen_latch_and_unsent_suggestions():
    bridge = CameraHistory()
    blind = contract.NOT_VISIBLE.copy()
    assert bridge.update(blind,np.zeros(4))[-1] == 1
    seen = blind.copy(); seen[contract.VISIBLE] = 1
    assert bridge.update(seen,np.zeros(4))[-1] == 0
    for _ in range(65):
        obs = bridge.update(blind,np.zeros(4))
    assert obs.shape == (109,) and obs[-1] == 0
    assert np.array_equal(obs[-13:-1], np.zeros(12))


def test_rejects_invalid_observations_and_actions():
    with pytest.raises(ValueError):
        CameraHistory().update(np.zeros(119),np.zeros(4))
    with pytest.raises(ValueError):
        CameraHistory().update(np.zeros(8),np.ones(4)*2)
