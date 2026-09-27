import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from src.bootstrap import GuidedSAC, bootstrap, restore_demonstrations
from src.curriculum import LessonEnv


def test_demonstrations_warm_start_actor_without_counting_online_steps(tmp_path):
    torch.set_num_threads(1)
    env = LessonEnv()
    obs, _ = env.reset(seed=1_000_001)
    transitions = []
    for _ in range(150):
        action = np.array([0., .5, 0., 0.], np.float32)
        following, reward, term, trunc, info = env.step(action)
        transitions.append(dict(obs=obs, action=action, next_obs=following,
                                reward=reward, terminated=term, truncated=trunc, info=info))
        obs = following
        if term or trunc:
            break
    assert info['outcome'] == 'passed'
    model = GuidedSAC('MlpPolicy', Monitor(LessonEnv()), device='cpu', seed=5,
                      buffer_size=1000, learning_starts=0, ent_coef='auto_0.01')
    report = bootstrap(model, transitions, tmp_path/'demonstrations.npz', updates=100)
    assert report['action_mse_after'] < report['action_mse_before'] * .1
    assert model.num_timesteps == 0 and model._n_updates == 0
    assert model.replay_buffer.size() == len(transitions)
    assert model.replay_buffer.dones[len(transitions)-1, 0] == 1
    assert not model.replay_buffer.timeouts[:len(transitions)].any()
    model.learn(2)
    assert model.num_timesteps == 2 and model._n_updates == 2
    assert np.isfinite(model.predict(obs, deterministic=True)[0]).all()
    expected = model.predict(obs, deterministic=True)[0]
    model.save(tmp_path/'model.zip')
    restored = GuidedSAC.load(tmp_path/'model.zip', device='cpu')
    assert not hasattr(restored, 'demo_obs')
    restore_demonstrations(restored, tmp_path/'demonstrations.npz')
    np.testing.assert_allclose(restored.predict(obs, deterministic=True)[0], expected)
    np.testing.assert_array_equal(restored.demo_actions.numpy(), model.demo_actions.numpy())


def test_bootstrap_rejects_missing_verified_data(tmp_path):
    import pytest
    with pytest.raises(ValueError, match='verified'):
        bootstrap(None, [], tmp_path/'empty.npz')
