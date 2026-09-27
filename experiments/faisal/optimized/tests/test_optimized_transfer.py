"""Contract checks for transferring a learned actor to a changed RL objective."""
import hashlib

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from src.unified_policy import UnifiedSAC, start_from_checkpoint
from src import unified_policy


class SmallObservationEnv(gym.Env):
    def __init__(self, size):
        self.observation_space = gym.spaces.Box(-2., 2., (size,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
        self.size = size
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.np_random.uniform(-1, 1, self.size).astype(np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = self.np_random.uniform(-1, 1, self.size).astype(np.float32)
        reward = float(1.-np.square(action).sum())
        return obs, reward, self.steps == 12, False, {}


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    old = SAC('MlpPolicy', SmallObservationEnv(84), seed=11, device='cpu',
              learning_starts=0, batch_size=8, buffer_size=64,
              policy_kwargs={'net_arch': [16, 16]}, ent_coef='auto_0.013')
    old.learn(8)
    old.num_timesteps = 520000
    old.imitation_updates = 49999
    path = tmp_path/'source.zip'
    old.save(path)
    return old, path


def transfer(source, tmp_path, monkeypatch, *, size=100):
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 8)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 4)
    return start_from_checkpoint(source[1], SmallObservationEnv(size), device='cpu',
                                 seed=7, buffer_size=64, tensorboard_log=tmp_path/'logs')


def tensors(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def unchanged(before, module):
    return all(torch.equal(before[k], v) for k, v in module.state_dict().items())


def test_transfer_preserves_distribution_for_arbitrary_extra_features(source, tmp_path, monkeypatch):
    old, path = source
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    model, metadata = transfer(source, tmp_path, monkeypatch)
    rng = np.random.default_rng(31)
    prefix = rng.uniform(-1, 1, (32, 84)).astype(np.float32)
    augmented = np.concatenate([prefix, rng.uniform(-100, 100, (32, 16)).astype(np.float32)], axis=1)
    with torch.no_grad():
        old_mean, old_logstd, _ = old.actor.get_action_dist_params(torch.tensor(prefix))
        new_mean, new_logstd, _ = model.actor.get_action_dist_params(torch.tensor(augmented))
    torch.testing.assert_close(old_mean, new_mean, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(old_logstd, new_logstd, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(old.predict(prefix, deterministic=True)[0],
                               model.predict(augmented, deterministic=True)[0], rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(model.actor.latent_pi[0].weight[:, 84:]) == 0
    assert model.num_timesteps == model._n_updates == model.replay_buffer.size() == 0
    assert not model.actor.optimizer.state and not model.critic.optimizer.state
    assert model.log_ent_coef is None and model.ent_coef_optimizer is None
    assert model.ent_coef_tensor.item() == pytest.approx(old.log_ent_coef.exp().item())
    assert model.seed == 7
    assert not hasattr(model, 'imitation_updates') and not hasattr(model, 'demo_obs')
    assert metadata['fresh_critics'] and metadata['fresh_optimizers']
    assert metadata['steps'] == 520000 and metadata['observation_dim'] == 100
    assert metadata['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert unchanged(tensors(model.critic), model.critic_target)
    assert not torch.equal(model.critic.qf0[2].weight, old.critic.qf0[2].weight)


def test_refill_then_calibration_never_steps_actor_and_sac_then_learns(source, tmp_path, monkeypatch):
    model, _ = transfer(source, tmp_path, monkeypatch)
    actor, critic = tensors(model.actor), tensors(model.critic)
    model.learn(7)
    assert model._n_updates == 0 and model.transfer_phase == 'replay_refill'
    assert unchanged(actor, model.actor) and unchanged(critic, model.critic)
    model.learn(4, reset_num_timesteps=False)
    assert model.critic_calibration_updates == model._n_updates == 4
    assert model.actor_updates == 0 and model.transfer_phase == 'sac_training'
    assert unchanged(actor, model.actor) and not unchanged(critic, model.critic)
    assert not model.actor.optimizer.state and model.critic.optimizer.state
    model.learn(1, reset_num_timesteps=False)
    assert model.actor_updates == 1 and model._n_updates == 5
    assert not unchanged(actor, model.actor)
    assert model.actor.optimizer.param_groups[0]['lr'] == 3e-5
    assert model.critic.optimizer.param_groups[0]['lr'] == 3e-4


def test_resume_preserves_partial_calibration_and_replay(source, tmp_path, monkeypatch):
    model, _ = transfer(source, tmp_path, monkeypatch)
    model.learn(9)
    assert model.critic_calibration_updates == 2 and model.transfer_phase == 'critic_calibration'
    actor = tensors(model.actor)
    model.save(tmp_path/'resume.zip')
    model.save_replay_buffer(tmp_path/'replay.pkl')
    restored = UnifiedSAC.load(tmp_path/'resume.zip', env=SmallObservationEnv(100), device='cpu')
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    assert restored.refill_transitions == 8 and restored.calibration_updates == 4
    assert restored.actor_updates == 0 and restored.critic_calibration_updates == 2
    assert restored.replay_buffer.size() == 9
    assert restored.transfer_phase == 'critic_calibration'
    restored.learn(2, reset_num_timesteps=False)
    assert restored.critic_calibration_updates == 4 and restored.actor_updates == 0
    assert unchanged(actor, restored.actor)
    restored.learn(1, reset_num_timesteps=False)
    assert restored.actor_updates == 1 and not unchanged(actor, restored.actor)


def test_transfer_rejects_missing_original_observation_prefix(source, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='84 observations'):
        transfer(source, tmp_path, monkeypatch, size=83)
