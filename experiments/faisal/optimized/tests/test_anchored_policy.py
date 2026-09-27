"""Regression checks for anchored actor updates and balanced online replay."""
from copy import deepcopy
import hashlib

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

from src import unified_policy
from src.anchored_policy import (AnchoredSAC, TaskBalancedReplayBuffer,
                                 gaussian_kl, protected_observations,
                                 start_from_checkpoint)


class TinyEnv(gym.Env):
    def __init__(self, size=100):
        self.observation_space = gym.spaces.Box(-10., 10., (size,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
        self.size, self.episode = size, 0

    def obs(self):
        observation = np.zeros(self.size, np.float32)
        observation[:6] = self.np_random.uniform(-1, 1, 6)
        if self.size == 100:
            observation[-1] = self.episode % 2
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return self.obs(), {}

    def step(self, action):
        self.steps += 1
        return self.obs(), float(1.-np.square(action).sum()), self.steps >= 3, False, {}


@pytest.fixture
def model():
    torch.set_num_threads(1)
    return AnchoredSAC('MlpPolicy', TinyEnv(), seed=7, device='cpu',
                       buffer_size=64, batch_size=8, learning_starts=0,
                       ent_coef=.013, policy_kwargs={'net_arch': [16, 16]})


def tensors(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def equal(before, module):
    return all(torch.equal(before[key], value) for key, value in module.state_dict().items())


def assert_nested_equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    elif torch.is_tensor(first):
        assert torch.equal(first, second)
    else:
        assert first == second


def test_mask_uses_only_public_task_camera_and_recent_memory():
    obs = torch.zeros((8, 100))
    obs[0, -1] = 1  # Single-pass clearance remains protected while blind.
    obs[1, 70:72] = 1  # Front currently detected.
    obs[2:6, 90:93] = 1  # Last detection was the front, and has_seen is true.
    obs[2, 93] = 1.8/60
    obs[3, 93] = 1.81/60
    obs[4, 92] = 0  # No actual detection history.
    obs[5, 91] = 0  # Back of the ring.
    obs[6, 70] = 1  # Current ring without the front marker.
    assert protected_observations(obs).tolist() == [True, True, True, False, False, False, False, False]
    mu = torch.zeros((2, 4))
    assert torch.equal(gaussian_kl(mu, mu, mu, mu), torch.zeros(2))
    assert gaussian_kl(torch.ones((2, 4)), mu, mu, mu).tolist() == [2., 2.]


class ActionValue(torch.nn.Module):
    def forward(self, observations, actions):
        q = 1e6*actions.sum(dim=1, keepdim=True)
        return q, q


def test_huge_value_gradient_rejects_and_restores_weights_and_adam_state(model):
    model.critic = ActionValue()
    obs = torch.zeros((16, 100))
    # Establish real Adam moments before testing transactional rollback.
    for group in model.actor.optimizer.param_groups:
        group['lr'] = 1e-4
    assert model._actor_step(obs, model.ent_coef_tensor)['accepted']
    obs[:, -1] = 1
    for group in model.actor.optimizer.param_groups:
        group['lr'] = 1.
    actor_before = tensors(model.actor)
    optimizer_before = deepcopy(model.actor.optimizer.state_dict())
    reference_before = tensors(model.reference_actor)
    result = model._actor_step(obs, model.ent_coef_tensor)
    assert not result['accepted'] and model.anchor_rejected == 1
    assert equal(actor_before, model.actor) and equal(reference_before, model.reference_actor)
    assert_nested_equal(optimizer_before, model.actor.optimizer.state_dict())
    assert model.actor_updates == 1 and model.actor_attempts == 2
    assert result['kl'] <= model.anchor_max_kl


def test_unprotected_lap_states_are_free_to_learn(model):
    model.critic = ActionValue()
    obs = torch.zeros((16, 100))
    actor_before, reference_before = tensors(model.actor), tensors(model.reference_actor)
    for group in model.actor.optimizer.param_groups:
        group['lr'] = .05
    result = model._actor_step(obs, model.ent_coef_tensor)
    assert result['accepted'] and result['protected_fraction'] == 0 and result['kl'] == 0
    assert not equal(actor_before, model.actor) and equal(reference_before, model.reference_actor)


def test_accepted_protected_updates_obey_batch_kl_budget(model):
    model.critic = ActionValue()
    obs = torch.zeros((16, 100))
    obs[:, -1] = 1
    for group in model.actor.optimizer.param_groups:
        group['lr'] = .005
    results = [model._actor_step(obs, model.ent_coef_tensor) for _ in range(40)]
    assert any(r['accepted'] for r in results)
    assert any(not r['accepted'] for r in results)
    assert all(r['kl'] <= model.anchor_max_kl for r in results)


def add_transition(buffer, index, single, *, timeout=False):
    obs = np.zeros((1, 100), np.float32)
    obs[0, 0], obs[0, -1] = index, float(single)
    next_obs = obs.copy()
    next_obs[0, 0] += .25
    buffer.add(obs, next_obs, np.full((1, 4), index, np.float32),
               np.array([index], np.float32), np.array([index % 2], np.float32),
               [{'TimeLimit.truncated': timeout}])


def test_replay_balance_capacity_overwrite_and_terminal_semantics():
    env = TinyEnv()
    buffer = TaskBalancedReplayBuffer(4, env.observation_space, env.action_space, device='cpu')
    for index in range(6):
        add_transition(buffer, index, index % 2 == 1, timeout=index == 3)
    assert buffer.full and buffer.pos == 2 and buffer.size() == 4
    sample = buffer.sample(200)
    ids = sample.observations[:, 0].numpy().astype(int)
    assert set(ids) == {2, 3, 4, 5}  # Overwritten slots 0 and 1 never return.
    assert sample.observations[:, -1].sum().item() == 100
    np.testing.assert_array_equal(sample.rewards[:, 0].numpy(), ids)
    np.testing.assert_allclose(sample.next_observations[:, 0].numpy(), ids+.25)
    np.testing.assert_array_equal(sample.dones[:, 0].numpy(), np.where(ids == 3, 0, ids % 2))
    # Overwrite every slot with the other task; stale labels must not survive.
    for index in range(6, 10):
        add_transition(buffer, index, False)
    assert buffer.sample(50).observations[:, -1].sum().item() == 0
    buffer.reset()
    add_transition(buffer, 10, True)
    assert buffer.sample(50).observations[:, -1].sum().item() == 50


def test_resume_preserves_frozen_reference_balanced_replay_and_counters(model, tmp_path):
    model.refill_transitions, model.calibration_updates = 4, 2
    reference = tensors(model.reference_actor)
    model.learn(12)
    assert model.critic_calibration_updates == 2 and model.actor_attempts == 7
    assert model.actor_updates + model.anchor_rejected == model.actor_attempts
    assert equal(reference, model.reference_actor)
    assert not equal(reference, model.actor)  # A fresh copied actor would be wrong on resume.
    model.save(tmp_path/'model.zip')
    model.save_replay_buffer(tmp_path/'replay.pkl')
    restored = AnchoredSAC.load(tmp_path/'model.zip', env=TinyEnv(), device='cpu')
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    assert isinstance(restored.replay_buffer, TaskBalancedReplayBuffer)
    assert equal(reference, restored.reference_actor)
    assert all(not p.requires_grad for p in restored.reference_actor.parameters())
    assert restored.actor_attempts == model.actor_attempts
    assert restored.anchor_rejected == model.anchor_rejected
    assert restored.actor_updates == model.actor_updates
    assert restored.anchor_max_kl == .10 and restored.anchor_coefficient == 10
    np.testing.assert_array_equal(restored.replay_buffer._single_pass_slots,
                                  model.replay_buffer._single_pass_slots)
    obs, _ = TinyEnv().reset(seed=9)
    np.testing.assert_array_equal(restored.predict(obs, deterministic=True)[0],
                                  model.predict(obs, deterministic=True)[0])
    restored.learn(1, reset_num_timesteps=False)
    assert restored.actor_attempts == model.actor_attempts+1
    assert equal(reference, restored.reference_actor)


def test_fresh_transfer_freezes_the_inherited_actor_after_copying(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    old = SAC('MlpPolicy', TinyEnv(84), device='cpu', seed=5, buffer_size=64,
              policy_kwargs={'net_arch': [16, 16]}, ent_coef=.017)
    path = tmp_path/'source.zip'
    old.save(path)
    with pytest.raises(ValueError, match='saved reference_actor'):
        AnchoredSAC.load(path, device='cpu')
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 4)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 2)
    model, metadata = start_from_checkpoint(path, TinyEnv(), device='cpu', seed=7,
                                            buffer_size=64, tensorboard_log=tmp_path/'logs')
    assert isinstance(model, AnchoredSAC) and isinstance(model.replay_buffer, TaskBalancedReplayBuffer)
    assert equal(tensors(model.actor), model.reference_actor)
    assert model.ent_coef_tensor.item() == pytest.approx(.017)
    assert metadata['policy_update_rule'] == 'prior-actor-kl-and-task-balanced-replay-v1'
    obs = np.zeros((8, 100), np.float32)
    obs[:, 84:] = 1
    np.testing.assert_allclose(old.predict(obs[:, :84], deterministic=True)[0],
                               model.predict(obs, deterministic=True)[0], atol=1e-7)
    model.save(tmp_path/'fresh.zip')
    restored = AnchoredSAC.load(tmp_path/'fresh.zip', env=TinyEnv(), device='cpu')
    assert equal(tensors(model.reference_actor), restored.reference_actor)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA transfer integration check'))])
def test_revised_reward_transfer_keeps_latest_actor_and_original_reference_separately(
        model, tmp_path, monkeypatch, device):
    model.refill_transitions, model.calibration_updates = 4, 2
    model.learn(12)
    assert not equal(tensors(model.reference_actor), model.actor)
    source = tmp_path/'learned-anchored.zip'
    model.save(source)
    before_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 8)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 4)
    transferred, metadata = start_from_checkpoint(source, TinyEnv(), device=device, seed=19,
                                                  buffer_size=64, tensorboard_log=tmp_path/'new-logs')
    assert metadata['source_has_frozen_reference']
    assert metadata['source_observation_dim'] == metadata['observation_dim'] == 100
    assert metadata['reference'] == 'original_frozen_reference_from_source_checkpoint'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_hash == metadata['source_sha256']
    for old_module, new_module in ((model.actor, transferred.actor),
                                    (model.reference_actor, transferred.reference_actor)):
        for key, value in old_module.state_dict().items():
            assert torch.equal(value.cpu(), new_module.state_dict()[key].cpu())
    assert transferred.num_timesteps == transferred._n_updates == transferred.replay_buffer.size() == 0
    assert transferred.actor_updates == transferred.actor_attempts == transferred.anchor_rejected == 0
    assert transferred.critic_calibration_updates == 0
    assert not transferred.actor.optimizer.state and not transferred.critic.optimizer.state
    assert not all(torch.equal(value.cpu(), transferred.critic.state_dict()[key].cpu())
                   for key, value in model.critic.state_dict().items())
    assert all(torch.equal(value, transferred.critic_target.state_dict()[key])
               for key, value in transferred.critic.state_dict().items())
    rng = np.random.default_rng(42)
    obs = rng.uniform(-1, 1, (16, 100)).astype(np.float32)
    np.testing.assert_allclose(model.predict(obs, deterministic=True)[0],
                               transferred.predict(obs, deterministic=True)[0], rtol=1e-6, atol=1e-7)
    transferred.save(tmp_path/'transferred.zip')
    restored = AnchoredSAC.load(tmp_path/'transferred.zip', env=TinyEnv(), device=device)
    for key, value in model.reference_actor.state_dict().items():
        assert torch.equal(value.cpu(), restored.reference_actor.state_dict()[key].cpu())
    assert all(not p.requires_grad for p in restored.reference_actor.parameters())
