"""Independent collector, frozen-calibration and resumable-learning checks."""
from copy import deepcopy
import hashlib

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src import unified_policy
from src.anchored_policy import AnchoredSAC
from src.retrace_policy import RetraceSAC, start_from_checkpoint


class CountingEnv(gym.Env):
    """Alternating true-terminal/timeout episodes expose autoreset mistakes."""
    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (100,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
        self.episode = 0
        self.last_action = None

    def observation(self):
        value = np.zeros(100, np.float32)
        value[0], value[1], value[99] = self.steps/10., self.episode/100., self.episode % 2
        return value

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.last_action = np.asarray(action).copy()
        self.steps += 1
        end = self.steps == 5
        return self.observation(), float(1.-.01*np.square(action).sum()), \
            end and self.episode % 2 == 1, end and self.episode % 2 == 0, {}


def small_model(device='cpu', **overrides):
    torch.set_num_threads(1)
    kwargs = dict(device=device, seed=14, buffer_size=64, batch_size=8,
                  learning_starts=0, ent_coef=.013, policy_kwargs={'net_arch': [16, 16]})
    kwargs.update(overrides)
    model = RetraceSAC('MlpPolicy', CountingEnv(), **kwargs)
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            equal(a, b)
    elif torch.is_tensor(first):
        assert torch.equal(first.detach().cpu(), second.detach().cpu())
    else:
        assert first == second


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_exact_behavior_density_retains_saturated_latent_samples(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = small_model(device)
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()
        model.actor.mu.bias.copy_(torch.tensor([12., -12., .3, -.3], device=device))
        model.actor.log_std.bias.fill_(-4.)
    model._last_obs = model.get_env().reset()
    torch.manual_seed(81)
    action, buffer_action = model._sample_action(0)
    observation, cached_action, gaussian, log_prob = deepcopy(model._pending_behavior)
    np.testing.assert_array_equal(action, buffer_action)
    assert action[0, :2].tolist() == [1., -1.]
    assert abs(gaussian[0, 0]) > 11. and abs(gaussian[0, 1]) > 11.
    with torch.no_grad():
        tensor_obs = torch.as_tensor(observation, device=device)
        mean, log_std, _ = model.actor.get_action_dist_params(tensor_obs)
        tensor_action = torch.as_tensor(action, device=device)
        tensor_gaussian = torch.as_tensor(gaussian, device=device)
        expected = torch.distributions.Normal(mean, log_std.exp()).log_prob(tensor_gaussian).sum(1)
        expected -= torch.log(1-tensor_action.square()+model.actor.action_dist.epsilon).sum(1)
        density = model.actor.action_dist.proba_distribution(mean, log_std)
        recovered = density.log_prob(tensor_action, tensor_gaussian)
        wrong_inverse = density.log_prob(tensor_action)
    np.testing.assert_array_equal(log_prob, expected.cpu().numpy())
    np.testing.assert_array_equal(log_prob, recovered.cpu().numpy())
    assert abs(float(wrong_inverse.item()-log_prob[0])) > 1000.
    next_obs, reward, done, infos = model.get_env().step(action)
    model._store_transition(model.replay_buffer, buffer_action, next_obs, reward, done, infos)
    assert model._pending_behavior is None
    np.testing.assert_array_equal(model.get_env().envs[0].unwrapped.last_action, cached_action[0])
    sample = model.replay_buffer._get_samples(np.array([0]))
    np.testing.assert_array_equal(sample.seq_behavior_gaussian_actions[0, 0].cpu().numpy(), gaussian[0])
    assert sample.seq_behavior_log_prob[0, 0, 0].item() == log_prob[0]


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_calibration_freezes_actor_adam_and_disk_resume_keeps_target_and_sequence_contract(tmp_path, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = small_model(device)
    model.refill_transitions, model.calibration_updates = 4, 8
    actor, reference, initial_critic = state(model.actor), state(model.reference_actor), state(model.critic)
    model.learn(11)
    assert model.critic_calibration_updates == 8 and model.actor_attempts == 0
    assert not model.actor.optimizer.state_dict()['state']
    equal(actor, state(model.actor)); equal(reference, state(model.reference_actor))
    assert any(not torch.equal(value, state(model.critic)[key]) for key, value in initial_critic.items())
    assert model.actor.observation_space.shape == model.critic.observation_space.shape == (100,)
    model.learn(5, reset_num_timesteps=False)
    assert model.actor_attempts == 5 and model.actor_updates > 0
    assert model.actor.optimizer.state_dict()['state']
    assert any(not torch.equal(value, state(model.actor)[key]) for key, value in actor.items())
    assert model.replay_buffer._stream_id == 1  # Consecutive learn without reset stays continuous.
    actor, optimizer = state(model.actor), deepcopy(model.actor.optimizer.state_dict())
    attempts = model.actor_attempts
    model._calibrate_critics(3, 8)
    equal(actor, state(model.actor)); equal(optimizer, model.actor.optimizer.state_dict())
    equal(reference, state(model.reference_actor))
    assert model.actor_attempts == attempts
    replay = model.replay_buffer
    assert replay.dones[:16].sum() == 3 and replay.timeouts[:16].sum() == 1
    # The terminal observation belongs to episode1; vector autoreset starts episode2.
    assert replay.next_observations[4, 0, 0] == .5
    assert replay.next_observations[4, 0, 1] == np.float32(.01)
    assert replay.observations[5, 0, 1] == np.float32(.02)
    assert replay._get_samples(np.array([4])).seq_valid.sum().item() == 1
    model.save(tmp_path/'model.zip'); model.save_replay_buffer(tmp_path/'replay.pkl')
    restored = RetraceSAC.load(tmp_path/'model.zip', env=CountingEnv(), device=device)
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    for name in ('actor', 'reference_actor', 'critic', 'critic_target'):
        equal(state(getattr(model, name)), state(getattr(restored, name)))
    equal(model.actor.optimizer.state_dict(), restored.actor.optimizer.state_dict())
    equal(model.critic.optimizer.state_dict(), restored.critic.optimizer.state_dict())
    for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts',
                 'behavior_log_probs', 'behavior_gaussian_actions', 'insertion_ids', 'episode_ids', 'stream_ids'):
        np.testing.assert_array_equal(getattr(replay, name), getattr(restored.replay_buffer, name))
    assert restored.replay_buffer._stream_id == replay._stream_id+1
    assert restored._pending_behavior is None and restored._last_obs is None
    for name in ('num_timesteps', 'actor_attempts', 'actor_updates', 'critic_calibration_updates',
                 'retrace_version', 'retrace_target_version', 'retrace_replay_version', 'behavior_density_version'):
        assert getattr(model, name) == getattr(restored, name)
    observation = replay.observations[0, 0]
    np.testing.assert_array_equal(model.predict(observation, deterministic=True)[0],
                                  restored.predict(observation, deterministic=True)[0])
    restored.learn(4, reset_num_timesteps=False)
    assert restored.num_timesteps == 20
    old_tail = restored.replay_buffer._get_samples(np.array([15]))
    assert old_tail.seq_valid.sum().item() == 1 and old_tail.dones.item() == 0.
    assert restored.replay_buffer.stream_ids[15] != restored.replay_buffer.stream_ids[16]
    assert all(not parameter.requires_grad for parameter in restored.reference_actor.parameters())


class StopBeforeStorage(BaseCallback):
    def _on_step(self):
        return self.n_calls < 3


def test_callback_abort_discards_pending_record_and_resumes_in_new_stream():
    model = small_model()
    model.learn(20, callback=StopBeforeStorage())
    replay = model.replay_buffer
    assert model.num_timesteps == 3 and replay.size() == 2
    assert model._pending_behavior is None and model._last_obs is None
    assert not replay.dones[:2].any()
    previous_stream = replay.stream_ids[1]
    model.learn(4, reset_num_timesteps=False)
    assert model.num_timesteps == 7 and replay.size() == 6
    assert replay.stream_ids[2] != previous_stream
    assert replay._get_samples(np.array([1])).seq_valid.sum().item() == 1
    assert replay.next_observations[1, 0, 0] == np.float32(.2)
    assert not replay.dones[1, 0]
    assert replay.observations[2, 0, 0] == 0.
    assert model._pending_behavior is None


def test_behavior_metadata_cannot_be_reused_or_attached_to_different_action():
    model = small_model(); model._last_obs = model.get_env().reset()
    action, buffer_action = model._sample_action(0)
    with pytest.raises(RuntimeError, match='consumed once'):
        model._sample_action(0)
    following, reward, done, info = model.get_env().step(action)
    wrong = buffer_action.copy(); wrong[0, 0] += .001
    with pytest.raises(ValueError, match='does not match'):
        model._store_transition(model.replay_buffer, wrong, following, reward, done, info)
    assert model.replay_buffer.size() == 0
    model._store_transition(model.replay_buffer, buffer_action, following, reward, done, info)
    with pytest.raises(RuntimeError, match='original behavior'):
        model._store_transition(model.replay_buffer, buffer_action, following, reward, done, info)
    assert model.replay_buffer.size() == 1


def test_actor_transfer_keeps_drifted_actor_original_reference_and_fresh_learning(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    source = AnchoredSAC('MlpPolicy', CountingEnv(), device='cpu', seed=4, learning_starts=0,
                         buffer_size=64, batch_size=8, ent_coef=.013,
                         policy_kwargs={'net_arch': [16, 16]})
    source.refill_transitions, source.calibration_updates = 4, 2
    source.learn(9)
    assert source.actor_updates > 0
    assert any(not torch.equal(value, state(source.reference_actor)[key])
               for key, value in state(source.actor).items())
    path = tmp_path/'source.zip'; source.save(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 4)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 2)
    transferred, metadata = start_from_checkpoint(path, CountingEnv(), device='cpu', seed=5,
                                                   buffer_size=64, tensorboard_log=tmp_path/'logs')
    equal(state(source.actor), state(transferred.actor))
    equal(state(source.reference_actor), state(transferred.reference_actor))
    assert not transferred.actor.optimizer.state_dict()['state']
    assert not transferred.critic.optimizer.state_dict()['state']
    assert transferred.num_timesteps == transferred.replay_buffer.size() == transferred.actor_attempts == 0
    assert transferred.observation_space.shape == transferred.critic.observation_space.shape == (100,)
    assert metadata['actor_receives_privileged_context'] is False
    assert metadata['trace_length'] == 8 and metadata['fresh_critics']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('override', [
    {'learning_starts': 1}, {'use_sde': True},
    {'action_noise': NormalActionNoise(np.zeros(4), np.ones(4)*.1)},
    {'replay_buffer_kwargs': {'sequence_length': 1}},
])
def test_unsupported_collector_and_target_contracts_fail_closed(override):
    with pytest.raises(ValueError):
        small_model(**override)


def test_vecnormalize_is_rejected_before_collection():
    env = VecNormalize(DummyVecEnv([CountingEnv]))
    with pytest.raises(ValueError, match='unnormalized'):
        RetraceSAC('MlpPolicy', env, learning_starts=0, ent_coef=.013, device='cpu')
    env.close()


def test_incompatible_saved_density_or_sequence_replay_is_rejected_transactionally(tmp_path):
    model = small_model()
    model.behavior_density_version = 'wrong-density'
    model.save(tmp_path/'wrong-model.zip')
    with pytest.raises(ValueError, match='complete saved'):
        RetraceSAC.load(tmp_path/'wrong-model.zip', device='cpu')
    bad = small_model(); bad.replay_buffer.behavior_density_version = 'wrong-density'
    bad.save_replay_buffer(tmp_path/'wrong-replay.pkl')
    target = small_model(); previous = target.replay_buffer
    with pytest.raises(ValueError, match='behavior-density'):
        target.load_replay_buffer(tmp_path/'wrong-replay.pkl')
    assert target.replay_buffer is previous
    bad.replay_buffer.behavior_density_version = previous.behavior_density_version
    bad.replay_buffer.behavior_gaussian_actions = np.zeros((64, 1, 3), np.float32)
    bad.save_replay_buffer(tmp_path/'wrong-shape.pkl')
    with pytest.raises(ValueError, match='behavior-density'):
        target.load_replay_buffer(tmp_path/'wrong-shape.pkl')
    assert target.replay_buffer is previous
