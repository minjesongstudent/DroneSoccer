"""Observation memory retains usable history without exposing hidden state."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from src import contract
from src.observation_memory import (BASE_OBS_DIM, MEMORY_DIM, MEMORY_LOW, MEMORY_HIGH,
                                    POLICY_INPUT_DIM, CameraCommandMemory, ObservationMemory)
from src.params import Drone


def observation(visible=True, u=.2):
    obs = np.zeros(BASE_OBS_DIM, dtype=np.float32)
    if visible:
        end = contract.OBS_STACK*contract.OBS_DIM
        obs[end-contract.OBS_DIM:end] = [u, -.1, .65, .8, -.1, .1, 1., 1.]
    return obs


class SensorOnlyEnv(gym.Env):
    def __init__(self, initial, subsequent, hidden_position):
        self.initial = initial.copy()
        self.subsequent = subsequent.copy()
        self.hidden_position = hidden_position
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (BASE_OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)

    @property
    def dyn(self):
        raise AssertionError('Memory must not read simulator dynamics')

    @property
    def drone(self):
        raise AssertionError('Nominal settings must not be silently read from the environment')

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.initial.copy(), {'drone_pos': self.hidden_position, 'lap_phase': 'approach'}

    def step(self, action):
        self.last_action = action.copy()
        return self.subsequent.copy(), float(self.hidden_position), False, False, {
            'drone_pos': self.hidden_position, 'lap_phase': 'return_side', 'route': 'left'}


def test_wrapper_preserves_prefix_actions_info_and_unwrapped():
    initial, current = observation(), observation(False)
    initial[:20] = np.linspace(-1., 1., 20)
    current[-12:] = np.arange(12)/12
    base = SensorOnlyEnv(initial, current, 123.)
    env = ObservationMemory(base)
    obs, info = env.reset(seed=1)
    assert obs.shape == (POLICY_INPUT_DIM,) == (BASE_OBS_DIM+MEMORY_DIM,)
    np.testing.assert_array_equal(obs[:BASE_OBS_DIM], initial)
    assert info['drone_pos'] == 123. and env.unwrapped is base
    obs, reward, terminated, truncated, info = env.step([2., .4, -.2, -3.])
    np.testing.assert_array_equal(obs[:BASE_OBS_DIM], current)
    np.testing.assert_array_equal(base.last_action, np.array([1., .4, -.2, -1.], np.float32))
    assert reward == 123. and not terminated and not truncated
    assert info['lap_phase'] == 'return_side'
    assert env.observation_space.contains(obs)


def test_same_sensor_history_and_commands_ignore_different_physical_state_reward_and_info():
    a = ObservationMemory(SensorOnlyEnv(observation(), observation(False), 1.))
    b = ObservationMemory(SensorOnlyEnv(observation(), observation(False), -1000.))
    np.testing.assert_array_equal(a.reset()[0], b.reset()[0])
    for action in np.random.default_rng(1).uniform(-2., 2., (300, 4)):
        np.testing.assert_array_equal(a.step(action)[0], b.step(action)[0])


def test_last_sighting_survives_beyond_old_window_and_distinguishes_blind_histories():
    a, b = CameraCommandMemory(), CameraCommandMemory()
    a.reset(observation(u=.4))
    b.reset(observation(u=-.4))
    blind = observation(False)
    for _ in range(100):
        oa = a.advance([0., .2, 0., 0.], blind)
        ob = b.advance([0., .2, 0., 0.], blind)
    np.testing.assert_array_equal(oa[:BASE_OBS_DIM], ob[:BASE_OBS_DIM])
    assert oa[BASE_OBS_DIM] == pytest.approx(.4)
    assert ob[BASE_OBS_DIM] == pytest.approx(-.4)
    assert oa[BASE_OBS_DIM+8] == 1.
    assert oa[BASE_OBS_DIM+9] == pytest.approx(10./60.)
    assert oa[BASE_OBS_DIM+10] > 0.


def test_command_history_is_retained_when_recent_observations_and_actions_match():
    a, b = CameraCommandMemory(), CameraCommandMemory()
    a.reset(observation())
    b.reset(observation())
    blind = observation(False)
    for _ in range(40):
        a.advance([0., .5, 0., 0.], blind)
        b.advance([.5, 0., 0., 0.], blind)
    for _ in range(30):
        oa = a.advance([0., 0., 0., 0.], blind)
        ob = b.advance([0., 0., 0., 0.], blind)
    assert oa[BASE_OBS_DIM+10] > .2
    assert ob[BASE_OBS_DIM+11] < -.2  # positive right is negative left
    assert not np.array_equal(oa[BASE_OBS_DIM:], ob[BASE_OBS_DIM:])


def test_reacquisition_reanchors_without_erasing_velocity_response():
    memory = CameraCommandMemory()
    memory.reset(observation())
    for _ in range(10):
        memory.advance([0., 1., 0., 0.], observation(False))
    seen = memory.advance([0., 1., 0., 0.], observation(u=-.3))
    np.testing.assert_array_equal(seen[BASE_OBS_DIM+9:BASE_OBS_DIM+14], np.zeros(5))
    assert seen[BASE_OBS_DIM+14] == 1.
    after = memory.advance([0., 0., 0., 0.], observation(False))
    assert after[BASE_OBS_DIM+10] > 0.  # still coasting after sighting
    assert after[BASE_OBS_DIM] == pytest.approx(-.3)


def test_reset_clears_response_pending_commands_and_last_sighting():
    d = Drone(delay_steps=2)
    memory = CameraCommandMemory(d)
    memory.reset(observation())
    for _ in range(20):
        memory.advance([.5, 1., .3, .8], observation(False))
    blind = memory.reset(observation(False))
    np.testing.assert_array_equal(blind[BASE_OBS_DIM:], np.zeros(MEMORY_DIM))
    memory.reset(observation())
    fresh = CameraCommandMemory(d)
    fresh.reset(observation())
    for _ in range(5):
        np.testing.assert_array_equal(memory.advance([0., 0., 0., 0.], observation(False)),
                                      fresh.advance([0., 0., 0., 0.], observation(False)))


def test_nominal_configuration_is_copied_and_pending_delay_is_respected():
    d = Drone(delay_steps=2)
    memory = CameraCommandMemory(d)
    d.delay_steps, d.max_speed = 0, 100.
    memory.reset(observation())
    for _ in range(2):
        out = memory.advance([0., 1., 0., 0.], observation(False))
        assert out[BASE_OBS_DIM+10] == 0.
    assert memory.advance([0., 1., 0., 0.], observation(False))[BASE_OBS_DIM+10] > 0.


def test_extra_features_remain_bounded_for_long_blind_sequences():
    memory = CameraCommandMemory()
    memory.reset(observation())
    for _ in range(2000):
        out = memory.advance([0., 2., 2., 0.], observation(False))
    extra = out[BASE_OBS_DIM:]
    assert extra.dtype == np.float32 and np.isfinite(extra).all()
    assert np.all(extra >= MEMORY_LOW) and np.all(extra <= MEMORY_HIGH)
    assert extra[9] == extra[10] == extra[12] == 1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA feature integration')
def test_cuda_can_use_finite_bounded_memory_features_with_gradients():
    memory = CameraCommandMemory()
    memory.reset(observation())
    batch = np.stack([memory.advance([.1, .3, .05, -.2], observation(False)) for _ in range(32)])
    features = torch.as_tensor(batch, device='cuda')
    layer = torch.nn.Linear(POLICY_INPUT_DIM, 4).to('cuda')
    loss = layer(features).square().mean()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(layer.weight.grad).all()
    assert torch.count_nonzero(layer.weight.grad[:, BASE_OBS_DIM:]).item() > 0
