"""Reward fixtures are accounting unit tests, not policy-flight results."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from src import contract
from src.clearance_training import ClearanceTrainingEnv, PROFILE, passage_reward_delta
from src.policy_runtime import LearnedPolicy
from src.sim import HoopEnv


class RewardFixture(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (4,), np.float32)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (109,), np.float32)

    def __init__(self, transitions):
        self.transitions = transitions
        self.heading = 0.0
        self.hoop_normal = np.array([1., 0, 0])
        self.hoop_centre = np.zeros(3)
        self._half_box = np.array([.1, .1, .1])
        self.hoop = SimpleNamespace(thickness=.06)

    def info(self, outcome):
        return dict(outcome=outcome, passes=self.passes, wall_steps=self.wall_steps)

    def reset(self, *, seed=None, options=None):
        self.steps = self.passes = self.wall_steps = 0
        self.pos = np.array([-.2, 0, 0])
        self.public = np.zeros(109, np.float32)
        self.public[88:96] = [0, .1, .2, .8, 0, .2, 1, 0]
        return self.public.copy(), self.info('running')

    def step(self, action):
        along, centre, collision, cutoff, wall = self.transitions[self.steps]
        self.steps += 1
        self.pos[0] = along
        self.passes += centre
        self.wall_steps += wall
        outcome = 'hit' if collision else 'passed' if centre else 'running'
        reward = contract.reward(outcome, 1., .9, wall, self.public[-21:-13], action)
        if cutoff and not collision:
            outcome = 'timeout'
        return self.public.copy(), reward, collision, cutoff, self.info(outcome)


def compare_fixture(transitions):
    arms = [ClearanceTrainingEnv(RewardFixture(transitions), a) for a in ('control', 'clearance')]
    for env in arms:
        env.reset()
    results = []
    for _ in transitions:
        c, t = [env.step(np.zeros(4, np.float32)) for env in arms]
        np.testing.assert_array_equal(c[0], t[0])
        assert c[2:] == t[2:]
        results.append(t[1] - c[1])
    return results, arms


def test_bonus_waits_for_entire_body_and_is_paid_only_once():
    differences, arms = compare_fixture([
        (.01, True, False, False, False),
        (.12, False, False, False, False),
        (.14, False, False, False, False),
        (.20, False, False, True, False)])
    np.testing.assert_allclose(differences, [-40, 0, 40, 0])
    assert arms[1].audit.count == 1


@pytest.mark.parametrize('ending', ['collision', 'wall', 'side', 'timeout_at_crossing'])
def test_incomplete_or_unsafe_pass_does_not_earn_bonus(ending):
    if ending == 'side':
        transitions = [(.2, False, False, True, False)]
        expected = [0]
    elif ending == 'timeout_at_crossing':
        transitions = [(.01, True, False, True, False)]
        expected = [-40]
    else:
        transitions = [(.01, True, False, False, False),
                       (.2, False, ending == 'collision', ending == 'wall', ending == 'wall')]
        expected = [-40, 0]
    differences, arms = compare_fixture(transitions)
    np.testing.assert_allclose(differences, expected)
    assert arms[1].audit.count == 0


def test_reset_discards_pending_transit_and_action_history():
    env = ClearanceTrainingEnv(RewardFixture([(.01, True, False, False, False)]), 'clearance')
    initial, _ = env.reset()
    env.step(np.ones(4, np.float32) * .25)
    assert env.audit.crossed
    reset, _ = env.reset()
    np.testing.assert_array_equal(reset, initial)
    assert not env.audit.crossed and env.audit.count == 0
    assert env.episode_seed == 11_000_001


@pytest.mark.parametrize('seed', [7_000_000, 9_410_000, 11_250_000])
def test_development_and_acceptance_seed_ranges_rejected(seed):
    env = ClearanceTrainingEnv(RewardFixture([]), 'control')
    with pytest.raises(ValueError, match='FIT range'):
        env.reset(seed=seed)


def test_invalid_action_refused_before_simulator_step():
    env = ClearanceTrainingEnv(RewardFixture([]), 'control')
    env.reset()
    for action in [np.full(4, np.nan), np.full(4, 1.1), np.zeros(3)]:
        with pytest.raises(ValueError, match='action'):
            env.step(action)
        assert env.unwrapped.steps == 0


def test_trace_retains_partial_episode_and_separates_truth(tmp_path):
    env = ClearanceTrainingEnv(RewardFixture([(.01, True, False, False, False)]),
                               'clearance', trace_dir=tmp_path/'traces')
    env.reset()
    env.step(np.zeros(4, np.float32))
    env.close()
    with np.load(next((tmp_path/'traces').glob('*.npz'))) as data:
        assert data['actor_observations'].shape == (1, 85)
        assert data['positions'].shape == (1, 3)
        assert data['centre_events'].tolist() == [True]
        assert data['clearance_events'].tolist() == [0]


def test_learned_rollout_preserves_public_history_and_physics():
    from pathlib import Path
    policy = LearnedPolicy(Path(__file__).resolve().parents[1]/'trainedModels/seenflag-s1-level1.zip', PROFILE)
    control = ClearanceTrainingEnv(HoopEnv(randomise=True), 'control')
    treatment = ClearanceTrainingEnv(HoopEnv(randomise=True), 'clearance')
    c, _ = control.reset()
    t, _ = treatment.reset()
    previous = np.zeros(4, np.float32)
    for _ in range(600):
        np.testing.assert_array_equal(c, t)
        expected, action = policy.predict(c[-21:-13], previous)
        np.testing.assert_array_equal(c, expected)
        previous = action
        c, rc, dc, tc, _ = control.step(action)
        t, rt, dt, tt, _ = treatment.step(action)
        np.testing.assert_array_equal(control.unwrapped.pos, treatment.unwrapped.pos)
        assert (dc, tc) == (dt, tt)
        if dc or tc:
            break
    assert control.audit.count == treatment.audit.count


def test_invalid_reward_arm_refused():
    with pytest.raises(ValueError):
        passage_reward_delta('unknown', False, 0)


@pytest.mark.parametrize('metric,value,stops', [
    ('first_pass', 18, False), ('first_pass', 17, True),
    ('crashes', 8, False), ('crashes', 9, True),
    ('clean_repeat', 15, False), ('clean_repeat', 14, True)])
def test_retention_stop_boundaries(metric, value, stops):
    from tools.run_clearance_experiment import retention_failures
    baseline = {'nominal': dict(episodes=25, first_pass=20, crashes=5, clean_repeat=18)}
    candidate = {'nominal': dict(baseline['nominal'], **{metric: value})}
    assert bool(retention_failures(baseline, candidate)) == stops


def test_single_training_lock_excludes_second_run(tmp_path):
    from tools.run_clearance_experiment import single_trainer
    with single_trainer(tmp_path/'trainer.lock'):
        with pytest.raises(RuntimeError, match='Another'):
            with single_trainer(tmp_path/'trainer.lock'):
                pass
    with single_trainer(tmp_path/'trainer.lock'):
        pass
