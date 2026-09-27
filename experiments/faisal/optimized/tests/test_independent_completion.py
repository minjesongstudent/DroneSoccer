"""Completion testing must remain separate from training and model selection."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import live
from src.evaluation_guard import EvaluationGuard, SUITES


DEFAULT_RATES = {'original_centered': .85, 'varied_approach': .80,
                 'full_task': .80, 'full_task_delay_200ms': .80, 'lesson': 0.}


def evaluation_report(step, rates=None, *, episodes=100, seed_start=7_000_000):
    rates = DEFAULT_RATES | (rates or {})
    suites = {}
    for name in SUITES:
        successful = round(rates[name] * episodes)
        single = name in ('original_centered', 'varied_approach')
        records = [{'seed': seed_start + index,
                    'outcome': ('passed' if single else 'timeout') if index < successful else 'hit',
                    'passes': (1 if single else 2) if index < successful else 0,
                    'steps': 100 if single else 600}
                   for index in range(episodes)]
        suites[name] = {'successful_episodes': successful, 'success_rate': successful / episodes,
                        'crash_rate': (episodes - successful) / episodes, 'records': records}
    return {'step': step, 'stage': 4, 'episodes_per_suite': episodes,
            'seed_start': seed_start, 'policy_mode': 'deterministic', 'suites': suites}


class SnapshotModel:
    """No learning implementation: unexpected inference or updates fail loudly."""

    def __init__(self, steps=100_000):
        self.num_timesteps = steps
        self._n_updates = 321
        self.weights = {'actor': [.25, -.5], 'critic': [1., 2.]}
        self.replay_buffer = {'transitions': [{'obs': [1, 2], 'action': [.3], 'reward': 4.}]}
        self.seed = 7

    def snapshot(self):
        return deepcopy(vars(self))

    @staticmethod
    def _write(destination, payload):
        data = json.dumps(payload).encode('utf-8')
        if hasattr(destination, 'write'):
            destination.write(data)
        else:
            Path(destination).write_bytes(data)

    def save(self, destination):
        self._write(destination, {'steps': self.num_timesteps, 'updates': self._n_updates,
                                  'weights': self.weights})

    def save_replay_buffer(self, destination):
        self._write(destination, self.replay_buffer)

    def load_replay_buffer(self, path):
        self.replay_buffer = json.loads(Path(path).read_text())

    def set_random_seed(self, seed):
        self.seed = seed

    def learn(self, *args, **kwargs):
        raise AssertionError('Independent testing must never train the policy')

    def predict(self, *args, **kwargs):
        raise AssertionError('Use the loaded best checkpoint, not the training model')


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    session = live.Session(100_000, 'completion-test')
    session.model = SnapshotModel()
    session.update(mode='Replay', steps=100_000)
    session.stop.set()
    for relative, data in {
        'best_model.zip': b'selected best model',
        'initial_model.zip': b'preserved initial policy',
        'model.zip': b'latest training model',
        'resume-0/model.zip': b'training snapshot',
        'resume-0/replay_buffer.pkl': b'training experience',
        'resume-0/state.json': b'{"training": "preserved"}',
        'resume.json': b'{"slot": 0, "step": 100000}',
    }.items():
        path = session.out / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
    return session


def files_under(folder):
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob('*') if path.is_file()}


def mock_fresh_evaluation(monkeypatch, session, rates=None):
    candidate = SimpleNamespace(num_timesteps=20_000)
    calls = {'loads': 0, 'evaluations': 0}

    def load(path, *, device):
        assert path == session.out / 'best_model.zip'
        assert path.read_bytes() == b'selected best model'
        assert device == 'cuda'
        calls['loads'] += 1
        return candidate

    def evaluate(model, stage, *, episodes, seed_start, progress):
        assert model is candidate and model is not session.model
        assert stage == session.curriculum.stage
        assert episodes == 100
        assert seed_start == 7_000_000
        assert set(range(seed_start, seed_start + episodes)).isdisjoint(range(5_000_000, 5_000_025))
        assert callable(progress)
        calls['evaluations'] += 1
        progress('full_task', 1, episodes)
        assert session.state['eval_progress'] == 'Fresh full_task: 1/100'
        return evaluation_report(candidate.num_timesteps, rates, episodes=episodes, seed_start=seed_start)

    monkeypatch.setattr(live.UnifiedSAC, 'load', staticmethod(load))
    monkeypatch.setattr(live, 'evaluate', evaluate)
    return candidate, calls


def test_fresh_test_uses_best_checkpoint_and_preserves_training_and_snapshot_files(session, monkeypatch):
    candidate, calls = mock_fresh_evaluation(monkeypatch, session)
    model_before = session.model.snapshot()
    replay_before = session.model.replay_buffer
    files_before = files_under(session.out)
    guard_before = session.guard.to_dict()
    session.run_independent_test()

    assert calls == {'loads': 1, 'evaluations': 1}
    assert session.model.snapshot() == model_before
    assert session.model.replay_buffer is replay_before
    assert session.guard.to_dict() == guard_before
    assert session.state['steps'] == session.model.num_timesteps == 100_000
    assert session.stop.is_set() and not session.wake.is_set()
    result = session.state['independent_test']
    assert result['step'] == candidate.num_timesteps == 20_000
    assert result['episodes_per_suite'] == 100 and result['seed_start'] == 7_000_000
    assert result['policy_mode'] == 'deterministic'
    assert result['accepted_in_simulation'] is True
    assert result['physical_flight_validated'] is False
    assert result['acceptance_thresholds'] == {name: DEFAULT_RATES[name] for name in SUITES if name != 'lesson'}
    assert session.state['eval_progress'] == ''
    for suite in result['suites'].values():
        assert [record['seed'] for record in suite['records']] == list(range(7_000_000, 7_000_100))
    files_after = files_under(session.out)
    assert set(files_after) - set(files_before) == {'independent-test.json'}
    assert {name: files_after[name] for name in files_before} == files_before
    assert json.loads((session.out / 'independent-test.json').read_text()) == result


@pytest.mark.parametrize('name,failed_rate', [
    ('original_centered', .84), ('varied_approach', .79),
    ('full_task', .79), ('full_task_delay_200ms', .79)])
def test_every_completion_gate_is_required_even_if_other_suites_succeed(session, monkeypatch, name, failed_rate):
    rates = {suite: 1. for suite in SUITES} | {name: failed_rate}
    mock_fresh_evaluation(monkeypatch, session, rates)
    session.run_independent_test()
    result = session.state['independent_test']
    assert result['accepted_in_simulation'] is False
    assert result['physical_flight_validated'] is False
    assert result['suites'][name]['success_rate'] == failed_rate
    assert json.loads((session.out / 'independent-test.json').read_text())['accepted_in_simulation'] is False


def test_saved_independent_result_is_reused_without_loading_or_evaluating_again(session, monkeypatch):
    _, calls = mock_fresh_evaluation(monkeypatch, session, {'full_task': .79})
    session.run_independent_test()
    before = files_under(session.out)
    saved_result = deepcopy(session.state['independent_test'])
    session.state.pop('independent_test')

    def forbidden(*args, **kwargs):
        raise AssertionError('An already completed independent test must not run again')

    monkeypatch.setattr(live.UnifiedSAC, 'load', staticmethod(forbidden))
    monkeypatch.setattr(live, 'evaluate', forbidden)
    session.run_independent_test()
    assert calls == {'loads': 1, 'evaluations': 1}
    assert session.state['independent_test'] == saved_result
    assert session.state['independent_test']['accepted_in_simulation'] is False
    assert files_under(session.out) == before


def stopped_guard():
    guard = EvaluationGuard()
    baseline = {'original_centered': .96, 'varied_approach': .88,
                'full_task': .04, 'full_task_delay_200ms': 0.}
    guard.observe(evaluation_report(0, baseline, episodes=25, seed_start=5_000_000))
    for step in (20_000, 30_000, 40_000):
        guard.observe(evaluation_report(step, baseline | {'original_centered': .20},
                                        episodes=25, seed_start=5_000_000))
    assert guard.should_stop and guard.stop_reason == 'severe_regression'
    return guard


@pytest.mark.parametrize('command', ['resume', 'extend'])
def test_guard_stop_cannot_be_bypassed_by_live_controls(session, command):
    session.guard = stopped_guard()
    before = deepcopy(session.state)
    with pytest.raises(ValueError, match='Evaluation stopped this experiment'):
        session.control(command)
    assert session.state == before
    assert session.stop.is_set() and not session.wake.is_set()


def test_disk_resume_restores_guard_stop_and_training_snapshot_without_allowing_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    original = live.Session(100_000, 'saved-stop')
    original.model = SnapshotModel(40_000)
    original.guard = stopped_guard()
    original.last_evaluated = 40_000
    original.update(mode='Replay', steps=40_000, stop_reason=original.guard.stop_reason)
    callback = SimpleNamespace(recent=[], history=[], total_passes=12,
                               episodes=30, elapsed=lambda: 123.5)
    env = live.MissionObservation(live.LessonEnv())
    config = {'unified_version': live.VERSION, 'reward_version': live.REWARD_VERSION,
              'observation_version': live.OBSERVATION_VERSION,
              'geometry': {key: asdict(getattr(env.unwrapped, key))
                           for key in ('cage', 'drone', 'hoop', 'camera', 'pad')},
              'seed': 7, 'eval_every': 10_000, 'eval_episodes': 25, 'learning_starts': 0}
    (original.out / 'config.json').write_text(json.dumps(config))
    original.save(callback)
    pointer = json.loads((original.out / 'resume.json').read_text())
    assert pointer['step'] == 40_000
    folder = original.out / f'resume-{pointer["slot"]}'
    assert {path.name for path in folder.iterdir()} == {'model.zip', 'replay_buffer.pkl', 'state.json'}
    saved = json.loads((folder / 'state.json').read_text())
    assert saved['evaluation_guard'] == original.guard.to_dict()
    files_before = files_under(original.out)

    def load(path, *, env, device):
        assert path == folder / 'model.zip'
        assert env.unwrapped is not None and device == 'cuda'
        payload = json.loads(path.read_text())
        model = SnapshotModel(payload['steps'])
        model.weights = payload['weights']
        model._n_updates = payload['updates']
        model.replay_buffer = None
        return model

    monkeypatch.setattr(live.UnifiedSAC, 'load', staticmethod(load))
    resumed = live.Session(200_000, 'saved-stop', resume=True)
    try:
        restored = resumed.load(env)
        assert restored == saved
        assert resumed.guard.to_dict() == original.guard.to_dict()
        assert resumed.guard.stop_reason == 'severe_regression'
        assert resumed.stop.is_set() and not resumed.wake.is_set()
        assert resumed.model.weights == original.model.weights
        assert resumed.model.replay_buffer == original.model.replay_buffer
        assert resumed.model.num_timesteps == original.model.num_timesteps
        assert resumed.model._n_updates == original.model._n_updates
        assert resumed.model.seed == 7 + 40_000
        assert resumed.target() == 200_000
        for command in ('resume', 'extend'):
            with pytest.raises(ValueError, match='Evaluation stopped this experiment'):
                resumed.control(command)
        assert files_under(original.out) == files_before
    finally:
        env.close()
