"""Return scoring must not count inherited passes, hidden phases, or crashes."""
import hashlib
import json

import numpy as np
import pytest

from src import return_evaluation as evaluation
from src.curriculum import REWARD_VERSION
from src.mission_observation import VERSION as OBSERVATION_VERSION
from src.return_practice import ReturnSnapshotLibrary
from src.sim import HoopEnv


class PredictOnly:
    def __init__(self):
        self.num_timesteps = 123
        self._n_updates = 99
        self.calls = 0

    def predict(self, observation, deterministic=False):
        assert deterministic and observation.shape == (100,)
        self.calls += 1
        return np.zeros(4, np.float32), None

    def learn(self, *args, **kwargs):
        pytest.fail('An evaluator must never train')


@pytest.fixture
def library(tmp_path):
    # A validated-library test double: environment restore is replaced below,
    # while real geometric plane/whole-body-clearance scoring is retained.
    value = object.__new__(ReturnSnapshotLibrary)
    value.directory = tmp_path
    value.records = [{'id': f'sample-{index}', 'seed': 8_000_000+index, 'source_step': 100,
                      'historical_passes': 3, 'phase': 'return_side' if index < 30 else 'return_front'}
                     for index in range(60)]
    value.manifest = {'split': 'development', 'seed_start': 8_000_000, 'seed_end': 8_000_059,
                      'minimum_remaining_steps': 200, 'source_sha256': 'f'*64,
                      'reward_version': REWARD_VERSION, 'observation_version': OBSERVATION_VERSION,
                      'snapshots': value.records}
    (tmp_path/'manifest.json').write_text(json.dumps(value.manifest))
    return value


class ScriptedPhysicalTrace:
    """Controlled scoring events; this class is never used for training."""
    positions = {1: [1.02, 1.6, 1.], 2: [.98, 1.6, 1.], 3: [.80, 1.6, 1.]}
    new_pass_at = None
    crash_at = None
    early_timeout = None
    fake_phase = 'return_side'

    def __init__(self, library, horizon):
        assert horizon == 200
        self.library, self.horizon = library, horizon
        self.unwrapped = HoopEnv()
        self.closed = False

    def reset(self, *, seed, options):
        self.record = next(r for r in self.library.records if r['id'] == options['snapshot_id'])
        self.unwrapped.place([1.4, 1.6, 1.], 0., [1., 1., 1.], [1., 0., 0.])
        self.unwrapped.passes = self.record['historical_passes']
        self.elapsed = 0
        info = {'snapshot_id': self.record['id'], 'snapshot_seed': self.record['seed'],
                'source_step': self.record['source_step']}
        return np.zeros(100, np.float32), info

    def step(self, action):
        self.elapsed += 1
        if self.elapsed in self.positions:
            self.unwrapped.pos = np.array(self.positions[self.elapsed], float)
        if self.elapsed == self.new_pass_at:
            self.unwrapped.passes += 1
        collision = self.elapsed == self.crash_at
        timeout = self.elapsed == (self.early_timeout or self.horizon) and not collision
        outcome = 'hit' if collision else 'timeout' if timeout else 'running'
        return np.zeros(100, np.float32), 0., collision, timeout, {
            'outcome': outcome, 'lap_phase': self.fake_phase,
            # Deliberately wrong cumulative info demonstrates that the scorer
            # derives new passes from the physical counter minus its start.
            'passes': self.unwrapped.passes}

    def close(self):
        self.closed = True


def run(monkeypatch, library, trace=ScriptedPhysicalTrace, episodes=1, model=None):
    monkeypatch.setattr(evaluation, 'ReturnPracticeEnv', trace)
    return evaluation.evaluate_return_practice(model or PredictOnly(), library, episodes=episodes)


def test_front_return_uses_actual_outside_crossing_and_body_clearance_not_phase(monkeypatch, library):
    suite, protocol = run(monkeypatch, library)
    assert suite['safe_front_return_episodes'] == 1
    record = suite['records'][0]
    assert record['front_reached_step'] == 3  # Plane crossed on 2, body cleared on 3.
    assert record['outside_back_to_front_crossing_steps'] == [2]
    assert record['historical_passes'] >= 1 and record['new_clean_passes'] == 0
    assert not record['successful'] and suite['successful_episodes'] == 0
    assert protocol['full_mission_test'] is False and not protocol['historical_passes_counted']


@pytest.mark.parametrize('positions', [{}, {1: [.98, 1.6, 1.]}])
def test_phase_or_plane_only_does_not_count_complete_front_return(monkeypatch, library, positions):
    class Trace(ScriptedPhysicalTrace):
        fake_phase = 'align_front'
    Trace.positions = positions
    suite, _ = run(monkeypatch, library, Trace)
    assert suite['safe_front_return_episodes'] == suite['front_reached_episodes'] == 0


def test_reverse_crossing_through_hole_is_not_a_safe_front_return(monkeypatch, library):
    class Trace(ScriptedPhysicalTrace):
        # Starting at (1.4,1.6), this segment intersects x=1 at y=1.
        positions = {1: [.80, .7, 1.]}
        crash_at = 1
        fake_phase = 'align_front'
    suite, _ = run(monkeypatch, library, Trace)
    assert suite['crash_rate'] == 1 and suite['safe_front_return_episodes'] == 0
    assert suite['records'][0]['outside_back_to_front_crossing_steps'] == []


def test_a_new_pass_requires_survival_of_all_200_steps(monkeypatch, library):
    class Trace(ScriptedPhysicalTrace):
        new_pass_at = 5
    suite, _ = run(monkeypatch, library, Trace)
    assert suite['successful_episodes'] == 1 and suite['total_passes'] == 1
    assert suite['records'][0]['steps'] == 200
    Trace.crash_at = 200
    crashed, _ = run(monkeypatch, library, Trace)
    assert crashed['successful_episodes'] == crashed['safe_front_return_episodes'] == 0
    assert crashed['total_passes'] == crashed['front_reached_episodes'] == 1
    assert crashed['front_reached_then_crash_episodes'] == 1 and crashed['crash_rate'] == 1.


def test_early_timeout_does_not_count_survival(monkeypatch, library):
    class Trace(ScriptedPhysicalTrace):
        new_pass_at = 5
        early_timeout = 50
    suite, _ = run(monkeypatch, library, Trace)
    assert suite['front_reached_episodes'] == 1
    assert suite['successful_episodes'] == suite['safe_front_return_episodes'] == 0
    assert suite['records'][0]['steps'] == 50 and not suite['records'][0]['survived_horizon']


def test_seed_slots_protocol_manifest_and_snapshot_selection_are_fixed(monkeypatch, library):
    first, protocol = run(monkeypatch, library, episodes=50)
    second, repeat = run(monkeypatch, library, episodes=50)
    assert first == second and protocol == repeat
    assert [r['seed'] for r in first['records']] == list(range(8_500_000, 8_500_050))
    assert len({r['seed'] for r in first['records']}) == 50
    assert first['unique_snapshots'] == first['unique_source_flights'] == 50
    assert protocol['phase_counts'] == {'return_front': 25, 'return_side': 25}
    assert protocol['bank_manifest_sha256'] == hashlib.sha256((library.directory/'manifest.json').read_bytes()).hexdigest()


def test_schedule_balances_overlapping_phase_snapshots_without_duplicate_flights():
    records = [{'id': f'{phase}-{seed}', 'seed': seed, 'phase': phase}
               for phase, seeds in [('return_front', range(35)), ('return_side', range(10, 50))]
               for seed in seeds]
    schedule = evaluation._snapshot_schedule(records, 50, 8_500_000)
    assert len({r['seed'] for r in schedule}) == 50
    assert sum(r['phase'] == 'return_front' for r in schedule) == 25
    assert schedule == evaluation._snapshot_schedule(records, 50, 8_500_000)


def test_schedule_uses_available_rare_phase_and_rejects_insufficient_unique_sources(monkeypatch, library):
    records = [{'id': f'{phase}-{seed}', 'seed': seed, 'phase': phase}
               for phase, seeds in [('return_front', range(10)), ('return_side', range(60))]
               for seed in seeds]
    schedule = evaluation._snapshot_schedule(records, 50, 8_500_000)
    assert sum(r['phase'] == 'return_front' for r in schedule) == 10
    assert len({r['seed'] for r in schedule}) == 50
    library.records[:] = library.records[:2]
    (library.directory/'manifest.json').write_text(json.dumps(library.manifest))
    with pytest.raises(ValueError, match='50 distinct source flights'):
        run(monkeypatch, library, episodes=50)


@pytest.mark.parametrize('change', [
    {'split': 'train'}, {'seed_start': 7_000_000}, {'minimum_remaining_steps': 199},
])
def test_rejects_wrong_split_reserved_final_seeds_and_short_library(monkeypatch, library, change):
    library.manifest.update(change)
    (library.directory/'manifest.json').write_text(json.dumps(library.manifest))
    with pytest.raises(ValueError):
        run(monkeypatch, library)


@pytest.mark.parametrize('seed,episodes', [(7_000_000, 50), (8_999_999, 2), (8_500_000, 0), (8_500_000, True)])
def test_rejects_invalid_selection_seeds_or_counts(library, seed, episodes):
    with pytest.raises(ValueError):
        evaluation.evaluate_return_practice(PredictOnly(), library, episodes=episodes, seed_start=seed)


def test_evaluation_only_predicts_deterministically_and_reports_progress(monkeypatch, library):
    monkeypatch.setattr(evaluation, 'ReturnPracticeEnv', ScriptedPhysicalTrace)
    progress = []
    model = PredictOnly()
    evaluation.evaluate_return_practice(model, library, episodes=2, progress=lambda *x: progress.append(x))
    assert model.calls == 400 and model.num_timesteps == 123 and model._n_updates == 99
    assert progress == [('return_practice', 1, 2), ('return_practice', 2, 2)]


def test_detects_accidental_model_counter_mutation(monkeypatch, library):
    class Mutating(PredictOnly):
        def predict(self, observation, deterministic=False):
            self.num_timesteps += 1
            return super().predict(observation, deterministic)
    with pytest.raises(RuntimeError, match='Model weights or training counters changed'):
        run(monkeypatch, library, model=Mutating())
