import copy
import io
import json

import pytest

from src import comparison


@pytest.fixture
def runs(tmp_path, monkeypatch):
    folders = {key: tmp_path/key for key in ('original', 'corrected', 'guided')}
    for folder in folders.values():
        (folder/'evaluations').mkdir(parents=True)
        (folder/'config.json').write_text(json.dumps({'geometry': {'drone': {'delay_steps': 0}}}))
    monkeypatch.setattr(comparison, 'BASELINE_RUN', folders['original'])
    monkeypatch.setattr(comparison, 'CORRECTED_RUN', folders['corrected'])
    monkeypatch.setattr(comparison, 'urlopen', lambda *a, **kw: io.StringIO('{"mode":"Training","steps":20000}'))
    return folders


def write(folder, step, stage=0, successes=0, guided=False, seed=3000000):
    suite = {'success_rate': successes/25, 'successful_episodes': successes,
             'total_passes': successes, 'crash_rate': .1}
    data = {'step': step, 'stage': stage, 'seed_start': seed, 'episodes_per_suite': 25,
            'suites': {k: copy.deepcopy(suite) for k in ('lesson', 'full_task', 'full_task_delay_200ms')}}
    if guided:
        data['suites']['original_centered'] = suite
        data['protocols'] = {}
        for name in ('original_centered', 'full_task', 'full_task_delay_200ms'):
            single = name == 'original_centered'
            data['protocols'][name] = {
                'id': 'centered-clean-pass-v1' if single else name + '-v1',
                'deterministic': True, 'max_steps': 150 if single else 600,
                'success': 'passed' if single else 'timeout and passes >= 2',
                'geometry': {'drone': {'delay_steps': 2 if name.endswith('200ms') else 0}},
            }
    (folder/'evaluations'/f'{step}.json').write_text(json.dumps(data))
    return data


def test_compare_latest_equal_online_steps_not_latest_wall_clock(runs):
    write(runs['original'], 10000, successes=1)
    write(runs['original'], 20000, successes=2)
    write(runs['corrected'], 10000, successes=3)
    write(runs['guided'], 10000, stage=3, successes=21, guided=True)
    result = comparison.read_comparison(runs['guided'])
    matched = result['matched']
    assert matched['step'] == 10000
    assert matched['suites']['original_centered']['original']['success_rate'] == 1/25
    assert matched['suites']['original_centered']['guided']['success_rate'] == 21/25
    assert set(matched['suites']) == {'original_centered', 'full_task', 'full_task_delay_200ms'}
    assert 'not equal total compute' in result['step_count_note']


def test_changed_legacy_lesson_not_compared_as_original_centered(runs):
    write(runs['original'], 10000)
    write(runs['corrected'], 10000, stage=1)
    write(runs['guided'], 10000, guided=True)
    matched = comparison.read_comparison(runs['guided'])['matched']
    assert 'original_centered' not in matched['suites']
    assert 'original_centered' in matched['omitted']
    assert set(matched['suites']) == {'full_task', 'full_task_delay_200ms'}


def test_seed_mismatch_is_not_comparable(runs):
    write(runs['original'], 10000, seed=1)
    write(runs['corrected'], 10000)
    write(runs['guided'], 10000, guided=True)
    result = comparison.read_comparison(runs['guided'])
    assert result['matched'] is None and 'differ' in result['warning']


def test_changed_geometry_is_not_comparable(runs):
    for key in runs:
        write(runs[key], 10000, guided=key == 'guided')
    (runs['original']/'config.json').write_text('{"geometry":{"drone":{"delay_steps":0,"width":5}}}')
    matched = comparison.read_comparison(runs['guided'])['matched']
    assert not matched['suites']
    assert len(matched['omitted']) == 3


def test_target_needs_two_consecutive_online_checkpoints(runs):
    write(runs['guided'], 0, successes=25, guided=True)
    write(runs['guided'], 10000, successes=20, guided=True)
    result = comparison.read_comparison(runs['guided'])
    assert not result['target']['met']
    assert result['target']['consecutive_checkpoints'] == 1
    write(runs['guided'], 20000, successes=21, guided=True)
    assert comparison.read_comparison(runs['guided'])['target']['met']
    write(runs['guided'], 30000, successes=19, guided=True)
    result = comparison.read_comparison(runs['guided'])
    assert not result['target']['met']
    assert result['latest_original_centered']['successful_episodes'] == 19


def test_no_common_checkpoint_is_not_a_zero_success_claim(runs):
    result = comparison.read_comparison(runs['guided'])
    assert result['matched'] is None
    assert result['latest_original_centered'] is None
