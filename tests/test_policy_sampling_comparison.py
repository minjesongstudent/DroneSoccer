from copy import deepcopy
import pytest
from tools.run_policy_sampling_comparison import compare, MAIN_SEEDS, MODES


def records():
    rows = []
    for seed in MAIN_SEEDS:
        for mode in MODES:
            won = mode == 'deterministic'
            rows.append(dict(seed=seed, mode=mode, initial={'public85':[seed]},
                             status='complete', steps=600 if won else 50,
                             repeat=won, first=True, returned=won, crashed=not won,
                             wall_steps=0, passes=2 if won else 1))
    return rows


def test_whole_pair_gate():
    d = compare(records(), MAIN_SEEDS)
    assert d['supports_sampling_mismatch'] and d['groups']['all']['repeat_gain'] == 64
    assert not d['acceptance'] and not d['policy_promoted']


def test_smoke_never_advances():
    assert not compare(records(), MAIN_SEEDS, smoke=True)['supports_sampling_mismatch']


@pytest.mark.parametrize('mutation', ['missing','duplicate','partial','reset','early_repeat','contact_repeat'])
def test_malformed_evidence_rejected(mutation):
    r = records()
    if mutation == 'missing': r.pop()
    elif mutation == 'duplicate': r.append(deepcopy(r[0]))
    elif mutation == 'partial': r[0]['status'] = 'partial'
    elif mutation == 'reset': r[0]['initial'] = {'wrong':1}
    elif mutation == 'early_repeat': r[0]['steps'] = 40
    else: r[0]['wall_steps'] = 1
    with pytest.raises(ValueError): compare(r, MAIN_SEEDS)


def test_gains_cannot_hide_failed_stratum():
    r = records()
    for row in r:
        if row['seed'] % 2:
            row.update(repeat=True, returned=True, crashed=False, steps=600, passes=2)
    assert not compare(r, MAIN_SEEDS)['supports_sampling_mismatch']


def test_equality_and_average_replicas():
    r = records()
    for row in r:
        row.update(repeat=True, returned=True, crashed=False, steps=600, passes=2)
    d = compare(r, MAIN_SEEDS)
    assert d['groups']['all']['repeat_gain'] == 0 and not d['supports_sampling_mismatch']
