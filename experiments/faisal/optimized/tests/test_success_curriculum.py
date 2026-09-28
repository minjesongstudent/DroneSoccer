import pytest
from src.success_curriculum import BackwardScheduler, SuccessMixture
from src.success_history import FIT, LANDMARKS, SuccessHistoryLibrary
from test_success_history import BANK

def rows(level, success=True):
    return [dict(seed=s, landmark=m, replicate=r, success=success)
            for s in FIT for m in LANDMARKS[:level+1] for r in range(9)]

def test_scheduler_requires_repeated_complete_fit_competence():
    s = BackwardScheduler()
    assert not s.observe(0, rows(0))
    assert s.observe(5000, rows(0)) and s.level == 1
    assert not s.observe(10000, rows(1, False))
    assert s.level == 1 and s.streak == 0
    bad = rows(1); bad[0]['seed'] = 2810380
    with pytest.raises(ValueError): s.observe(15000, bad)
    with pytest.raises(ValueError): s.observe(10000, rows(1))

@pytest.mark.skipif(not BANK.exists(), reason='Local bank required')
def test_mixture_only_fit_and_enabled_stages():
    s = BackwardScheduler(); env = SuccessMixture(SuccessHistoryLibrary(BANK), s)
    kinds = set()
    for seed in range(60):
        obs, info = env.reset(seed=seed)
        assert obs.shape == (119,)
        if info.get('return_practice'):
            assert info['snapshot_seed'] in FIT
            assert env.practice.record['landmark'] == 'near_pass'
            assert not info['practice_success']
            kinds.add('practice')
        else:
            kinds.add('single' if obs[99] else 'full')
    assert kinds == {'single', 'full', 'practice'}
    env.close()
