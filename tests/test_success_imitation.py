import numpy as np
from tools.run_success_imitation import success_indices


def test_only_complete_clean_fit_successes_selected():
    clean=dict(steps=600,status='complete',outcome='timeout',full_body_passes=2,wall_steps=0)
    rows=[clean,dict(clean,outcome='hit'),dict(clean,wall_steps=1),dict(clean,full_body_passes=1),clean,clean]
    actual=success_indices(rows,3500) # final episode exceeds preserved replay
    np.testing.assert_array_equal(actual,np.arange(600)) # fifth episode is held out
    actual=success_indices(rows,3600)
    np.testing.assert_array_equal(actual,np.r_[np.arange(600),np.arange(3000,3600)])


def test_partial_or_short_episode_never_selected():
    row=dict(steps=599,status='complete',outcome='timeout',full_body_passes=2,wall_steps=0)
    assert len(success_indices([row],600))==0
    assert len(success_indices([dict(row,steps=600,status='budget_end')],600))==0
