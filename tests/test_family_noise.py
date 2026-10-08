import numpy as np
from tools.audit_family_noise import summarize


def test_equal_outcomes_do_not_count_tied_best_actions_as_agreement():
    result=summarize(np.zeros((3,4,6)))
    assert result['unique_best_both_halves_states']==0
    assert result['half_best_action_agreement'] is None
    assert result['half_contrast_sign_agreement'] is None


def test_identical_replica_rankings_agree_without_noise():
    data=np.broadcast_to(np.arange(4)[None,:,None],(3,4,6)).copy()
    result=summarize(data)
    assert result['unique_best_both_halves_states']==3
    assert result['half_best_action_agreement']==1
    assert result['half_contrast_sign_agreement']==1
    assert result['median_within_action_std']==0
