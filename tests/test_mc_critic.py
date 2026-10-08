import numpy as np
from tools.calibrate_mc_critic import soft_targets
from tools.audit_critic_rankings import soft_return


def test_reverse_targets_match_explicit_full_suffix_returns():
    rewards=[1.,-2.,4.,5.];logps=[-3.,.2,-.1,2.]
    targets=soft_targets(rewards,logps,.995,.024,-7.)
    for i in range(len(rewards)):
        np.testing.assert_allclose(targets[i],soft_return(rewards[i:],logps[i:],.995,.024,-7.),rtol=1e-6)


def test_terminal_target_excludes_its_own_action_entropy():
    np.testing.assert_array_equal(soft_targets([-30],[-100],.995,.024,0),[-30])
