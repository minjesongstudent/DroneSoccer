import pytest
from tools.audit_critic_rankings import soft_return


def test_soft_q_excludes_first_action_entropy_and_discounts_tail():
    assert soft_return([2,3],[-100,-2],.5,.1,4)==pytest.approx(2+.5*3.2+.25*4)
    assert soft_return([2],[-100],.5,.1)==2


def test_entropy_zero_reduces_to_discounted_reward():
    assert soft_return([1,2,3],[-1,-2,-3],.9,0)==pytest.approx(1+.9*2+.81*3)
