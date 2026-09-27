import pytest
from src.repair_anchor_policy import accepts

@pytest.mark.parametrize('before,after,expected',[(.05,.09,True),(.05,.1001,False),(.1,.101,False),(.11,.109,False),(.11,.108,True),(1.,.99,True),(1.,.999,False),(1.,1.01,False),(1.,.05,True),(float('nan'),.01,False),(.2,float('inf'),False),(-1,.01,False)])
def test_only_in_budget_or_strict_monotonic_repair(before,after,expected):
 assert accepts(before,after) is expected
