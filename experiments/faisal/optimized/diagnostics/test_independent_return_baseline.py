from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.independent_return_baseline import leave_one_out_baselines as baseline

class BaselineTests(unittest.TestCase):
    def test_own_outcome_excluded(self):
        data=[np.array([1.,2.,3.]),np.array([4.,6.]),np.array([8.])]
        before=baseline(data,[0,0,0]);data[0][:]=999
        after=baseline(data,[0,0,0])
        np.testing.assert_array_equal(before[0],after[0])
        np.testing.assert_array_equal(before[0],[6.,3.,0.])
        self.assertFalse(np.array_equal(before[1],after[1]))
    def test_groups_isolated(self):
        values=[np.array([1.]),np.array([3.]),np.array([100.]),np.array([200.])]
        got=baseline(values,[0,0,1,1])
        np.testing.assert_array_equal(np.concatenate(got),[3.,1.,200.,100.])
    def test_reject_invalid(self):
        for values,groups in [([np.array([1.])],[0]),([np.array([np.nan]),np.array([1.])],[0,0]),([np.array([]),np.array([1.])],[0,0])]:
            with self.assertRaises(ValueError):baseline(values,groups)
if __name__=='__main__':unittest.main()
