from pathlib import Path
import sys,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.return_objective_mask import reinforcement_coefficient

class ObjectiveMaskTests(unittest.TestCase):
    def test_basic_gradient_zero(self):
        x=torch.tensor(2.,requires_grad=True)
        (reinforcement_coefficient({'natural':'basic'})*x.square()).backward()
        self.assertEqual(x.grad.item(),0.)
    def test_return_and_full_gradient_preserved(self):
        for entry in ({'natural':'full'},{'bank':'authentic'}):
            x=torch.tensor(2.,requires_grad=True)
            (reinforcement_coefficient(entry)*x.square()).backward()
            self.assertEqual(x.grad.item(),4.)
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):reinforcement_coefficient({'natural':'other'})
if __name__=='__main__':unittest.main()
