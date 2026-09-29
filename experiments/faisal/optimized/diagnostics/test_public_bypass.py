from pathlib import Path
import sys,tempfile,unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from src.public_bypass_policy import add_public_bypass,PublicBypassPolicy

SOURCE='D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip'

class BypassTests(unittest.TestCase):
    def test_zero_parity_training_paths_reset_and_roundtrip(self):
        torch.set_num_threads(1);torch.manual_seed(80)
        old=RecurrentPPO.load(SOURCE,device='cuda')
        new=add_public_bypass(RecurrentPPO.load(SOURCE,device='cuda'))
        old.policy.set_training_mode(False);new.policy.set_training_mode(False)
        data=np.load('D:/DroneSoccerData/diagnostics/learned-success-collection-v3/trajectories/2820026.npz')['observations'][:32]
        state_a=state_b=None
        for i,o in enumerate(data):
            starts=np.array([i in (0,17)])
            a,state_a=old.predict(o,state=state_a,episode_start=starts,deterministic=True)
            b,state_b=new.predict(o,state=state_b,episode_start=starts,deterministic=True)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(state_a[0],state_b[0])
        x=torch.as_tensor(data,device='cuda');z=torch.zeros(1,1,128,device='cuda')
        starts=torch.zeros(32,device='cuda');starts[0]=starts[17]=1
        states=RNNStates((z,z.clone()),(z.clone(),z.clone()))
        with torch.no_grad():
            a,v,logp,_=new.policy(x,states,starts,deterministic=True)
            dist,_=new.policy.get_distribution(x,states.pi,starts)
            v2,logp2,_=new.policy.evaluate_actions(x,a,states,starts)
            torch.testing.assert_close(a,dist.distribution.mean)
            torch.testing.assert_close(v,v2);torch.testing.assert_close(logp,logp2)
        new.policy.set_training_mode(True)
        new.policy.optimizer.zero_grad()
        dist,_=new.policy.get_distribution(x,states.pi,starts)
        (dist.distribution.mean-.25).square().mean().backward()
        skip=new.policy.lstm_actor.public_skip.weight
        self.assertGreater(float(skip.grad.norm()),0)
        self.assertTrue(any(skip is p for g in new.policy.optimizer.param_groups for p in g['params']))
        new.policy.optimizer.step()
        self.assertGreater(float(skip.detach().norm()),0)
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'model.zip';new.save(target)
            loaded=RecurrentPPO.load(target,device='cuda')
            self.assertIsInstance(loaded.policy,PublicBypassPolicy)
            s1=s2=None
            for i,o in enumerate(data):
                start=np.array([i in (0,17)])
                a,s1=new.predict(o,state=s1,episode_start=start,deterministic=True)
                b,s2=loaded.predict(o,state=s2,episode_start=start,deterministic=True)
                np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
