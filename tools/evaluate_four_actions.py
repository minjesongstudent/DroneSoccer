"""Explicit experimental 89-input evaluator; never load as the 85-input profile."""
import json
import numpy as np
from stable_baselines3 import SAC
from src.camera_history import CameraHistory
from src.policy_runtime import SPECS
from tools.validate_flight_policies import SUITES,episode
from tools.run_clearance_experiment import preserved_rng,write_json,sha


class FourActionPolicy:
    def __init__(self,path):
        self.model=SAC.load(path,device='cpu')
        assert self.model.observation_space.shape==(89,)
        self.reset()

    def reset(self):
        self.history=CameraHistory(SPECS['seenflag-s1-level1'].lags,4)

    def predict(self,detection,previous):
        obs=self.history.update(detection,previous)
        action,_=self.model.predict(obs,deterministic=True)
        assert action.shape==(4,) and np.isfinite(action).all() and np.all(abs(action)<=1)
        return obs,action


def evaluate_four_actions(checkpoint,folder,count,*,policy_class=FourActionPolicy,inputs=89,action_history=4):
    with preserved_rng():
        folder.mkdir(parents=True,exist_ok=False)
        write_json(folder/'profile.json',dict(inputs=inputs,lags=SPECS['seenflag-s1-level1'].lags,
            action_history=action_history,model_sha256=sha(checkpoint),acceptance=False))
        policy=policy_class(checkpoint);metrics={}
        for suite,base in SUITES.items():
            dest=folder/suite;dest.mkdir();rows=[]
            for seed in range(base,base+count):
                row=episode(policy,suite,seed,dest/f'{seed}.npz');rows.append(row)
                with (dest/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            metrics[suite]=dict(episodes=count,first_pass=sum(r['clean_first_pass'] for r in rows),
                crashes=sum(r['crashed'] for r in rows),returned=sum(r['returns_to_front']>0 for r in rows),
                clean_repeat=sum(r['clean_repeated_episode'] for r in rows),
                full_body_passes=sum(r['full_body_passes'] for r in rows))
        write_json(folder/'summary.json',metrics)
        return metrics
