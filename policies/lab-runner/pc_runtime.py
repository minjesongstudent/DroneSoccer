"""Explicit PC composite adapter. Imports and checks never contact a drone."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.camera_history import CameraHistory
from src.policy_runtime import SPECS
from src.estimated_motion_actor import EstimatedMotionPolicy

ROOT=Path(__file__).resolve().parents[2]

class PCRuntime:
    inputs=89
    require_visible_start=True
    startup_search=True

    def __init__(self, root=ROOT, checkpoint_id=None):
        root=Path(root).resolve()
        selection=json.loads((root/'policies/lab-runner/selection.json').read_text())
        ident=checkpoint_id or selection['checkpoint_id']
        entries=json.loads((root/'policies/registry.json').read_text())['checkpoints']
        self.identity=next((e for e in entries if e['id']==ident),None)
        if self.identity is None:raise ValueError('Unknown checkpoint identity')
        if checkpoint_id is None and self.identity['sha256']!=selection['sha256']:
            raise ValueError('Selection checksum does not match registry')
        def checked(info):
            p=(root/info['path']).resolve()
            if not p.is_relative_to(root):raise ValueError('Artifact outside package')
            if hashlib.sha256(p.read_bytes()).hexdigest()!=info['sha256']:
                raise ValueError('Artifact checksum mismatch: '+info['path'])
            return p
        head=checked(self.identity)
        base=checked(self.identity['shared_models']['baseline'])
        estimator=checked(self.identity['shared_models']['estimator'])
        torch.set_num_threads(1)
        self.policy=EstimatedMotionPolicy.load(head,base,estimator,device='cpu',sha256=self.identity['sha256'])
        self.spec=SimpleNamespace(name=ident,sha256=self.identity['sha256'],inputs=89,lags=SPECS['seenflag-s1-level1'].lags)
        self.reset()
        # Warm CPU inference before any hardware connection, then reset recurrence.
        for _ in range(3):self.predict(np.zeros(8,np.float32),np.zeros(4,np.float32))
        self.reset()

    def reset(self):
        self.history=CameraHistory(SPECS['seenflag-s1-level1'].lags)
        self.policy.reset()

    def observe(self,detection,previous_issued_action):
        public85=self.history.update(detection,previous_issued_action)
        return self.policy.observe(public85)

    def predict(self,detection,previous_issued_action):
        features=self.observe(detection,previous_issued_action)
        action,_=self.policy.predict(features,deterministic=True)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('Invalid PC policy output')
        return features,action

if __name__=='__main__':
    runtime=PCRuntime()
    print(json.dumps(runtime.identity,indent=2))
    print('Verified all three model components; CPU inference works. No hardware contacted.')
