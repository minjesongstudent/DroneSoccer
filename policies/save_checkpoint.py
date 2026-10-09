"""Archive each completed PC residual checkpoint without promotion or overwrites."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'policies/lab-runner'))
from src.estimated_motion_actor import checkpoint_manifest

def save(checkpoint,evidence,run,arm,step,root=ROOT):
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',x) for x in (run,arm)) or step<0:
        raise ValueError('Safe run/arm names and nonnegative step required')
    payload=Path(checkpoint).read_bytes();digest=hashlib.sha256(payload).hexdigest()
    from io import BytesIO
    with np.load(BytesIO(payload),allow_pickle=False) as z:
        if set(z.files)!={'manifest','residual_weight'} or json.loads(z['manifest'].tobytes())!=checkpoint_manifest():
            raise ValueError('Checkpoint architecture mismatch')
        w=z['residual_weight']
        if w.dtype!=np.float32 or w.shape!=(4,258) or not np.isfinite(w).all():
            raise ValueError('Invalid correction weights')
    results=Path(evidence).read_bytes();metrics=json.loads(results)
    registry=root/'policies/registry.json';lock=registry.with_suffix('.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        data=json.loads(registry.read_text());shared=data['checkpoints'][0]['shared_models']
        ident=f'{run}-{arm}-step-{step:06d}'
        folder=root/'policies/checkpoints'/run/arm;folder.mkdir(parents=True,exist_ok=True)
        target=folder/f'step-{step:06d}.npz';result=folder/f'step-{step:06d}.evidence.json'
        for p,content in ((target,payload),(result,results)):
            if p.exists():
                if p.read_bytes()!=content:raise FileExistsError('Refusing to replace '+str(p))
            else:
                with p.open('xb') as f:f.write(content)
        entry=dict(id=ident,arm=arm,round=step,path=target.relative_to(root).as_posix(),sha256=digest,
            architecture='estimated-motion-residual-v1',inputs=89,status='saved-not-qualified',qualified=False,
            physical_flight_validated=False,source_run=run,shared_models=shared,
            guard_metrics=metrics,evidence=result.relative_to(root).as_posix(),evidence_sha256=hashlib.sha256(results).hexdigest())
        old=next((e for e in data['checkpoints'] if e['id']==ident),None)
        if old is not None and old!=entry:raise FileExistsError('Conflicting registry identity')
        if old is None:data['checkpoints'].append(entry)
        temp=registry.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8');temp.replace(registry)
        return entry
    finally:os.close(fd);lock.unlink()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','evidence','run','arm'):p.add_argument('--'+name,required=True)
    p.add_argument('--step',required=True,type=int)
    print(json.dumps(save(**vars(p.parse_args())),indent=2))
