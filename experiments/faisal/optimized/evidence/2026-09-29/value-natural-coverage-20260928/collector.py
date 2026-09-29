"""Source-policy value data with predeclared fresh natural-flight holdout."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'diagnostics/acquisition-return-mixture-20260928'
OUT=ROOT/'diagnostics/value-natural-coverage-20260928'
sys.path.insert(0,str(SOURCE/'source_snapshot'))
from src.recurrent_snapshot_rollout import collect_natural_segment
from src.return_reset_curriculum import wrap
from src.sim import HoopEnv
from src.params import Drone
from src.curriculum import LessonEnv

def main():
    OUT.mkdir(exist_ok=False);shutil.copytree(SOURCE/'source_snapshot',OUT/'source_snapshot');shutil.copy2(__file__,OUT/'collector.py')
    torch.set_num_threads(1);path=SOURCE/'model_0.zip';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    cases=[dict(index=i,group='natural_nominal' if i<32 else 'natural_delayed' if i<64 else 'centered' if i<72 else 'varied',
                seed=2970000+i,torch_seed=988000+i,split='heldout' if i%4==0 else 'fit') for i in range(80)]
    (OUT/'protocol.json').write_text(json.dumps(dict(cases=cases,source=str(path),source_sha256=digest,gamma=.999,
        selection='All 80 predetermined flights retained, no outcome filtering',actor_updates=0),indent=2))
    model=RecurrentPPO.load(path,device='cuda');rows=[]
    for case in cases:
        i=case['index'];base=HoopEnv(level=1.,drone=Drone(delay_steps=0 if i<32 else 2)) if i<64 else LessonEnv(stage=0 if i<72 else 1)
        env=wrap(base,single=i>=64,metric_ring=True,geometric_return=True)
        torch.manual_seed(case['torch_seed']);s=collect_natural_segment(model,env,case['seed'],600)
        assert s.mission_ended
        returns=np.zeros(len(s.rewards),np.float32);carry=0.
        for j in range(len(returns)-1,-1,-1):carry=s.rewards[j]+.999*carry;returns[j]=carry
        np.savez_compressed(OUT/f'episode_{i}.npz',prefix=s.prefix,observations=s.observations,raw_actions=s.actions,
            rewards=s.rewards,returns=returns,log_probs=s.log_probs,next_observation=s.next_observation,natural_start=True)
        rows.append(dict(**case,steps=len(returns),outcome=s.final_info['outcome'],passes=s.final_info['passes']))
        env.close()
        if (i+1)%16==0:print('collected',i+1,flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    (OUT/'result.json').write_text(json.dumps(dict(rows=rows,source_unchanged=True,actor_updates=0),indent=2))
    print('complete',flush=True)
if __name__=='__main__':main()
