"""Source policy on all seven authentic reapproaches; no optimization."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'diagnostics/authentic-reapproaches-20260928'
OUT=BANK/'policy_audit'
sys.path.insert(0,str(BANK/'source_snapshot'))
from src.recurrent_burnin import rebuild_actor_memory

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    source=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip')
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    m=RecurrentPPO.load(source,device='cuda');rows=[]
    for i,entry in enumerate(json.loads((BANK/'result.json').read_text())['snapshots']):
        blob=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
        assert hashlib.sha256(blob).hexdigest()==entry['sha256']
        for rep in range(4):
            d=pickle.loads(blob);e=d['environment'];o=d['observation'];initial=e.unwrapped.passes
            state,start=rebuild_actor_memory(m,d['prefix_observations']);torch.manual_seed(984000+100*i+rep)
            observations=[];actions=[]
            while True:
                observations.append(o.copy())
                a,state=m.predict(o,state=state,episode_start=start,deterministic=rep==0)
                start[:]=False;actions.append(a.copy());o,r,t,tr,info=e.step(a)
                if t or tr:break
            rows.append(dict(id=entry['id'],replicate=rep,new_passes=info['passes']-initial,outcome=info['outcome'],
                steps=len(actions),noodle=bool(e.unwrapped._hit()),hanger=bool(e.unwrapped._hits_hanger())))
            np.savez_compressed(OUT/f'{i}_{rep}.npz',observations=observations,actions=actions,next_observation=o)
            e.close()
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    result=dict(source_sha256=before,seed_rule='984000+100*snapshot_index+replicate',rows=rows,acceptance=False)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(rows))
if __name__=='__main__':main()
