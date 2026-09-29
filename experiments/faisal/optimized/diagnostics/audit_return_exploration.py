"""Paired exploration-scale diagnostic; unchanged policy means, no training."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'diagnostics/recurrent-return-histories-20260928'
RUN=ROOT/'diagnostics/authentic-return-pg-guarded-20260928';OUT=RUN/'exploration_audit'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_burnin import rebuild_actor_memory

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    path=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip')
    digest=hashlib.sha256(path.read_bytes()).hexdigest();m=RecurrentPPO.load(path,device='cuda')
    initial=m.policy.log_std.detach().clone();rows=[]
    entries=[r for r in json.loads((BANK/'result.json').read_text())['snapshots'] if r['phase']=='return_side'][:8]
    for scale in (.5,1.,1.5):
        with torch.no_grad():m.policy.log_std.copy_(initial+np.log(scale))
        for i,entry in enumerate(entries):
            blob=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
            assert hashlib.sha256(blob).hexdigest()==entry['sha256']
            for rep in range(2):
                d=pickle.loads(blob);e=d['environment'];o=d['observation'];initial_passes=e.unwrapped.passes
                state,start=rebuild_actor_memory(m,d['prefix_observations']);torch.manual_seed(985000+100*i+rep)
                observations=[];actions=[];front=False
                while True:
                    observations.append(o.copy());a,state=m.predict(o,state=state,episode_start=start,deterministic=False)
                    start[:]=False;actions.append(a.copy());o,r,t,tr,info=e.step(a)
                    front=front or info['lap_phase']=='align_front'
                    if t or tr:break
                rows.append(dict(scale=scale,snapshot=i,replicate=rep,new_passes=info['passes']-initial_passes,
                    front=front,outcome=info['outcome'],steps=len(actions)))
                np.savez_compressed(OUT/f'{scale}_{i}_{rep}.npz',observations=observations,actions=actions,next_observation=o)
                e.close()
        print('finished scale',scale,flush=True)
    summary={str(s):dict(episodes=16,crashes=sum(r['outcome']!='timeout' for r in rows if r['scale']==s),
        front=sum(r['front'] for r in rows if r['scale']==s),new_passes=sum(r['new_passes'] for r in rows if r['scale']==s)) for s in (.5,1.,1.5)}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    (OUT/'result.json').write_text(json.dumps(dict(protocol='Same eight early-return snapshots, two shared seeds per scale; diagnostic only, means unchanged; no optimization',
        seed_rule='985000+100*i+rep',source_sha256=digest,summary=summary,rows=rows),indent=2))
    print(json.dumps(summary))
if __name__=='__main__':main()
