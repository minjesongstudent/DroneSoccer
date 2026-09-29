"""Post-hoc paired source check for the batch with additional passages."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'
BANK=ROOT/'diagnostics/recurrent-return-histories-20260928'
OUT=RUN/'paired_source_batch3'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_snapshot_rollout import collect_segment

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    protocol=json.loads((RUN/'protocol.json').read_text())
    source=Path(protocol['source']);assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['sha256']
    m=RecurrentPPO.load(source,device='cuda');rows=[]
    for i,entry in enumerate(protocol['snapshots']):
        data=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
        d=pickle.loads(data);e=d['environment'];initial=e.unwrapped.passes
        torch.manual_seed(983200+i)
        s=collect_segment(m,e,d['observation'],d['prefix_observations'],600)
        rows.append(dict(snapshot=i,new_passes=s.final_info['passes']-initial,outcome=s.final_info['outcome'],steps=len(s.rewards)))
        np.savez_compressed(OUT/f'{i}.npz',observations=s.observations,raw_actions=s.actions,rewards=s.rewards)
        e.close()
    result=dict(selection='Post-hoc batch3 with same snapshots and torch seeds; diagnostic not acceptance',rows=rows)
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
