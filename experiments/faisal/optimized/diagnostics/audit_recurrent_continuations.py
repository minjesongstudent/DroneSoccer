"""Bounded learned continuations after authentic recurrent burn-in, FIT only."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np,torch
R=Path(__file__).parent;BANK=R/'recurrent-return-histories-20260928';OUT=BANK/'continuation_audit'
sys.path.insert(0,str(BANK/'source_snapshot'))
import src
src.__path__.append(str(R.parent/'src'))
from src.recurrent_burnin import rebuild_actor_memory
from sb3_contrib import RecurrentPPO

def main():
    OUT.mkdir(exist_ok=False)
    entries=[r for r in json.loads((BANK/'result.json').read_text())['snapshots'] if r['phase']=='return_side'][:8]
    (OUT/'protocol.json').write_text(json.dumps(dict(snapshot_ids=[r['id'] for r in entries],
        selection='First eight side snapshots in predeclared collection order, no outcome filter',
        modes=['deterministic','stochastic_0','stochastic_1','stochastic_2'],seed_rule='960000+100*index+replicate',
        horizon='Original episode limit unchanged; saved passes are historical, not new successes',
        training_updates=0,acceptance=False),indent=2))
    torch.set_num_threads(1)
    path=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip')
    before=hashlib.sha256(path.read_bytes()).hexdigest();m=RecurrentPPO.load(path,device='cuda');rows=[]
    with torch.inference_mode():
        for i,entry in enumerate(entries):
            payload=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
            assert hashlib.sha256(payload).hexdigest()==entry['sha256']
            for replicate in range(4):
                d=pickle.loads(payload);e=d['environment'];o=d['observation']
                state,start=rebuild_actor_memory(m,d['prefix_observations'])
                for x,y in zip(state,d['recurrent_state']):np.testing.assert_array_equal(x,y)
                torch.manual_seed(960000+100*i+replicate)
                passes=e.unwrapped.passes;side=False;front=False;new_pass=False
                observations=[];actions=[]
                while True:
                    observations.append(o.copy())
                    a,state=m.predict(o,state=state,episode_start=start,deterministic=replicate==0);start[:]=False;actions.append(a.copy())
                    o,_,t,tr,info=e.step(a)
                    side=side or info['lap_phase']=='return_front';front=front or info['lap_phase']=='align_front';new_pass=new_pass or info['passes']>passes
                    if t or tr:break
                observations.append(o.copy())
                np.savez_compressed(OUT/f'{i}_{replicate}.npz',observations=observations,actions=actions)
                rows.append(dict(id=entry['id'],replicate=replicate,mode='deterministic' if replicate==0 else 'stochastic',
                    initial_step=entry['step'],final_step=info['steps'],historical_passes=passes,new_passes=info['passes']-passes,
                    reached_return_front=side,reached_align_front=front,outcome=info['outcome'],
                    continuation_completed=info['outcome']=='timeout' and info['steps']==600 and info['passes']>=2))
                e.close()
            print('snapshots',i+1,flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    summary={name:dict(episodes=len([r for r in rows if r['mode']==name]),side=sum(r['reached_return_front'] for r in rows if r['mode']==name),
        front=sum(r['reached_align_front'] for r in rows if r['mode']==name),new_pass=sum(r['new_passes']>0 for r in rows if r['mode']==name),
        completed=sum(r['continuation_completed'] for r in rows if r['mode']==name),crashes=sum(r['outcome']!='timeout' for r in rows if r['mode']==name)) for name in ('deterministic','stochastic')}
    (OUT/'result.json').write_text(json.dumps(dict(rows=rows,summary=summary,source_unchanged=True,training_updates=0,mission_accepted=False),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
