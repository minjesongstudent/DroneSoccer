import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np,torch
from src.standard_source_policy import StandardSourceSAC
from src.success_history import SuccessHistoryLibrary,SuccessHistoryPractice
from src.return_evaluation import _fingerprint

torch.set_num_threads(1)
bank=SuccessHistoryLibrary('D:/DroneSoccerData/diagnostics/success-return-bank-v1')
records=[];arrays={};counters={}
for step in (0,15000):
    model=StandardSourceSAC.load(ROOT/f'model_{step}.zip',device='cuda');before=_fingerprint(model)
    counters[str(step)]={k:v for k,v in vars(model).items() if isinstance(v,(int,float,str)) and any(s in k for s in ('update','reject','accept','calibr'))}
    env=SuccessHistoryPractice(bank)
    for row in bank.records:
        if row['landmark']!='near_pass':continue
        obs,info=env.reset(options={'snapshot_id':row['id']});base=env.unwrapped
        positions=[base.pos.copy()];observations=[];actions=[];rewards=[];events=[]
        with torch.inference_mode():
            while True:
                old=base.pos.copy();passes=base.passes
                observations.append(obs.copy());action=model.predict(obs,deterministic=True)[0];actions.append(action)
                obs,r,t,tr,info=env.step(action);positions.append(base.pos.copy());rewards.append(r)
                crossing=base._crossing(old)
                if crossing is not None:events.append(dict(local_step=info['steps'],crossing=[int(crossing[0]),bool(crossing[1])]))
                if base.passes>passes:events.append(dict(local_step=info['steps'],new_pass=True))
                if t or tr:break
        key=f'{step}_{row["seed"]}';o=np.asarray(observations);a=np.asarray(actions);p=np.asarray(positions)
        arrays[key+'_obs']=o;arrays[key+'_actions']=a;arrays[key+'_positions']=p;arrays[key+'_rewards']=np.asarray(rewards)
        records.append(dict(checkpoint=step,seed=row['seed'],outcome=info['outcome'],elapsed=info['steps'],new_passes=info['passes'],success=info['practice_success'],
            path_metres=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()),net_metres=float(np.linalg.norm(p[-1]-p[0])),
            visible_fraction=float(np.mean(o[:,70]>.5)),mean_action=a.mean(axis=0).tolist(),events=events))
    env.close();assert before==_fingerprint(model)
np.savez_compressed(ROOT/'diagnostic_trajectories.npz',**arrays)
(ROOT/'diagnostic.json').write_text(json.dumps(dict(records=records,counters=counters,models_unchanged=True,reserved_layout_rollouts=0),indent=2))
print(json.dumps(records),flush=True)
