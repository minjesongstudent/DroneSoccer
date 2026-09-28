import sys,json,pickle,hashlib
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np,torch
from src.standard_source_policy import StandardSourceSAC
from src.success_history import SuccessHistoryLibrary,SuccessHistoryPractice
from src.return_evaluation import _fingerprint
from src.sim import CRASHES

torch.set_num_threads(1)
bank=SuccessHistoryLibrary('D:/DroneSoccerData/diagnostics/success-return-bank-v1')
buffer=pickle.loads((ROOT/'replay.pkl').read_bytes())
assert buffer.pos==15000 and not buffer.full
starts={}
for row in bank.records:
    if row['landmark']=='near_pass':
        starts[bank.restore(row['id'])['observation'].tobytes()]=row
replay=[]
for eid in np.unique(buffer.episode_ids[:buffer.pos]):
    ids=np.flatnonzero(buffer.episode_ids[:buffer.pos]==eid)
    row=starts.get(buffer.observations[ids[0],0].tobytes())
    if row is None:continue
    env=SuccessHistoryPractice(bank);obs,info=env.reset(options={'snapshot_id':row['id']})
    passed=[]
    for i in ids:
        assert np.array_equal(obs,buffer.observations[i,0])
        prior=info['passes']
        obs,reward,t,tr,info=env.step(buffer.actions[i,0])
        assert np.array_equal(obs,buffer.next_observations[i,0])
        assert np.isclose(reward,buffer.rewards[i,0],atol=1e-5)
        assert bool(t or tr)==bool(buffer.dones[i,0])
        if info['passes']>prior:passed.append(int(i))
    replay.append(dict(episode=int(eid),seed=row['seed'],start=int(ids[0]),end=int(ids[-1]),
        complete=bool(t or tr),passes=info['passes'],outcome=info['outcome'],
        success=info['practice_success'],pass_indices=passed,
        steps_after_last_pass=int(ids[-1]-passed[-1]) if passed else None))
    env.close()
model=StandardSourceSAC.load(ROOT/'model_15000.zip',device='cuda');before=_fingerprint(model)
alpha=float(model.ent_coef_tensor.item());records=[];arrays={}
for row in bank.records:
    if row['landmark']!='near_pass':continue
    for rep in range(8):
        torch.manual_seed(3300000+row['seed']*8+rep)
        env=SuccessHistoryPractice(bank);obs,info=env.reset(options={'snapshot_id':row['id']})
        rewards=[];logs=[];actions=[];positions=[env.unwrapped.pos.copy()];observations=[]
        with torch.inference_mode():
            while True:
                x=torch.as_tensor(obs,device='cuda').unsqueeze(0)
                a,lp=model.actor.action_log_prob(x)
                if not rewards:q=float(torch.cat(model.critic(x,a),dim=1).min().item())
                action=a[0].cpu().numpy();observations.append(obs.copy());actions.append(action.copy())
                obs,r,t,tr,info=env.step(action)
                rewards.append(r);logs.append(float(lp.item()));positions.append(env.unwrapped.pos.copy())
                if t or tr:break
        soft=np.asarray(rewards,dtype=float)
        soft[1:]-=alpha*np.asarray(logs[1:])
        actual=float(np.sum(soft*.995**np.arange(len(soft))))
        terminal=bool(t and info['outcome'] in CRASHES)
        index=len(records)
        arrays[f'{index}_obs']=np.asarray(observations);arrays[f'{index}_actions']=np.asarray(actions)
        arrays[f'{index}_positions']=np.asarray(positions);arrays[f'{index}_rewards']=np.asarray(rewards);arrays[f'{index}_log_probs']=np.asarray(logs)
        records.append(dict(seed=row['seed'],replicate=rep,elapsed=len(rewards),outcome=info['outcome'],
            new_passes=info['passes'],success=info['practice_success'],initial_min_q=q,
            finite_soft_return=actual,true_terminal=terminal,terminal_error=q-actual if terminal else None))
        env.close()
assert before==_fingerprint(model)
summary=[]
for low,high in ((0,10000),(10000,15000)):
    rows=[r for r in replay if low<=r['start']<high]
    summary.append(dict(start=low,end=high,authentic_episodes=len(rows),complete=sum(r['complete'] for r in rows),
        episodes_with_new_pass=sum(r['passes']>0 for r in rows),surviving_success=sum(r['success'] for r in rows),
        crashes_after_new_pass=sum(r['passes']>0 and r['outcome'] in CRASHES for r in rows)))
errors=[r['terminal_error'] for r in records if r['true_terminal']]
result=dict(replay_verified=True,replay_summary=summary,replay_records=replay,on_policy=records,
    terminal_count=len(errors),terminal_bias=float(np.mean(errors)),terminal_mae=float(np.mean(np.abs(errors))),
    caveat='Terminal-selected stochastic trajectories; horizon timeouts excluded. Conditional calibration, not unconditional Q accuracy or causal attribution.',
    source_frozen=True,reserved_layout_rollouts=0)
(ROOT/'replay_audit.json').write_text(json.dumps(result,indent=2));np.savez_compressed(ROOT/'replay_audit_trajectories.npz',**arrays)
print(json.dumps({k:v for k,v in result.items() if k not in ('replay_records','on_policy')}),flush=True)
