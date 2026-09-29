"""One deterministic and three stochastic continuations per FIT snapshot."""
from pathlib import Path
import sys,json,pickle,hashlib,shutil
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];BANK=ROOT/'diagnostics/latency-return-histories-20260929'
OUT=BANK/'continuation_audit';sys.path.insert(0,str(BANK/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory

def main():
    OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/'runner.py');torch.set_num_threads(1)
    bank=json.loads((BANK/'result.json').read_text());entries=[e for e in bank['snapshots'] if e['split']=='fit']
    assert len(entries)==30
    source=Path(json.loads((BANK/'protocol.json').read_text())['source']);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    model=RecurrentPPO.load(source,device='cuda');model.policy.set_training_mode(False)
    (OUT/'protocol.json').write_text(json.dumps(dict(entries=entries,repeats=4,repeat0='deterministic',others='stochastic learned actions',
        torch_seed_rule='996000+100*entry_index+repeat',source_sha256=digest,training_updates=0,fit_only=True),indent=2))
    rows=[]
    with torch.no_grad():
        for i,e in enumerate(entries):
            payload=(BANK/'snapshots'/f"{e['id']}.pkl").read_bytes();assert hashlib.sha256(payload).hexdigest()==e['sha256']
            for repeat in range(4):
                d=pickle.loads(payload);env=d['environment'];obs=d['observation'];initial=int(env.unwrapped.passes)
                state,start=_initial_memory(model,d['prefix_observations'],False)
                torch.manual_seed(996000+100*i+repeat);observations=[];actions=[];rewards=[];phases=[]
                while True:
                    observations.append(obs.copy());phases.append(env.unwrapped._lap_phase)
                    tensor,_=model.policy.obs_to_tensor(obs)
                    action,_,_,state=model.policy.forward(tensor,state,start,deterministic=repeat==0)
                    start=torch.zeros_like(start);raw=action[0].cpu().numpy();actions.append(raw)
                    obs,reward,t,tr,info=env.step(np.clip(raw,env.action_space.low,env.action_space.high));rewards.append(reward)
                    if t or tr:break
                if repeat==0:
                    expected=next(f for f in bank['flights'] if f['seed']==e['seed'] and f['delay']==e['delay'])
                    assert all(info[k]==expected[k] for k in ('outcome','passes','steps'))
                np.savez_compressed(OUT/f'entry_{i}_repeat_{repeat}.npz',prefix=d['prefix_observations'],observations=np.asarray(observations),raw_actions=np.asarray(actions),rewards=np.asarray(rewards),phases=np.asarray(phases))
                rows.append(dict(index=i,id=e['id'],seed=e['seed'],delay=e['delay'],start_phase=e['phase'],repeat=repeat,
                    outcome=info['outcome'],steps=info['steps'],initial_passes=initial,passes=int(info['passes']),new_passes=int(info['passes'])-initial,
                    front_reached=any(p in ('return_front','align_front') for p in phases),align_reached='align_front' in phases,
                    full=info['outcome']=='timeout' and info['passes']>=2))
                env.close()
            (OUT/'progress.json').write_text(json.dumps(rows,indent=2))
            if (i+1)%5==0:print('snapshots completed',i+1,flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    summaries=[]
    for delay in (0,2):
        for mode in ('deterministic','stochastic'):
            selected=[r for r in rows if r['delay']==delay and (r['repeat']==0)==(mode=='deterministic')]
            summaries.append(dict(delay=delay,mode=mode,episodes=len(selected),front=sum(r['front_reached'] for r in selected),new_pass_episodes=sum(r['new_passes']>0 for r in selected),crashes=sum(r['outcome'] in ('hit','left_cage') for r in selected),full=sum(r['full'] for r in selected)))
    (OUT/'result.json').write_text(json.dumps(dict(rows=rows,summary=summaries,source_unchanged=True,deterministic_endpoints_exact=True),indent=2));print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
