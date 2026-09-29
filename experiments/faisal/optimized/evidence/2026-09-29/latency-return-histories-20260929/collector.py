"""Bounded authentic recurrent snapshots with prefix and continuation checks."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import sys,json,pickle,hashlib,shutil
import numpy as np
import torch
ROOT=Path(__file__).parent
OUT=ROOT/'latency-return-histories-20260929'
SOURCE=ROOT/'acquisition-return-mixture-20260928'
sys.path.insert(0,str(SOURCE/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.sim import HoopEnv
from src.params import Drone
from itertools import product
from src.return_reset_curriculum import wrap

def main():
    OUT.mkdir(exist_ok=False);(OUT/'snapshots').mkdir();(OUT/'flights').mkdir()
    shutil.copytree(SOURCE/'source_snapshot',OUT/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,OUT/'collector.py')
    path=SOURCE/'model_0.zip';before=hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=dict(source=str(path),sha256=before,seed_start=2990000,episodes=128,layouts=64,seed_end=2990063,delays=[0,2],split_rule="Every fourth layout heldout, shared across both delay conditions and all descendant snapshots",
        horizon=600,minimum_remaining=200,phases=['return_side','return_front','align_front'],
        provenance='Full uninterrupted current119D recurrent learned-policy flights; full wrappers,RNG,public history,recurrent state,observation/action prefix.',
        checks='Recurrent state reconstructed from authentic full prefix; pickle-restored next action and next observation/reward/termination match uninterrupted flight.',
        acceptance=False,training_updates=0,no_success_conditioned_collection=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(1);assert torch.cuda.is_available()
    model=RecurrentPPO.load(path,device='cuda');flights=[];snapshots=[]
    with torch.inference_mode():
        for seed,delay in product(range(2990000,2990064),(0,2)):
            env=wrap(HoopEnv(level=1.,drone=Drone(delay_steps=delay)),single=False,metric_ring=True,geometric_return=True)
            split="heldout" if (seed-2990000)%4==0 else "fit"
            obs,_=env.reset(seed=seed);p=RecurrentPredictor(model)
            prefix=[];actions=[];seen=set()
            while True:
                base=env.unwrapped;phase=base._lap_phase
                capture=base.passes>0 and phase in protocol['phases'] and phase not in seen and base.steps<=400
                if capture:
                    seen.add(phase);identifier=f'{seed}_delay{delay}_{base.steps}_{phase}'
                    payload=pickle.dumps(dict(environment=env,observation=obs.copy(),recurrent_state=deepcopy(p.state),episode_start=p.episode_start.copy(),
                        prefix_observations=np.asarray(prefix),prefix_actions=np.asarray(actions)),protocol=pickle.HIGHEST_PROTOCOL)
                    (OUT/'snapshots'/f'{identifier}.pkl').write_bytes(payload)
                    restored=pickle.loads(payload)
                    # Recompute recurrent state from the real input prefix, not simulator pose.
                    rebuilt=RecurrentPredictor(model)
                    for old_obs in prefix:rebuilt.predict(old_obs)
                    for expected,actual in zip(p.state,rebuilt.state):np.testing.assert_array_equal(expected,actual)
                    clone=RecurrentPredictor(model);clone.state=restored['recurrent_state'];clone.episode_start=restored['episode_start']
                    clone_action=clone.predict(restored['observation'])[0]
                    clone_result=restored['environment'].step(clone_action)
                prefix.append(obs.copy());a=p.predict(obs)[0];actions.append(a.copy())
                obs,reward,t,tr,info=env.step(a)
                if capture:
                    np.testing.assert_array_equal(a,clone_action)
                    np.testing.assert_array_equal(obs,clone_result[0])
                    assert (reward,t,tr)==clone_result[1:4]
                    restored['environment'].close()
                    snapshots.append(dict(id=identifier,seed=seed,delay=delay,split=split,phase=phase,step=len(actions)-1,sha256=hashlib.sha256(payload).hexdigest(),prefix_and_next_transition_exact=True,
                        front_seen=bool(restored['observation'][108]>.5),estimate_valid=bool(restored['observation'][118]>.5)))
                if t or tr:break
            np.savez_compressed(OUT/'flights'/f'{seed}_delay{delay}.npz',observations=np.asarray(prefix+[obs.copy()]),actions=np.asarray(actions))
            flights.append(dict(seed=seed,delay=delay,split=split,steps=info['steps'],passes=info['passes'],outcome=info['outcome'],full=info['outcome']=='timeout' and info['passes']>=2 and info['steps']==600))
            env.close()
            if len(flights)%8==0:print('flights',len(flights),'snapshots',len(snapshots),flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    result=dict(flights=flights,snapshots=snapshots,phase_counts=dict(Counter(r['phase'] for r in snapshots)),outcomes=dict(Counter(r['outcome'] for r in flights)),
        full=sum(r['full'] for r in flights),source_unchanged=True,training_updates=0,mission_accepted=False)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('flights','snapshots')}),flush=True)

if __name__=='__main__':main()
