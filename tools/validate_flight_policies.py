"""Bounded development comparison, never a physical-flight approval or mission acceptance.

python -m tools.validate_flight_policies --out <new-directory> --episodes 100
All motion comes from unchanged learned policies. Simulator truth is used only
to reset scenarios and score full-body passage, contacts and return-to-front.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from src import contract
from src.pass_audit import PassageAudit
from src.policy_runtime import LearnedPolicy, SPECS
from src.sim import HoopEnv

ROOT = Path(__file__).resolve().parents[1]
SUITES = {'aligned': 9410000, 'nominal': 9420000, 'randomised': 9430000}


def start(env, suite, seed):
    public, _ = env.reset(seed=seed)
    if suite == 'aligned':
        rng = np.random.default_rng(seed)
        centre = np.array([1.025,1.025,rng.uniform(.8,1.2)])
        position = centre + [-rng.uniform(.6,.8),rng.uniform(-.08,.08),rng.uniform(-.04,.04)]
        public = env.place(position,rng.uniform(-np.pi/18,np.pi/18),centre,[1,0,0])
    return public


def episode(policy, suite, seed, path, *, diagnostic_dynamics=None):
    env = HoopEnv(level=1,randomise=suite=='randomised',hanger=True)
    public = start(env,suite,seed)
    if diagnostic_dynamics is not None:
        # Explicit sensitivity probe only. Normal development/acceptance callers
        # never use this hook; observation and random reset draws stay unchanged.
        from collections import deque
        from dataclasses import replace
        if set(diagnostic_dynamics)-{'delay_steps','tau'}:
            raise ValueError('Only diagnostic delay/tau overrides allowed')
        if not 1 <= diagnostic_dynamics.get('delay_steps',env.dyn.delay_steps) <= 4:
            raise ValueError('Delay outside existing randomisation range')
        if not .25 <= diagnostic_dynamics.get('tau',env.dyn.tau) <= .7:
            raise ValueError('Tau outside existing randomisation range')
        env.dyn = replace(env.dyn, **diagnostic_dynamics)
        env._pending = deque([np.zeros(4,np.float32) for _ in range(env.dyn.delay_steps)])
    policy.reset()
    previous = np.zeros(4,np.float32)
    audit = PassageAudit(); audit.update(env)
    trace = {'actor_observations':[], 'actions':[], 'positions':[], 'headings':[]}
    before_crossing=False
    while True:
        # Only the newest eight public camera features enter the profile history.
        end = contract.OBS_STACK*contract.OBS_DIM
        obs,action = policy.predict(public[end-contract.OBS_DIM:end],previous)
        trace['actor_observations'].append(obs.copy()); trace['actions'].append(action.copy())
        previous=action
        public,_,terminated,truncated,info=env.step(action)
        trace['positions'].append(env.pos.copy()); trace['headings'].append(env.heading)
        unsafe=terminated or info['wall_steps']>0
        audit.update(env,info['outcome'],unsafe)
        if not audit.count and unsafe: before_crossing=True
        if terminated or truncated: break
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in trace.items()},
        hoop_centre=env.hoop_centre,hoop_normal=env.hoop_normal,drone_half_box=env._half_box)
    return dict(seed=seed,steps=env.steps,outcome=info['outcome'],wall_steps=env.wall_steps,
        centre_passes=env.passes,full_body_passes=audit.count,full_body_steps=audit.steps,
        returns_to_front=audit.returns,crashed=bool(terminated),
        clean_first_pass=bool(audit.count and not before_crossing),
        clean_repeated_episode=bool(audit.count>=2 and not terminated and not env.wall_steps))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.episodes<=200: parser.error('Use 1..200 episodes per suite')
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    protocol=dict(kind='development; not reserved acceptance',episodes=args.episodes,suites=SUITES,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        profiles={name:asdict(spec) for name,spec in SPECS.items()},
        max_steps=contract.LAPS_MAX_STEPS,pass_metric='Full body past far noodle face; collision-free transit',
        actor_inputs='Public camera features and actually issued learned actions only',
        flight_release=False,training=False)
    # Freeze all sources and models before drawing the first evaluation episode.
    files=list((ROOT/'src').glob('*.py'))+[Path(__file__).resolve()]
    files += [ROOT/'trainedModels'/(name+'.zip') for name in SPECS]
    hashes={}
    for f in files:
        rel=f.relative_to(ROOT);dest=args.out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest);hashes[str(rel)]=hashlib.sha256(f.read_bytes()).hexdigest()
    protocol['source_sha256']=hashes
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results={}
    for name,spec in SPECS.items():
        policy=LearnedPolicy(ROOT/'trainedModels'/(name+'.zip'),spec)
        for suite,base in SUITES.items():
            folder=args.out/name/suite;folder.mkdir(parents=True)
            rows=[]
            for i in range(args.episodes):
                r=episode(policy,suite,base+i,folder/f'{base+i}.npz');rows.append(r)
                with (folder/'episodes.jsonl').open('a') as log: log.write(json.dumps(r)+'\n')
                if (i+1)%25==0: print(name,suite,i+1,flush=True)
            summary=dict(episodes=len(rows),clean_first_pass=sum(r['clean_first_pass'] for r in rows),
                crashes=sum(r['crashed'] for r in rows),
                mean_full_body_passes=float(np.mean([r['full_body_passes'] for r in rows])),
                returned_episodes=sum(r['returns_to_front']>0 for r in rows),
                clean_repeated_episodes=sum(r['clean_repeated_episode'] for r in rows),
                endings=dict(Counter(r['outcome'] for r in rows)))
            results[name+'/'+suite]=summary
            (args.out/'results.json').write_text(json.dumps(results,indent=2))
            print(name,suite,json.dumps(summary),flush=True)


if __name__=='__main__': main()
