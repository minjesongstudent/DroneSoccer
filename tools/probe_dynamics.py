"""Paired dynamics sensitivity, not validation or altered mission acceptance."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from src.policy_runtime import LearnedPolicy, SPECS
from tools.validate_flight_policies import ROOT, episode
from tools.run_clearance_experiment import sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    source=ROOT/'trainedModels/seenflag-s1-level1.zip'
    profile=SPECS['seenflag-s1-level1'];assert sha(source)==profile.sha256
    hashes={}
    for f in list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tools/validate_flight_policies.py',source]:
        rel=f.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(f,dest);hashes[str(rel)]=sha(dest)
    variants={'baseline':None,'fixed_delay':{'delay_steps':2},'fixed_tau':{'tau':.55},
              'fixed_both':{'delay_steps':2,'tau':.55}}
    write_json(out/'protocol.json',dict(kind='dynamics sensitivity only; not acceptance',variants=variants,
        suite='randomised',seeds=[9530000,9530099],episodes=100,source_hashes=hashes,
        training=False,actor_truth_inputs=False,promotion=False,
        interpretation='Same reset draws/layouts; override only named dynamics after reset. Do not reduce training/validation ranges.'))
    policy=LearnedPolicy(source,profile);results={};baseline=None
    for name,override in variants.items():
        folder=out/name;folder.mkdir();rows=[]
        for seed in range(9530000,9530100):
            row=episode(policy,'randomised',seed,folder/f'{seed}.npz',diagnostic_dynamics=override)
            rows.append(row)
            with (folder/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if name=='baseline':
                previous=Path('D:/DroneSoccerData/runs/boundary-fresh-development-20261007/original/randomised')/f'{seed}.npz'
                with np.load(previous) as a,np.load(folder/f'{seed}.npz') as b:
                    for key in a.files:assert np.array_equal(a[key],b[key]),(seed,key)
            if (seed+1)%25==0:print(name,seed+1,flush=True)
        if baseline is None:baseline=rows
        results[name]=dict(first=sum(r['clean_first_pass'] for r in rows),crashes=sum(r['crashed'] for r in rows),
            returned=sum(r['returns_to_front']>0 for r in rows),repeat=sum(r['clean_repeated_episode'] for r in rows),
            gained_repeat=sum(b['clean_repeated_episode'] and not a['clean_repeated_episode'] for a,b in zip(baseline,rows)),
            lost_repeat=sum(a['clean_repeated_episode'] and not b['clean_repeated_episode'] for a,b in zip(baseline,rows)))
        write_json(out/'results.json',results);print(name,results[name],flush=True)
    (out/'DONE').write_text('400 diagnostic learned-policy episodes completed. No acceptance claim.\n')


if __name__=='__main__':main()
