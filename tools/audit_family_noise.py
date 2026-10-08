"""Measure repeated-continuation noise with whole-state fitting/holdout separation."""
import argparse
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
from tools.run_clearance_experiment import sha,write_json


def summarize(values):
    assert values.ndim==3 and values.shape[1:]==(4,6)
    left=values[:,:,:3].mean(2);right=values[:,:,3:].mean(2)
    contrasts=np.stack([values[:,a]-values[:,b] for a,b in itertools.combinations(range(4),2)],axis=1)
    delta=contrasts.mean(2);se=contrasts.std(2,ddof=1)/np.sqrt(6)
    dl=contrasts[:,:,:3].mean(2);dr=contrasts[:,:,3:].mean(2)
    informative=(dl!=0)&(dr!=0)
    unique_left=(left==left.max(1,keepdims=True)).sum(1)==1
    unique_right=(right==right.max(1,keepdims=True)).sum(1)==1
    unique=unique_left&unique_right
    return dict(states=len(values),state_action_pairs=len(values)*4,
        median_within_action_std=float(np.median(values.std(2,ddof=1))),
        mean_sampling_variance=float(np.mean(values.var(2,ddof=1)/6)),
        mean_return_variance_between_state_actions=float(values.mean(2).var()),
        replica_half_value_rmse=float(np.sqrt(np.mean((left-right)**2))),
        informative_pair_contrasts=int(informative.sum()),
        half_contrast_sign_agreement=float(np.mean(np.sign(dl[informative])==np.sign(dr[informative]))) if informative.any() else None,
        fraction_pair_deltas_above_two_standard_errors=float(np.mean(np.abs(delta)>2*se)),
        median_abs_pair_delta=float(np.median(np.abs(delta))),median_pair_standard_error=float(np.median(se)),
        unique_best_both_halves_states=int(unique.sum()),
        half_best_action_agreement=float(np.mean(left[unique].argmax(1)==right[unique].argmax(1))) if unique.any() else None)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    assert (args.run/'DONE').exists()
    rows=[json.loads(line) for line in (args.run/'continuations.jsonl').read_text().splitlines()]
    assert len(rows)==1536
    keys=[(r['state'],r['candidate'],r['replica']) for r in rows]
    assert len(set(keys))==len(keys)
    assert set(keys)==set(itertools.product(range(64),range(4),range(6)))
    rows.sort(key=lambda r:(r['state'],r['candidate'],r['replica']))
    fit=np.flatnonzero(np.arange(64)%5!=4);held=np.flatnonzero(np.arange(64)%5==4)
    args.out.mkdir(parents=True,exist_ok=False)
    arrays={name:np.array([float(r[name]) for r in rows]).reshape(64,4,6)
            for name in ['finite_soft_return','bootstrapped_soft_return','mission']}
    np.savez_compressed(args.out/'outcomes_split.npz',**arrays,fit_states=fit,heldout_states=held)
    result=dict(input_hashes={str(args.run/name):sha(args.run/name) for name in ['protocol.json','continuations.jsonl']},
        fields={name:{'fit':summarize(v[fit]),'heldout':summarize(v[held])} for name,v in arrays.items()},
        caveat='Within-family samples and six contrasts are dependent. Two-standard-error fraction is descriptive, not an adjusted significance test. No policy update or acceptance.')
    write_json(args.out/'report.json',result);shutil.copy2(__file__,args.out/'audit_source.py')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
