"""Summarize a completed paired critic diagnostic without fitting to its outcomes."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tools.run_clearance_experiment import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();root=args.run
    assert (root/'DONE').exists()
    states=json.loads((root/'results.json').read_text())
    rows=[json.loads(line) for line in (root/'continuations.jsonl').read_text().splitlines()]
    paired=[]
    for state in states:
        q=np.array([a['q'] for a in state['actions']]);high=int(q.argmax());low=int(q.argmin())
        selected=[r for r in rows if r['state']==state['state']]
        by_action={a:sorted([r for r in selected if r['candidate']==a],key=lambda r:r['replica']) for a in [high,low]}
        comparisons={}
        for field in ['finite_soft_return','bootstrapped_soft_return','mission']:
            delta=np.array([float(a[field])-float(b[field]) for a,b in zip(by_action[high],by_action[low])])
            comparisons[field]=dict(mean=float(delta.mean()),paired_standard_error=float(delta.std(ddof=1)/np.sqrt(len(delta))))
        paired.append(dict(state=state['state'],q_gap=float(q[high]-q[low]),high=high,low=low,**comparisons))
    q=[];observed=[]
    for state in states:
        for action in state['actions']:
            q.append(action['q']);observed.append(action['bootstrapped_mean'])
    error=np.array(q)-np.array(observed)
    summary=dict(states=len(states),continuations=len(rows),
        truncation_fraction=float(np.mean([r['truncated'] for r in rows])),
        critic_minus_sample_mean_bias=float(error.mean()),critic_vs_sample_mean_rmse=float(np.sqrt(np.mean(error**2))),
        paired=paired,top_q_beats_bottom_q_finite=sum(r['finite_soft_return']['mean']>0 for r in paired),
        top_q_beats_bottom_q_bootstrapped=sum(r['bootstrapped_soft_return']['mean']>0 for r in paired),
        caveat='Paired continuations on selected surviving states. Sample means are noisy; bootstrapped return partly depends on critic. No general performance or acceptance claim.',
        input_hashes={name:sha(root/name) for name in ['protocol.json','results.json','continuations.jsonl']})
    write_json(root/'ranking_summary.json',summary);shutil.copy2(__file__,root/'summary_source.py')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
