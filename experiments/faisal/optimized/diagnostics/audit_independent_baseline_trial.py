"""Summarize saved trial outcomes and scalar baseline weights, no simulation."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.independent_return_baseline import leave_one_out_baselines
RUN=ROOT/'diagnostics/independent-baseline-actor-20260929'
entries=json.loads((RUN/'protocol.json').read_text())['snapshots']
fixed=[]
for update in range(3):
    report=json.loads((RUN/f'evaluation_{update}.json').read_text())
    fixed.append(dict(update=update,suites={k:{f:s[f] for f in ('successful_episodes','crash_rate','total_passes','episodes_with_two_passes','returned_to_front_episodes')} for k,s in report['suites'].items()}))
batches=[]
for batch in range(3):
    update=json.loads((RUN/f'update_{batch+1}.json').read_text())
    returns=[]
    for i in range(len(entries)):
        with np.load(RUN/f'rollout_{batch}_{i}.npz') as a:returns.append(a['returns'].copy())
    baselines=leave_one_out_baselines(returns,[e['group'] for e in entries])
    groups={}
    for i,entry in enumerate(entries):
        category=('early_return' if entry['group']<8 else 'reapproach') if 'natural' not in entry else entry['natural']
        g=groups.setdefault(category,dict(episodes=0,crashes=0,new_pass_episodes=0,weights=[],raw=[]))
        row=update['rows'][i];info=row['final_info']
        g['episodes']+=1;g['crashes']+=int(info['outcome'] in ('hit','left_cage'))
        g['new_pass_episodes']+=int(info['passes']>row['initial_passes'])
        g['weights'].extend(((returns[i]-baselines[i])*(.999**np.arange(len(returns[i])))/40).tolist())
        g['raw'].extend((returns[i]*(.999**np.arange(len(returns[i])))/40).tolist())
    for g in groups.values():
        weights=np.array(g.pop('weights'));raw=np.array(g.pop('raw'))
        g.update(weight_rms=float(np.sqrt(np.mean(weights**2))),raw_weight_rms=float(np.sqrt(np.mean(raw**2))),weight_std=float(weights.std()))
    batches.append(dict(batch=batch,guard=update['guard'],preupdate_log_error=update['preupdate_log_error'],groups=groups))
result=dict(fixed=fixed,batches=batches,caution='Scalar weight RMS is not gradient variance; starts and policies differ between batches')
(RUN/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
