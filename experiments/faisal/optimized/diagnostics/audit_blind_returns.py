"""Compare post-pass camera visibility on FIT teacher and saved student traces."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).parent
BANK=Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3')
seeds=json.loads(Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/protocol.json').read_text())['fit_lap_seeds']
records={r['seed']:r for r in json.loads((BANK/'records.json').read_text()) if r['seed'] in seeds}
students=json.loads((ROOT/'fit_trajectories.json').read_text())

def describe(visibility):
    visibility=np.asarray(visibility)>0.5
    longest=run=0
    for visible in visibility:
        run=0 if visible else run+1
        longest=max(longest,run)
    first=next((i for i,v in enumerate(visibility) if v),None)
    return dict(samples=len(visibility),visible_fraction=float(visibility.mean()) if len(visibility) else None,
                first_visible_offset_steps=first,longest_blind_steps=longest)

rows=[]
for seed in seeds:
    assert seed%4!=3
    record=records[seed]
    passes=[e['step'] for e in record['events'] if e['event']=='full_pass_credit']
    start,end=passes[:2]
    with np.load(BANK/'trajectories'/f'{seed}.npz') as data:
        obs=data['observations'];actions=data['actions']
        rows.append(dict(kind='teacher',seed=seed,first_pass=start,second_pass=end,
            interval='first full-pass state through state before second pass',
            visibility=describe(obs[start:end,70]),
            initial_30_postpass_mean_action=actions[start:min(start+30,end)].mean(0).tolist(),
            duration_to_second_pass=end-start))
    for checkpoint in (300,500):
        trace=next(t['trace'] for t in students if t['checkpoint']==checkpoint and t['seed']==seed)
        post=[r for r in trace if r['passes']>=1]
        rows.append(dict(kind=f'student{checkpoint}',seed=seed,first_pass=post[0]['step'] if post else None,
            end_outcome=trace[-1]['outcome'],passes=trace[-1]['passes'],
            interval='first full-pass state through termination; unequal horizons, descriptive only',
            visibility=describe([r['visible'] for r in post]),
            initial_30_postpass_mean_action=np.mean([r['action'] for r in post[1:31]],axis=0).tolist() if post else None))
summary={}
for kind in ('teacher','student300','student500'):
    subset=[r for r in rows if r['kind']==kind and r['visibility']['samples']]
    offsets=[r['visibility']['first_visible_offset_steps'] for r in subset]
    summary[kind]=dict(episodes_with_first_pass=len(subset),never_reacquired=sum(x is None for x in offsets),
        reacquisition_steps=[x for x in offsets if x is not None],
        longest_blind_steps=[r['visibility']['longest_blind_steps'] for r in subset])
(ROOT/'blind_returns_audit.json').write_text(json.dumps(dict(fit_only=True,rows=rows,summary=summary),indent=2))
print(json.dumps(summary,indent=2))
