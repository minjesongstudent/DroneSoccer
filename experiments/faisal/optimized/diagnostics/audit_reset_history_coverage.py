"""Compare fresh lesson inputs with genuine post-pass public histories."""
from pathlib import Path
import sys,json
import numpy as np
R=Path(__file__).parent
sys.path.insert(0,str(R/'lap-recurrent-anchor10-20260928/source_snapshot'))
from src.return_reset_lessons import ReturnResetLesson,LEVELS
from src.return_reset_curriculum import wrap

def flags(obs):
    return dict(visible=bool(obs[70]>.5),front_seen=bool(obs[108]>.5),estimate_valid=bool(obs[118]>.5),
        command_history_nonzero=bool(np.any(np.abs(obs[72:84])>1e-6)))
rows=[]
for level,name in enumerate(LEVELS):
    e=wrap(ReturnResetLesson(level),metric_ring=True)
    for seed in range(25):
        o,_=e.reset(seed=seed)
        rows.append(dict(level=level,name=name,seed=seed,**flags(o)))
    e.close()
actual=[]
for path in sorted((R/'history-guard-rl-20260928/first_divergence').glob('original_*.json')):
    trace=json.loads(path.read_text())['trace']
    row=next(r for r in trace if r['passes']>=1)
    actual.append(dict(seed=int(path.stem.split('_')[1]),step=row['step'],**flags(row['observation'])))
summary=[]
for level,name in enumerate(LEVELS):
    selected=[r for r in rows if r['level']==level]
    summary.append(dict(name=name,episodes=len(selected),**{k:sum(r[k] for r in selected) for k in ('visible','front_seen','estimate_valid','command_history_nonzero')}))
report=dict(fresh_lessons=summary,genuine_first_pass=dict(episodes=len(actual),**{k:sum(r[k] for r in actual) for k in ('visible','front_seen','estimate_valid','command_history_nonzero')}),
    rows=rows,actual=actual,training_updates=0,
    caveat='Initial-input coverage only, not lesson trajectory outcomes or proof that mismatch causes failures. Genuine examples are selected FIT successful source flights.')
(R/'reset_history_coverage.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('rows','actual')},indent=2))
