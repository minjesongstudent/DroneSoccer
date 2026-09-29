"""Counterfactual discount audit on unchanged recorded rewards, no training."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'

def main():
    paths=json.loads((RUN/'trajectory_audit/result.json').read_text())['rows']
    rows=[]
    for row in paths:
        rewards=np.load(RUN/f"rollout_{row['update']}_{row['snapshot']}.npz")['rewards']
        returns={str(g):float(np.dot(rewards,g**np.arange(len(rewards)))) for g in (.99,.995,.999,1.)}
        terminal_weight={str(g):float(g**(len(rewards)-1)) for g in (.99,.995,.999)}
        rows.append(dict(update=row['update'],snapshot=row['snapshot'],outcome=row['outcome'],
            new_passes=row['new_passes'],steps=len(rewards),returns=returns,terminal_weight=terminal_weight))
    summary={}
    for gamma in ('0.99','0.995','0.999','1.0'):
        ordered=sorted(rows,key=lambda r:r['returns'][gamma],reverse=True)
        survivor=next(r for r in rows if r['outcome']=='timeout')
        summary[gamma]=dict(survivor_rank=ordered.index(survivor)+1,
            survivor_return=survivor['returns'][gamma],
            top_five=[dict(update=r['update'],snapshot=r['snapshot'],outcome=r['outcome'],new_passes=r['new_passes'],value=r['returns'][gamma]) for r in ordered[:5]])
    result=dict(note='Different snapshots and sampled paths: descriptive reward audit, not causal policy comparison',
        weights_at_steps={str(g):{str(t):float(g**t) for t in (100,300,600)} for g in (.99,.995,.999)},
        summary=summary,rows=rows)
    (RUN/'credit_horizon_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
