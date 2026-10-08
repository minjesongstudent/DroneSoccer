"""Summarize saved learned trajectories; no new episodes or policy changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    summaries={}
    for log in sorted(args.run.glob('*/*/episodes.jsonl')):
        rows=[json.loads(line) for line in log.read_text().splitlines()]
        failures=[]
        for r in rows:
            if not r['crashed']: continue
            with np.load(log.parent/f"{r['seed']}.npz") as data:
                obs=data['actor_observations']
                current=obs[:,-21:-13]
                normal=data['hoop_normal'];relative=data['positions'][-1]-data['hoop_centre']
                seen=np.where(current[:,6]>0)[0]
                failures.append(dict(seed=r['seed'],outcome=r['outcome'],
                    stage='after_first_clearance' if r['full_body_passes'] else 'before_first_clearance',
                    started_blind=not bool(current[0,6]),
                    blind_actions_before_crash=int(len(obs)-1-seen[-1]) if len(seen) else len(obs),
                    along_m=float(relative@normal),
                    lateral_m=float(relative@np.array([-normal[1],normal[0],0])),
                    vertical_m=float(relative[2]),steps=r['steps']))
        summary=dict(episodes=len(rows),crashes=len(failures),
            crash_stages=dict(Counter(f['stage'] for f in failures)),
            crash_types=dict(Counter(f['outcome'] for f in failures)),
            started_blind_crashes=sum(f['started_blind'] for f in failures),
            started_blind_episodes=0,failures=failures)
        for r in rows:
            with np.load(log.parent/f"{r['seed']}.npz") as data:
                summary['started_blind_episodes']+=int(not data['actor_observations'][0,-15])
        summaries[str(log.parent.relative_to(args.run))]=summary
    (args.run/'failure_analysis.json').write_text(json.dumps(summaries,indent=2))
    print(json.dumps({k:{f:v for f,v in s.items() if f!='failures'} for k,s in summaries.items()},indent=2))


if __name__=='__main__':main()
