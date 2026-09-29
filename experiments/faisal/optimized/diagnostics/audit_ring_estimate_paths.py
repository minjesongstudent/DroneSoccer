"""Public estimate accuracy against truth, diagnostic only; exact action replay."""
from pathlib import Path
import sys,json,pickle,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'
BANK=ROOT/'diagnostics/recurrent-return-histories-20260928';OUT=RUN/'ring_estimate_audit'
sys.path.insert(0,str(RUN/'source_snapshot'))

def stats(rows):
    if not rows:return dict(frames=0)
    e=np.array([r['error'] for r in rows]);return dict(frames=len(rows),median_m=float(np.median(e)),
        p95_m=float(np.quantile(e,.95)),max_m=float(e.max()),over_10cm=float(np.mean(e>.1)),over_25cm=float(np.mean(e>.25)))

def main():
    OUT.mkdir(exist_ok=False);entries=json.loads((RUN/'protocol.json').read_text())['snapshots'];rows=[];ends=[]
    for update in range(4):
        for i,entry in enumerate(entries):
            blob=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes();assert hashlib.sha256(blob).hexdigest()==entry['sha256']
            d=pickle.loads(blob);env=d['environment'];base=env.unwrapped;o=d['observation']
            a=np.load(RUN/f'rollout_{update}_{i}.npz');flight=[]
            for j,raw in enumerate(a['raw_actions']):
                np.testing.assert_array_equal(o,a['observations'][j])
                rel=base.hoop_centre-base.pos;c,s=np.cos(base.heading),np.sin(base.heading)
                truth=np.array([c*rel[0]+s*rel[1],-s*rel[0]+c*rel[1],rel[2]])
                row=dict(update=update,snapshot=i,step=base.steps,visible=bool(o[70]>.5),valid=bool(o[118]>.5),
                    error=float(np.linalg.norm(o[115:118]*3-truth)),estimate=(o[115:118]*3).tolist(),truth=truth.tolist(),
                    front_age=float(o[109]),phase=base._lap_phase)
                flight.append(row);rows.append(row)
                o,r,t,tr,info=env.step(np.clip(raw,env.action_space.low,env.action_space.high))
            np.testing.assert_array_equal(o,a['next_observation']);ends.append(dict(outcome=info['outcome'],**flight[-1]))
            (OUT/f'{update}_{i}.json').write_text(json.dumps(flight));env.close()
    valid=[r for r in rows if r['valid']]
    result=dict(total_frames=len(rows),invalid_frames=len(rows)-len(valid),all_valid=stats(valid),
        visible_valid=stats([r for r in valid if r['visible']]),blind_valid=stats([r for r in valid if not r['visible']]),
        near_end_valid=stats([r for r in ends if r['valid']]),endpoints=ends,
        note='Drone-centre relative forward/left/up in metres. Truth only used as diagnostic comparator; no policy update.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='endpoints'},indent=2))
if __name__=='__main__':main()
