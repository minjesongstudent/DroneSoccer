"""FIT-only diagnostic of center-aligned rewards versus projected guard collision."""
import json
from pathlib import Path
import shutil
import numpy as np
from src.sim import HoopEnv
from tools.run_clearance_experiment import ROOT, sha, write_json


def projected_contacts(env, positions, headings):
    """Translate each pose along the ring normal to its plane, retaining yaw."""
    delta=positions-env.hoop_centre
    along=delta@env.hoop_normal
    projected=positions-along[:,None]*env.hoop_normal
    difference=env._centreline[None,:,:]-projected[:,None,:]
    c=np.cos(headings)[:,None];s=np.sin(headings)[:,None]
    body=np.stack([c*difference[:,:,0]+s*difference[:,:,1],
                   -s*difference[:,:,0]+c*difference[:,:,1],difference[:,:,2]],axis=2)
    distances=np.linalg.norm(np.maximum(np.abs(body)-env._half_box,0),axis=2)
    return distances.min(axis=1)<env.hoop.thickness/2


def main():
    source=Path('D:/DroneSoccerData/runs/clearance-reward-20261007')
    traces=source/'clearance/training_trajectories'
    out=Path('D:/DroneSoccerData/runs/guard-alignment-20261007');out.mkdir(exist_ok=False)
    for name in ['sim.py','params.py','contract.py']:
        assert sha(source/'source/src'/name)==sha(ROOT/'src'/name)
    rows=[];hashes={str(traces/'episodes.jsonl'):sha(traces/'episodes.jsonl')};total=0
    for index,line in enumerate((traces/'episodes.jsonl').read_text().splitlines()):
        row=json.loads(line);total+=row['steps']
        if total>200000:break
        if row['status']!='complete':continue
        path=traces/row['trace'];hashes[str(path)]=sha(path)
        env=HoopEnv(level=1,randomise=True,hanger=True);env.reset(seed=row['seed'])
        with np.load(path) as z:
            np.testing.assert_array_equal(env.hoop_centre,z['hoop_centre'])
            np.testing.assert_array_equal(env.hoop_normal,z['hoop_normal'])
            np.testing.assert_array_equal(env._half_box,z['drone_half_box'])
            # Use pre-action poses; initial pose is recovered by exact reset.
            pos=np.vstack([env.pos,z['positions'][:-1]])
            heading=np.r_[env.heading,z['headings'][:-1]]
            delta=pos-env.hoop_centre;along=delta@env.hoop_normal
            inside=(delta@env._axis_h/env._hole_h)**2+(delta@env._axis_v/env._hole_v)**2<1
            eligible=(along<0)&(along>=-.4)&inside
            ids=np.flatnonzero(eligible)
            collision=projected_contacts(env,pos[ids],heading[ids]) if len(ids) else np.array([],bool)
            rows.append(dict(seed=row['seed'],split='heldout' if index%5==4 else 'fit',
                outcome=row['outcome'],passes=row['full_body_passes'],eligible=int(eligible.sum()),
                projected_guard_contact=int(collision.sum()),indices=ids[collision].tolist()))
        env.close()
    summary={}
    for split in ['fit','heldout']:
        selected=[r for r in rows if r['split']==split]
        eligible=sum(r['eligible'] for r in selected);contacts=sum(r['projected_guard_contact'] for r in selected)
        summary[split]=dict(episodes=len(selected),eligible_states=eligible,projected_contacts=contacts,
                           fraction=contacts/eligible,episodes_with_mismatch=sum(r['projected_guard_contact']>0 for r in selected))
    write_json(out/'report.json',dict(summary=summary,rows=rows,input_hashes=hashes,
        caveat='Geometric translation diagnostic only, not a simulated movement or predicted crash. Yaw and lateral offset may change before crossing. Noodle contacts only, no hanger test. Correlated states, no causal claim.'))
    shutil.copy2(__file__,out/'audit_source.py');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
