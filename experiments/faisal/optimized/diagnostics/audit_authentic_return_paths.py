"""Replay recorded learned actions exactly; truth is diagnostic only."""
from pathlib import Path
import sys,json,pickle,hashlib
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'
BANK=ROOT/'diagnostics/recurrent-return-histories-20260928'
OUT=RUN/'trajectory_audit'
sys.path.insert(0,str(RUN/'source_snapshot'))

def main():
    OUT.mkdir(exist_ok=True)
    entries=json.loads((RUN/'protocol.json').read_text())['snapshots'];rows=[]
    for update in range(4):
        for i,entry in enumerate(entries):
            payload=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
            assert hashlib.sha256(payload).hexdigest()==entry['sha256']
            d=pickle.loads(payload);env=d['environment'];base=env.unwrapped
            a=np.load(RUN/f'rollout_{update}_{i}.npz');obs=d['observation']
            phases=[];positions=[];headings=[];events=[];rewards=[];passes=[]
            initial_position=base.pos.copy()
            initial_passes=base.passes;prev_phase=base._lap_phase;prev_pass=base.passes
            for j,raw in enumerate(a['raw_actions']):
                np.testing.assert_array_equal(obs,a['observations'][j])
                obs,reward,t,tr,info=env.step(np.clip(raw,env.action_space.low,env.action_space.high))
                phases.append(info['lap_phase']);positions.append(base.pos.copy());headings.append(base.heading)
                rewards.append(reward);passes.append(base.passes)
                if info['lap_phase']!=prev_phase or base.passes!=prev_pass:
                    events.append(dict(step=info['steps'],phase=info['lap_phase'],passes=base.passes))
                prev_phase=info['lap_phase'];prev_pass=base.passes
                assert (t or tr)==(j==len(a['raw_actions'])-1)
            np.testing.assert_array_equal(obs,a['next_observation'])
            np.testing.assert_allclose(rewards,a['rewards'],rtol=0,atol=0)
            visible=a['observations'][:,70]>.5
            pos=np.asarray(positions);yaw=np.unwrap(headings)
            signed=(pos-base.hoop_centre)@base.hoop_normal
            last_visible=np.flatnonzero(visible)
            row=dict(update=update,snapshot=i,id=entry['id'],outcome=info['outcome'],steps=len(pos),
                new_passes=base.passes-initial_passes,final_step=info['steps'],events=events,
                phase_steps=dict(Counter(phases)),visible_fraction=float(visible.mean()),
                final_invisible_steps=int(len(pos)-1-last_visible[-1]) if len(last_visible) else len(pos),
                travel_m=float(np.linalg.norm(np.diff(np.vstack([initial_position,pos]),axis=0),axis=1).sum()),
                net_yaw_degrees=float(np.rad2deg(yaw[-1]-yaw[0])),
                absolute_yaw_degrees=float(np.rad2deg(np.abs(np.diff(yaw)).sum())),
                signed_ring_distance_min=float(signed.min()),signed_ring_distance_final=float(signed[-1]),
                final_position=pos[-1].tolist(),altitude_min=float(pos[:,2].min()),
                reward=float(sum(rewards)))
            rows.append(row)
            np.savez_compressed(OUT/f'path_{update}_{i}.npz',positions=pos,headings=headings,
                phases=phases,passes=passes,visible=visible,rewards=rewards)
            env.close()
    result=dict(exact_replays=len(rows),outcomes=dict(Counter(r['outcome'] for r in rows)),
        new_passage_rows=[r for r in rows if r['new_passes']],rows=rows)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
if __name__=='__main__':main()
