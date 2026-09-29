"""Preserve all live align-front entries in the 32 recorded learned flights."""
from pathlib import Path
import sys,json,pickle,hashlib,shutil
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'
BANK=ROOT/'diagnostics/recurrent-return-histories-20260928'
OUT=ROOT/'diagnostics/authentic-reapproaches-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))

def main():
    OUT.mkdir(exist_ok=False);(OUT/'snapshots').mkdir()
    shutil.copytree(RUN/'source_snapshot',OUT/'source_snapshot')
    shutil.copy2(__file__,OUT/'collector.py')
    entries=json.loads((RUN/'protocol.json').read_text())['snapshots'];snapshots=[];flights=[]
    for update in range(4):
        for i,entry in enumerate(entries):
            payload=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
            assert hashlib.sha256(payload).hexdigest()==entry['sha256']
            d=pickle.loads(payload);env=d['environment'];b=env.unwrapped;o=d['observation']
            previous_phase=b._lap_phase;data=np.load(RUN/f'rollout_{update}_{i}.npz')
            actions=np.clip(data['raw_actions'],env.action_space.low,env.action_space.high)
            for j,action in enumerate(actions):
                np.testing.assert_array_equal(o,data['observations'][j])
                o,r,t,tr,info=env.step(action)
                if not (t or tr) and info['lap_phase']=='align_front' and previous_phase!='align_front':
                    prefix=np.concatenate([d['prefix_observations'],data['observations'][:j+1]])
                    command_prefix=np.concatenate([d['prefix_actions'],actions[:j+1]])
                    assert len(prefix)==len(command_prefix)
                    name=f'{update}_{i}_{info["steps"]}_align_front'
                    state=dict(environment=env,observation=o.copy(),prefix_observations=prefix,
                        prefix_actions=command_prefix,source_model=str(RUN/f'model_{update}.zip'),
                        source_rollout=str(RUN/f'rollout_{update}_{i}.npz'))
                    blob=pickle.dumps(state);(OUT/'snapshots'/f'{name}.pkl').write_bytes(blob)
                    restored=pickle.loads(blob)
                    predicted=restored['environment'].step(actions[j+1])
                    target=data['observations'][j+2] if j+2<len(actions) else data['next_observation']
                    np.testing.assert_array_equal(predicted[0],target)
                    np.testing.assert_allclose(predicted[1],data['rewards'][j+1],rtol=0,atol=0)
                    assert (predicted[2] or predicted[3]) == (j+1==len(actions)-1)
                    restored['environment'].close()
                    rel=b.pos-b.hoop_centre
                    snapshots.append(dict(id=name,sha256=hashlib.sha256(blob).hexdigest(),update=update,snapshot=i,
                        step=info['steps'],passes=info['passes'],prefix_steps=len(prefix),visible=bool(o[70]>.5),
                        ring_distance=float(rel@b.hoop_normal),lateral_error=float(rel@b._axis_h),
                        vertical_error=float(rel@b._axis_v),next_transition_exact=True))
                previous_phase=info['lap_phase']
            np.testing.assert_array_equal(o,data['next_observation'])
            flights.append(dict(update=update,snapshot=i,outcome=info['outcome'],phase=info['lap_phase'],
                noodle=bool(b._hit()),hanger=bool(b._hits_hanger()),visible=bool(o[70]>.5)))
            env.close()
    result=dict(selection='Every live transition to align_front in all 32 recorded training continuations; no outcome filter',
        snapshots=snapshots,flights=flights,hidden_states='Not reused: rebuild current actor and critic from actual prefix',
        acceptance=False)
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(snapshots=len(snapshots),visible=sum(x['visible'] for x in snapshots),
        collision_types=dict(Counter(('both' if r['noodle'] and r['hanger'] else 'noodle' if r['noodle'] else 'hanger' if r['hanger'] else r['outcome']) for r in flights)),entries=snapshots),indent=2))
if __name__=='__main__':main()
