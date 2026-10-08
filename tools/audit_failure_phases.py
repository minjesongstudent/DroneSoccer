"""Offline geometric phases and pre-crash commands from paired learned traces."""
from collections import Counter
import json
from pathlib import Path
import shutil
import numpy as np
from tools.run_clearance_experiment import sha, write_json


def classify(row, data, thickness):
    positions = data['positions']; headings = data['headings']
    normal = data['hoop_normal']; delta = positions-data['hoop_centre']
    along = delta @ normal
    c, s = np.cos(headings), np.sin(headings)
    reach = (np.abs(c*normal[0]+s*normal[1])*data['drone_half_box'][0]
             + np.abs(-s*normal[0]+c*normal[1])*data['drone_half_box'][1]
             + abs(normal[2])*data['drone_half_box'][2])
    margin = reach + thickness/2
    full = row['full_body_steps']
    if full:
        returned = np.flatnonzero(along[full[-1]:] < -margin[full[-1]:])
        phase = 'subsequent_approach' if len(returned) else 'departure_or_return'
    else:
        phase = 'first_approach'
    if abs(along[-1]) <= margin[-1]:
        phase += '_in_plane_slab'
    obs = data['actor_observations']
    assert obs.shape[1] in (85,121)
    visible = obs[:,64+6] > .5
    tail = data['actions'][-10:]
    blind = 0
    for seen in visible[::-1]:
        if seen: break
        blind += 1
    return dict(seed=row['seed'], crashed=row['crashed'], outcome=row['outcome'], phase=phase,
                passes=row['full_body_passes'], returns=row['returns_to_front'], wall_steps=row['wall_steps'],
                clean_repeat=row['clean_repeated_episode'], clean_first=row['clean_first_pass'],
                terminal_blind_steps=blind, last10_visible_fraction=float(visible[-10:].mean()),
                last10_mean_action=tail.mean(0).tolist(), last10_abs_action=np.abs(tail).mean(0).tolist(),
                terminal_along=float(along[-1]), terminal_vertical=float(delta[-1,2]))


def main():
    root = Path('D:/DroneSoccerData/runs/twelve-adapter-fresh-20261007')
    out = Path('D:/DroneSoccerData/runs/failure-phase-20261007'); out.mkdir(exist_ok=False)
    assert (root/'DONE').exists()
    # Bind the geometric slab assumption to the evaluated source, not current settings.
    from src.sim import HoopEnv
    from tools.validate_flight_policies import start, ROOT
    for name in ['sim.py','params.py','contract.py']:
        assert sha(root/'source/src'/name)==sha(ROOT/'src'/name), 'Reset source changed'
    records={}; hashes={}; summary={}
    for model in ['original','adapter']:
        for suite in ['aligned','nominal','randomised']:
            folder=root/model/suite; log=folder/'episodes.jsonl'
            hashes[str(log)]=sha(log)
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            values=[]
            for row in rows:
                path=folder/f"{row['seed']}.npz"; hashes[str(path)]=sha(path)
                env=HoopEnv(level=1,randomise=suite=='randomised',hanger=True)
                start(env,suite,row['seed'])
                with np.load(path) as data:
                    np.testing.assert_array_equal(env.hoop_centre,data['hoop_centre'])
                    np.testing.assert_array_equal(env.hoop_normal,data['hoop_normal'])
                    np.testing.assert_array_equal(env._half_box,data['drone_half_box'])
                    values.append(classify(row,data,env.hoop.thickness))
                env.close()
            key=model+'/'+suite; records[key]=values
            groups={}
            for v in values:
                if v['crashed']: groups.setdefault(v['outcome']+'/'+v['phase'],[]).append(v)
            summary[key]={k:dict(count=len(vs), median_blind_steps=float(np.median([v['terminal_blind_steps'] for v in vs])),
                mean_abs_action=np.mean([v['last10_abs_action'] for v in vs],axis=0).tolist()) for k,vs in groups.items()}
    paired={}
    for suite in ['aligned','nominal','randomised']:
        a=records['original/'+suite]; b=records['adapter/'+suite]
        assert [v['seed'] for v in a]==[v['seed'] for v in b]
        paired[suite]=dict(gains=[dict(original=x,adapter=y) for x,y in zip(a,b) if y['clean_repeat'] and not x['clean_repeat']],
                           losses=[dict(original=x,adapter=y) for x,y in zip(a,b) if x['clean_repeat'] and not y['clean_repeat']])
    write_json(out/'report.json',dict(summary=summary,records=records,paired=paired,input_hashes=hashes,
        caveat='Terminal crash phase only; first wall contact time absent. Geometric plane slab is not proof of a legal crossing. Post-pass return search uses saved full-body step boundaries. Associations are not causation.'))
    shutil.copy2(__file__,out/'audit_source.py')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
