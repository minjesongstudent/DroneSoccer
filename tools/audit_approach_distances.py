"""Compare saved first approaches at fixed pre-action plane distances."""
import json
from pathlib import Path
import shutil
import numpy as np
from tools.run_clearance_experiment import sha,write_json


def samples(row,trace):
    if row['wall_steps']: return []
    if row['clean_first_pass']:
        label='first_pass'; stop=row['full_body_steps'][0]
    elif row['outcome']=='hit' and not row['full_body_passes']:
        label='ring_hit'; stop=row['steps']
    else: return []
    # Position i-1 is state before action i; index zero has no saved pre-state.
    pos=trace['positions']; normal=trace['hoop_normal']; centre=trace['hoop_centre']
    along=(pos-centre)@normal; out=[]
    for distance in [.30,.15,.08]:
        crossings=np.flatnonzero((along[:-1]<-distance)&(along[1:]>=-distance))+1
        crossings=crossings[(crossings+1<stop)&(crossings+1<len(pos))]
        if not len(crossings):continue
        i=int(crossings[0]+1)
        obs=trace['actor_observations'][i]; actions=trace['actions'];heading=trace['headings'][i-1]
        rel=pos[i-1]-centre; side=np.array([-normal[1],normal[0],0])
        visible=trace['actor_observations'][:i+1,70]>.5; blind=0
        for v in visible[::-1]:
            if v:break
            blind+=1
        action=actions[i]; c,s=np.cos(heading),np.sin(heading)
        world=np.array([action[1]*c+action[0]*s,action[1]*s-action[0]*c,action[2]])
        out.append(dict(seed=row['seed'],label=label,distance=distance,index=i,
            lateral=float(rel@side),vertical=float(rel[2]),abs_lateral=float(abs(rel@side)),
            abs_vertical=float(abs(rel[2])),blind_steps=blind,visible=float(visible[-1]),
            right=float(action[0]),forward=float(action[1]),up=float(action[2]),yaw=float(action[3]),
            abs_yaw=float(abs(action[3])),command_along=float(world@normal),
            command_lateral=float(world@side),camera_u=float(obs[64]),camera_v=float(obs[65])))
    return out


def main():
    root=Path('D:/DroneSoccerData/runs/twelve-adapter-fresh-20261007')
    out=Path('D:/DroneSoccerData/runs/approach-distance-20261007');out.mkdir(exist_ok=False)
    rows=[];hashes={}
    for model in ['original','adapter']:
        for suite in ['aligned','nominal','randomised']:
            folder=root/model/suite; log=folder/'episodes.jsonl';hashes[str(log)]=sha(log)
            for line in log.read_text().splitlines():
                row=json.loads(line);path=folder/f"{row['seed']}.npz";hashes[str(path)]=sha(path)
                with np.load(path) as trace:
                    rows.extend(dict(model=model,suite=suite,**v) for v in samples(row,trace))
    fields=['abs_lateral','abs_vertical','blind_steps','visible','abs_yaw','command_along','command_lateral']
    groups={}
    for r in rows:groups.setdefault(f"{r['model']}/{r['suite']}/{r['distance']}/{r['label']}",[]).append(r)
    summary={k:dict(n=len(vs),**{f:float(np.median([v[f] for v in vs])) for f in fields}) for k,vs in groups.items()}
    write_json(out/'report.json',dict(rows=rows,summary=summary,input_hashes=hashes,
        caveat='Descriptive first threshold crossings, not randomized or matched-layout effects. All wall-contact episodes excluded. Features pre-action, outcome labels retrospective. No actor updates.'))
    shutil.copy2(__file__,out/'audit_source.py')
    print(json.dumps({k:v for k,v in summary.items() if k.startswith('original/')},indent=2))


if __name__=='__main__':main()
