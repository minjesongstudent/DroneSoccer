"""Diagnostic-only paired learned-policy replay; truth never enters actor input."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/return-focused-actor-20260929'
OUT=RUN/'timing_replay';sys.path.insert(0,str(RUN/'source_snapshot'))
from src.sim import HoopEnv
from src.params import Drone
from src.return_reset_curriculum import wrap
from src.recurrent_baseline import RecurrentPredictor

def main():
    OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/'runner.py');torch.set_num_threads(1)
    paths=[RUN/f'model_{i}.zip' for i in range(3)]
    (OUT/'protocol.json').write_text(json.dumps(dict(seed=6920010,delays=[0,2],models=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        purpose='Development trajectory diagnostics only, no training; truth fields logged after actor prediction'),indent=2))
    summaries=[]
    for checkpoint,path in enumerate(paths):
        model=RecurrentPPO.load(path,device='cuda')
        for delay in (0,2):
            env=wrap(HoopEnv(level=1.,drone=Drone(delay_steps=delay)),single=False,metric_ring=True,geometric_return=True)
            obs,_=env.reset(seed=6920010);b=env.unwrapped;predictor=RecurrentPredictor(model)
            observations=[];actions=[];trace=[];events=[];previous=(0,b._lap_phase)
            while True:
                action,_=predictor.predict(obs,deterministic=True)
                observations.append(obs.copy());actions.append(action.copy())
                visible=bool(obs[70]>.5)
                obs,reward,t,tr,info=env.step(action)
                rel=b.pos-b.hoop_centre
                clearance=float(np.min(b._box_distance(b._centreline))-b.hoop.thickness/2)
                row=dict(step=b.steps,position=b.pos.tolist(),velocity=b.vel.tolist(),heading=float(b.heading),
                    along=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),vertical=float(rel[2]),
                    noodle_clearance=clearance,hanger_hit=bool(b._hits_hanger()),noodle_hit=bool(b._hit()),
                    phase=b._lap_phase,passes=int(b.passes),visible_before_action=visible,action=action.tolist())
                trace.append(row)
                current=(b.passes,b._lap_phase)
                if current!=previous:events.append(row);previous=current
                if t or tr:break
            suite='full_task' if delay==0 else 'full_task_delay_200ms'
            original=json.loads((RUN/f'development_comparison/evaluation_{checkpoint}.json').read_text())['suites'][suite]['records']
            expected=next(r for r in original if r['seed']==6920010)
            assert all(info[k]==expected[k] for k in ('outcome','passes','steps'))
            stem=f'model_{checkpoint}_delay_{delay}'
            np.savez_compressed(OUT/f'{stem}.npz',observations=np.asarray(observations),actions=np.asarray(actions))
            (OUT/f'{stem}.json').write_text(json.dumps(trace,indent=2))
            summaries.append(dict(checkpoint=checkpoint,delay=delay,outcome=info['outcome'],passes=int(info['passes']),steps=int(info['steps']),
                events=events,final=trace[-1],minimum_noodle_clearance=min(r['noodle_clearance'] for r in trace),
                blind_frames=sum(not r['visible_before_action'] for r in trace),matches_saved_evaluation=True))
            env.close();print(json.dumps({k:v for k,v in summaries[-1].items() if k not in ('events','final')}),flush=True)
    (OUT/'result.json').write_text(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
