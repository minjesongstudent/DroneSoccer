"""Estimate diagnostics in existing fixed natural-start validation suites."""
from pathlib import Path
import sys,json
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'
OUT=RUN/'natural_estimate_audit';sys.path.insert(0,str(RUN/'source_snapshot'))
from src.sim import HoopEnv
from src.params import Drone
from src.mission_observation import MissionObservation
from src.front_memory import FrontMemoryObservation
from src.ring_estimate import MetricRingObservation
from src.return_geometry import GeometricReturnProgress
from src.recurrent_baseline import RecurrentPredictor

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    m=RecurrentPPO.load('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip',device='cuda')
    summaries={};rows=[]
    for delay in (0,2):
        errors=[];blind=[];invalid=0;total=0
        for i in range(25):
            e=GeometricReturnProgress(MetricRingObservation(FrontMemoryObservation(MissionObservation(HoopEnv(level=1.,drone=Drone(delay_steps=delay)),single_pass=False))))
            o,_=e.reset(seed=5000000+i);p=RecurrentPredictor(m);b=e.unwrapped
            trace=[];initial_valid=bool(o[118]>.5)
            while True:
                total+=1
                error=None
                if o[118]>.5:
                    rel=b.hoop_centre-b.pos;c,s=np.cos(b.heading),np.sin(b.heading)
                    truth=np.array([c*rel[0]+s*rel[1],-s*rel[0]+c*rel[1],rel[2]])
                    error=float(np.linalg.norm(o[115:118]*3-truth));errors.append(error)
                    if o[70]<.5:blind.append(error)
                else:invalid+=1
                a,_=p.predict(o);trace.append(dict(step=b.steps,error=error,visible=bool(o[70]>.5),passes=b.passes,action=a.tolist()))
                o,r,t,tr,info=e.step(a)
                if t or tr:break
            rows.append(dict(delay=delay,seed=5000000+i,initial_valid=initial_valid,outcome=info['outcome'],passes=info['passes']))
            (OUT/f'{delay}_{i}.json').write_text(json.dumps(trace));e.close()
        summaries[str(delay)]=dict(total_frames=total,invalid_frames=invalid,valid_frames=len(errors),
            median_error=float(np.median(errors)),p95_error=float(np.quantile(errors,.95)),max_error=max(errors),
            blind_valid_frames=len(blind),blind_p95=float(np.quantile(blind,.95)),
            first_pass_episodes=sum(r['passes']>0 for r in rows if r['delay']==delay),
            initial_valid_episodes=sum(r['initial_valid'] for r in rows if r['delay']==delay))
        print(json.dumps({str(delay):summaries[str(delay)]}),flush=True)
    (OUT/'result.json').write_text(json.dumps(dict(summary=summaries,episodes=rows,training_updates=0,seed_start=5000000,acceptance=False),indent=2))
if __name__=='__main__':main()
