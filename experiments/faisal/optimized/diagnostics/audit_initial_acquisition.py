"""Read reset observations and diagnose missing estimate; no actor changes."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/authentic-return-pg-stepwise-v2-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.sim import HoopEnv
from src.mission_observation import MissionObservation
from src.front_memory import FrontMemoryObservation
from src.ring_estimate import MetricRingObservation
from src.return_geometry import GeometricReturnProgress

def main():
    rows=[]
    for i in range(25):
        e=GeometricReturnProgress(MetricRingObservation(FrontMemoryObservation(MissionObservation(HoopEnv(level=1.),single_pass=False))))
        o,_=e.reset(seed=5000000+i);b=e.unwrapped
        valid=bool(o[118]>.5);visible=bool(o[70]>.5);marker=bool(o[71]>.5)
        reason='valid' if valid else 'ring_not_visible' if not visible else 'front_marker_absent' if not marker else 'estimate_geometry_invalid'
        rel=b.pos-b.hoop_centre
        signed=float(rel@b.hoop_normal)
        front_angle=float(np.rad2deg(np.arccos(np.clip((-rel/np.linalg.norm(rel))@b.hoop_normal,-1,1))))
        trace=json.loads((RUN/'natural_estimate_audit'/f'0_{i}.json').read_text())
        first_valid=next((r['step'] for r in trace if r['error'] is not None),None)
        rows.append(dict(seed=5000000+i,reason=reason,visible=visible,marker=marker,front_seen=bool(o[108]>.5),
            any_seen=bool(o[92]>.5),signed_ring_side_m=signed,front_angle_deg=front_angle,
            first_valid_step=first_valid,initial_observation=o.tolist()))
        e.close()
    from collections import Counter
    result=dict(counts=dict(Counter(r['reason'] for r in rows)),rows=rows,actor_changes=False)
    (RUN/'initial_acquisition_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(counts=result['counts'],invalid=[{k:v for k,v in r.items() if k!='initial_observation'} for r in rows if r['reason']!='valid']),indent=2))
if __name__=='__main__':main()
