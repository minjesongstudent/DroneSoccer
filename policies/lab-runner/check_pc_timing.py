"""Benchmark the actual selected PC policy on this host; never opens sockets."""
import json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from pc_runtime import PCRuntime
from src.command_rehearsal import encode_action

if __name__=='__main__':
    r=PCRuntime();samples=[];previous=np.zeros(4,np.float32)
    for i in range(100):
        start=time.perf_counter()
        _,action=r.predict(np.array([0,0,.2,.2,0,0,1,0],np.float32),previous)
        _,previous=encode_action(action);samples.append(time.perf_counter()-start)
    report=dict(utc=datetime.now(timezone.utc).isoformat(),policy_identity=r.identity,
        steps=len(samples),median_seconds=float(np.median(samples)),max_seconds=max(samples),
        passes_100ms_policy_only_budget=max(samples)<.1,
        limitation='Policy-only timing: excludes camera detection, decoding, UI, UDP and physical response. Not flight qualification.')
    folder=Path(__file__).parent/'checks';folder.mkdir(exist_ok=True)
    path=folder/f'timing-{time.time_ns()}.json';path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));print('Saved:',path)
    raise SystemExit(0 if report['passes_100ms_policy_only_budget'] else 1)
