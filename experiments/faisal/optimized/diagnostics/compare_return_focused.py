"""Paired development validation of source and both accepted checkpoints."""
from pathlib import Path
import json,sys,hashlib,shutil
import torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/return-focused-actor-20260929'
OUT=RUN/'development_comparison'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate

def main():
    OUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUT/'runner.py');torch.set_num_threads(1)
    paths=[RUN/f'model_{i}.zip' for i in range(3)]
    (OUT/'protocol.json').write_text(json.dumps(dict(seed_start=6920000,episodes=25,checkpoints=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        purpose='Development comparison of all accepted checkpoints, not independent acceptance',optimization=False),indent=2))
    summaries=[]
    for i,path in enumerate(paths):
        predictor=RecurrentPredictor(RecurrentPPO.load(path,device='cuda'))
        report=evaluate(predictor,4,episodes=25,seed_start=6920000,metric_ring=True,geometric_return=True,episode_reset=predictor.reset_episode)
        (OUT/f'evaluation_{i}.json').write_text(json.dumps(report,indent=2))
        summaries.append(dict(checkpoint=i,suites={name:{k:s[k] for k in ('successful_episodes','crash_rate','total_passes','episodes_with_two_passes','returned_to_front_episodes')} for name,s in report['suites'].items()}))
        print(json.dumps(summaries[-1]),flush=True)
    (OUT/'result.json').write_text(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
