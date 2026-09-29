"""Paired diagnostic validation; not the reserved independent acceptance test."""
from pathlib import Path
import sys,json,hashlib
import torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'diagnostics/authentic-return-pg-mixed-20260928'
OUT=RUN/'development_comparison'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    sources=dict(source=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip'),mixed=RUN/'model_4.zip')
    (OUT/'protocol.json').write_text(json.dumps(dict(seed_start=6910000,episodes=25,
        purpose='Paired development check after weak fixed-lesson gain; not acceptance',
        models={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()}),indent=2))
    for name,path in sources.items():
        m=RecurrentPPO.load(path,device='cuda');p=RecurrentPredictor(m)
        r=evaluate(p,4,seed_start=6910000,metric_ring=True,geometric_return=True,episode_reset=p.reset_episode)
        (OUT/f'{name}.json').write_text(json.dumps(r,indent=2))
        print(json.dumps(dict(model=name,success={k:v['successful_episodes'] for k,v in r['suites'].items()})),flush=True)
if __name__=='__main__':main()
