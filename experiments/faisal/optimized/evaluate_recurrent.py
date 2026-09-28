"""Evaluate a saved recurrent checkpoint; no training or physical connection."""
import argparse,json
from pathlib import Path
import torch
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 assert torch.cuda.is_available(),'CUDA required for this evaluation'
 torch.set_num_threads(1);model=RecurrentPPO.load(args.model,device='cuda');predictor=RecurrentPredictor(model)
 with torch.inference_mode():report=evaluate(predictor,4,metric_ring=True,geometric_return=True,episode_reset=predictor.reset_episode)
 args.output.write_text(json.dumps(report,indent=2));print({k:v['successful_episodes'] for k,v in report['suites'].items()})
if __name__=='__main__':main()
