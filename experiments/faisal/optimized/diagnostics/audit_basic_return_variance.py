"""Repeat FIT basic reset seeds under the unchanged stochastic learned actor."""
from pathlib import Path
import sys,json,shutil,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'diagnostics/acquisition-return-mixture-20260928'
OUT=ROOT/'diagnostics/basic-return-variance-20260928'
sys.path.insert(0,str(SOURCE/'source_snapshot'))
from src.recurrent_snapshot_rollout import collect_natural_segment
from src.return_reset_curriculum import wrap
from src.curriculum import LessonEnv

def main():
    OUT.mkdir(exist_ok=False)
    shutil.copytree(SOURCE/'source_snapshot',OUT/'source_snapshot');shutil.copy2(__file__,OUT/'runner.py')
    torch.set_num_threads(1)
    checkpoint=SOURCE/'model_0.zip';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model=RecurrentPPO.load(checkpoint,device='cuda')
    cases=[c for c in json.loads((ROOT/'diagnostics/value-natural-coverage-20260928/protocol.json').read_text())['cases'] if c['split']=='fit' and c['group'] in ('centered','varied')]
    assert len(cases)==12
    protocol=dict(cases=cases,repeats=8,torch_seed_rule='991000 + 100 * original index + repeat',source_sha256=digest,
        purpose='FIT-only conditional return variability diagnostic, no optimization or outcome selection',gamma=.999)
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for case in cases:
        for repeat in range(8):
            env=wrap(LessonEnv(stage=0 if case['group']=='centered' else 1),single=True,metric_ring=True,geometric_return=True)
            seed=991000+100*case['index']+repeat;torch.manual_seed(seed)
            s=collect_natural_segment(model,env,case['seed'],600);assert s.mission_ended
            returns=np.zeros(len(s.rewards),np.float32);carry=0.
            for j in range(len(returns)-1,-1,-1):carry=float(s.rewards[j])+.999*carry;returns[j]=carry
            filename=f"case_{case['index']}_repeat_{repeat}.npz"
            np.savez_compressed(OUT/filename,observations=s.observations,raw_actions=s.actions,rewards=s.rewards,returns=returns,prefix=s.prefix,natural_start=True)
            rows.append(dict(index=case['index'],group=case['group'],repeat=repeat,torch_seed=seed,reset_seed=case['seed'],
                outcome=s.final_info['outcome'],passes=s.final_info['passes'],steps=len(returns),initial_return=float(returns[0]),file=filename))
            env.close()
        (OUT/'progress.json').write_text(json.dumps(rows,indent=2));print('completed case',case['index'],flush=True)
    summaries=[]
    for case in cases:
        data=[r for r in rows if r['index']==case['index']];values=np.array([r['initial_return'] for r in data])
        summaries.append(dict(index=case['index'],group=case['group'],mean=float(values.mean()),std=float(values.std(ddof=1)),
            passes=sum(r['passes']>0 for r in data),episodes=len(data)))
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    (OUT/'result.json').write_text(json.dumps(dict(rows=rows,summaries=summaries,actor_unchanged=True,optimization_steps=0),indent=2))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()
