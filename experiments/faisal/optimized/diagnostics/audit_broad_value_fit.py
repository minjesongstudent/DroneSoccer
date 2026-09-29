"""FIT-only residuals and sequential/whole-sequence critic parity; no optimization."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'diagnostics/recurrent-value-broad-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory

def main():
    torch.set_num_threads(1)
    model=RecurrentPPO.load(RUN/'model_16.zip',device='cuda');p=model.policy
    p.set_training_mode(False)
    protocol=json.loads((RUN/'protocol.json').read_text())
    outcomes={r['index']:r for r in json.loads((ROOT/'diagnostics/value-natural-coverage-20260928/result.json').read_text())['rows']}
    rows=[];all_predictions={};all_targets={}
    with torch.no_grad():
        for i in protocol['fit']:
            record=protocol['records'][i];path=Path(record['path'])
            assert record['split']=='fit' and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
            with np.load(path) as a:
                obs=torch.as_tensor(a['observations'],device='cuda')
                state,start=_initial_memory(model,a['prefix'],bool(a['natural_start']))
                starts=torch.zeros(len(obs),device='cuda');starts[0]=start[0]
                batch=p.predict_values(obs,state.vf,starts).flatten().cpu().numpy()*40
                sequential=[]
                for frame in obs:
                    _,value,_,state=p.forward(frame[None],state,start,deterministic=True)
                    sequential.append(float(value.item())*40);start=torch.zeros_like(start)
                target=a['returns'].copy()
            outcome=outcomes[int(path.stem.split('_')[-1])] if path.name.startswith('episode_') else {}
            group=record['group'];all_predictions.setdefault(group,[]).append(batch);all_targets.setdefault(group,[]).append(target)
            rows.append(dict(index=i,group=group,path=str(path),frames=len(target),outcome=outcome.get('outcome','continuation'),passes=outcome.get('passes'),
                rmse=float(np.sqrt(np.mean((batch-target)**2))),bias=float(np.mean(batch-target)),target_mean=float(target.mean()),prediction_mean=float(batch.mean()),
                target_std=float(target.std()),prediction_std=float(batch.std()),max_sequence_error=float(np.max(np.abs(batch-sequential)))))
    groups={}
    for group in all_targets:
        targets=np.concatenate(all_targets[group]);preds=np.concatenate(all_predictions[group])
        groups[group]=dict(target_std=float(targets.std()),prediction_std=float(preds.std()),rmse=float(np.sqrt(np.mean((targets-preds)**2))),
            target_mean=float(targets.mean()),prediction_mean=float(preds.mean()))
    result=dict(fit_only=True,optimization_steps=0,rows=rows,groups=groups,max_sequence_error=max(r['max_sequence_error'] for r in rows))
    (RUN/'fit_residual_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(groups=groups,max_sequence_error=result['max_sequence_error'],worst=sorted(rows,key=lambda r:r['rmse'],reverse=True)[:8]),indent=2))
    assert result['max_sequence_error']<.001,'Critic batch/step prediction mismatch'
if __name__=='__main__':main()
