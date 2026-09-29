"""Check identical FIT starts and compare values to repeated return estimates."""
from pathlib import Path
import json,sys
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'diagnostics/basic-return-variance-20260928'
sys.path.insert(0,str(OUT/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory
torch.set_num_threads(1)
result=json.loads((OUT/'result.json').read_text());rows=[]
models={name:RecurrentPPO.load(ROOT/f'diagnostics/recurrent-value-{name}-20260928/model_16.zip',device='cuda') for name in ('broad','balanced')}
for summary in result['summaries']:
    i=summary['index'];obs=[np.load(OUT/f'case_{i}_repeat_{r}.npz')['observations'][0] for r in range(8)]
    assert all(np.array_equal(obs[0],o) for o in obs)
    row=dict(summary,initial_observations_exact=True)
    for name,model in models.items():
        model.policy.set_training_mode(False)
        with torch.no_grad():
            state,start=_initial_memory(model,np.empty((0,119),np.float32),True)
            row[name]=float(model.policy.predict_values(torch.as_tensor(obs[0][None],device='cuda'),state.vf,start).item()*40)
    rows.append(row)
metrics={name:float(np.sqrt(np.mean([(r[name]-r['mean'])**2 for r in rows]))) for name in models}
(OUT/'value_comparison.json').write_text(json.dumps(dict(rows=rows,rmse_against_eight_repeat_mean=metrics,fit_diagnostic_only=True),indent=2))
print(json.dumps(dict(rmse=metrics,rows=rows),indent=2))
