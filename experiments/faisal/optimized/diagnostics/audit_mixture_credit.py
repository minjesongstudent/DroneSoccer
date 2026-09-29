"""Inspect task weighting and normalized reward targets without training."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/acquisition-return-mixture-20260928'
def main():
    batches=[]
    for batch in range(4):
        arrays=[np.load(RUN/f'rollout_{batch}_{i}.npz') for i in range(27)]
        all_returns=np.concatenate([a['returns'] for a in arrays]);mean=all_returns.mean();std=all_returns.std()+1e-8
        metrics=json.loads((RUN/f'update_{batch+1}.json').read_text());groups={}
        for name,indices in dict(early_return=range(8),reapproach=range(8,15),natural_full=range(15,23),basic=range(23,27)).items():
            values=np.concatenate([arrays[i]['returns'] for i in indices]);normalized=(values-mean)/std
            for i in indices:
                a=arrays[i];assert bool(a['natural_start'])==(i>=15)
                assert len(a['prefix'])==0 if i>=15 else len(a['prefix'])>0
            groups[name]=dict(episodes=len(indices),frames=len(values),frame_share=float(len(values)/len(all_returns)),
                mean_return=float(values.mean()),normalized_mean=float(normalized.mean()),positive_fraction=float((normalized>0).mean()),
                crashes=sum(metrics['rows'][i]['final_info']['outcome'] in ('hit','left_cage','wrong_way') for i in indices),
                new_passes=sum(metrics['rows'][i]['final_info']['passes']-metrics['rows'][i]['initial_passes'] for i in indices))
        batches.append(dict(batch=batch,global_mean=float(mean),global_std=float(std),groups=groups,guard=metrics['guard']))
    result=dict(note='Target distribution audit; signs are not action quality labels and do not alone prove biased gradients',batches=batches,natural_flags_verified=108)
    (RUN/'credit_balance_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
