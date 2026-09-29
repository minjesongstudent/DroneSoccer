"""Read-only CUDA gradient and action-distribution audit on saved training data."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/authentic-return-pg-mixed-20260928'
OUT=RUN/'gradient_audit'
sys.path.insert(0,str(RUN/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_burnin import rebuild_actor_memory
from src.recurrent_snapshot_rollout import evaluate_segment

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1)
    original=RecurrentPPO.load(RUN/'model_0.zip',device='cuda')
    final=RecurrentPPO.load(RUN/'model_4.zip',device='cuda')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUN/'model_0.zip',RUN/'model_4.zip')}
    results=[]
    for batch in (0,2):
        model=RecurrentPPO.load(RUN/f'model_{batch}.zip',device='cuda')
        arrays=[np.load(RUN/f'rollout_{batch}_{i}.npz') for i in range(15)]
        targets=np.concatenate([a['returns'] for a in arrays]);mean=targets.mean();std=targets.std()+1e-8
        gradients=[]
        for a in arrays:
            seg=SimpleNamespace(prefix=a['prefix'],observations=a['observations'],actions=a['raw_actions'])
            model.policy.zero_grad()
            _,logs,_=evaluate_segment(model,seg)
            weights=torch.as_tensor((a['returns']-mean)/std,device='cuda')
            (-(logs*weights).sum()/len(targets)).backward()
            gradients.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in model.policy.parameters()]).cpu())
        g=torch.stack(gradients);norms=torch.linalg.vector_norm(g,dim=1)
        cos=(g@g.T)/(norms[:,None]*norms[None,:]).clamp_min(1e-20)
        pairs=cos[torch.triu(torch.ones_like(cos,dtype=torch.bool),diagonal=1)]
        result=dict(batch=batch,sum_gradient_norm=float(g.sum(0).norm()),sum_individual_norms=float(norms.sum()),
            cancellation_ratio=float(g.sum(0).norm()/norms.sum()),negative_pair_fraction=float((pairs<0).float().mean()),
            mean_pair_cosine=float(pairs.mean()),norms=norms.tolist(),cosines=cos.tolist())
        results.append(result)
    # Compare both actors on identical authentic inputs, with their own rebuilt memory.
    kl=[];mse=[];clipped=[]
    for i in range(15):
        a=np.load(RUN/f'rollout_0_{i}.npz');outputs=[]
        for model in (original,final):
            state,_=rebuild_actor_memory(model,a['prefix'])
            state=tuple(torch.as_tensor(x,device='cuda') for x in state)
            means=[];scales=[];model.policy.set_training_mode(False)
            with torch.no_grad():
                for obs in a['observations']:
                    tensor,_=model.policy.obs_to_tensor(obs)
                    dist,state=model.policy.get_distribution(tensor,state,torch.zeros(1,device='cuda'))
                    means.append(dist.distribution.mean[0].cpu().numpy());scales.append(dist.distribution.scale[0].cpu().numpy())
            outputs.append((np.array(means),np.array(scales)))
        (mu,sigma),(nu,tau)=outputs
        kl.extend(np.sum(np.log(tau/sigma)+(sigma*sigma+(mu-nu)**2)/(2*tau*tau)-.5,axis=1).tolist())
        mse.extend(np.mean((mu-nu)**2,axis=1).tolist())
        clipped.extend(np.mean((np.clip(mu,-1,1)-np.clip(nu,-1,1))**2,axis=1).tolist())
    result=dict(gradients=results,actor_drift=dict(frames=len(kl),mean_kl=float(np.mean(kl)),max_kl=float(np.max(kl)),
        mean_action_mse=float(np.mean(mse)),clipped_action_mse=float(np.mean(clipped))),optimization_steps=0)
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(gradients=[{k:v for k,v in r.items() if k not in ('norms','cosines')} for r in results],actor_drift=result['actor_drift'])))
if __name__=='__main__':main()
