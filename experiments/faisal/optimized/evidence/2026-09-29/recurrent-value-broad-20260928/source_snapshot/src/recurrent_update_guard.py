"""Transactional optimizer step with current-memory Gaussian KL checks."""
from copy import deepcopy
import numpy as np
import torch
from .recurrent_burnin import rebuild_actor_memory

def distributions(model, segments):
    result=[]
    was_training=model.policy.training
    model.policy.set_training_mode(False)
    try:
        with torch.no_grad():
            for seg in segments:
                natural=getattr(seg,'natural_start',False)
                if natural:
                    if np.asarray(seg.prefix).shape!=(0,119):raise ValueError('Natural reset must have an empty prefix')
                    state=tuple(torch.zeros(model.policy.lstm_hidden_state_shape,device=model.device) for _ in range(2))
                else:
                    state,_=rebuild_actor_memory(model,seg.prefix)
                    state=tuple(torch.as_tensor(x,device=model.device) for x in state)
                start=torch.full((1,),float(natural),device=model.device)
                for obs in seg.observations:
                    tensor,_=model.policy.obs_to_tensor(obs)
                    dist,state=model.policy.get_distribution(tensor,state,start)
                    start=torch.zeros_like(start)
                    result.append(torch.stack((dist.distribution.mean[0],dist.distribution.scale[0])).cpu())
    finally:
        model.policy.set_training_mode(was_training)
    return torch.stack(result)

def guarded_step(model, optimizer, segments, reference, mean_limit=.001, max_limit=.01):
    before=deepcopy(model.policy.state_dict());opt_before=deepcopy(optimizer.state_dict())
    accepted=False
    try:
        optimizer.step()
        candidate=distributions(model,segments)
        mu,sigma=reference[:,0].double(),reference[:,1].double()
        nu,tau=candidate[:,0].double(),candidate[:,1].double()
        kl=(torch.log(tau/sigma)+(sigma.square()+(mu-nu).square())/(2*tau.square())-.5).sum(1)
        mean=float(kl.mean());peak=float(kl.max())
        accepted=bool(torch.isfinite(kl).all() and mean<=mean_limit and peak<=max_limit)
        return dict(accepted=accepted,mean_kl=mean,max_kl=peak)
    finally:
        if not accepted:
            model.policy.load_state_dict(before)
            optimizer.load_state_dict(opt_before)
