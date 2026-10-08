"""Finite offline optimizer proposals on saved rollouts; never a promoted model."""
import copy
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from src.constrained_sac import ConstrainedSAC,episode_partition,gaussian_kl,boundary_penalty
from tools.run_onpolicy_trial import action_log_prob,clipped_loss
from tools.run_clearance_experiment import ROOT,write_json,sha


def main():
    torch.set_num_threads(1)
    run=Path('D:/DroneSoccerData/runs/approach-curriculum-20261007')
    model=ConstrainedSAC.load(run/'policy_01.zip',device='cuda')
    reference=ConstrainedSAC.load(ROOT/'trainedModels/seenflag-s1-level1.zip',device='cuda')
    with open('D:/DroneSoccerData/runs/clearance-reward-20261007/clearance/step_200000/replay_buffer.pkl','rb') as stream:r=pickle.load(stream)
    fit,_,_=episode_partition(r.dones[:r.pos]);ids=np.random.default_rng(11520000).choice(fit,8192,replace=False)
    guard=torch.tensor(r.observations[ids,0],device='cuda');del r
    with torch.no_grad():refdist=reference.actor.get_action_dist_params(guard)[:2]
    z=np.load(run/'rollout_02.npz');advantages=np.load(run/'advantages_02.npy')
    weights=copy.deepcopy(model.actor.state_dict());state=copy.deepcopy(model.actor.optimizer.state_dict())
    records=[]
    for start in range(0,min(len(advantages),4096),256):
        model.actor.load_state_dict(weights);model.actor.optimizer.load_state_dict(copy.deepcopy(state))
        sl=slice(start,start+256);x=torch.tensor(z['observations'][sl],device='cuda');a=torch.tensor(z['actions'][sl],device='cuda')
        old=torch.tensor(z['old_log_prob'][sl],device='cuda');adv=torch.tensor(advantages[sl],device='cuda')
        m,s,_=model.actor.get_action_dist_params(x)
        with torch.no_grad():rm,rs,_=reference.actor.get_action_dist_params(x)
        gm,gs,_=model.actor.get_action_dist_params(guard)
        loss=clipped_loss(action_log_prob(model.actor,x,a),old,adv)+gaussian_kl(rm,rs,m,s).mean()+.1*boundary_penalty(gaussian_kl(*refdist,gm,gs),.01,.1)
        model.actor.optimizer.zero_grad();loss.backward()
        finite=all(torch.isfinite(p.grad).all() for p in model.actor.parameters() if p.grad is not None)
        norm=torch.nn.utils.clip_grad_norm_(model.actor.parameters(),.5)
        model.actor.optimizer.step()
        with torch.no_grad():
            gm,gs,_=model.actor.get_action_dist_params(guard);kl=gaussian_kl(*refdist,gm,gs)
        records.append(dict(start=start,loss=float(loss.detach()),gradient_finite=bool(finite),gradient_norm=float(norm),
                            proposed_mean_kl=float(kl.mean()),proposed_max_kl=float(kl.max())))
    write_json(run/'rejection_check_independent.json',dict(records=records,model_sha256=sha(run/'policy_01.zip'),
        note='16 independent discarded proposals from checkpoint; sequential batches not exact historical update order. No new flights or saved actor.'))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
