"""Three-batch finite on-policy trial, separate from SAC resume artifacts."""
import sys,json,hashlib
from pathlib import Path
from copy import deepcopy
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT/'source_snapshot'))
import numpy as np,torch
from src.standard_source_policy import StandardSourceSAC
from src.success_history import SuccessHistoryLibrary,SuccessHistoryPractice
from src.return_reset_curriculum import wrap
from src.guided_curriculum import GuidedLessonEnv
from src.finite_policy_update import soft_returns
from src.bounded_finite_update import bounded_update as update_actor,VERSION
from src.unified_evaluation import evaluate
from src.return_evaluation import evaluate_return_practice,_fingerprint
from src.geometric_library import GeometricReturnLibrary

def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2))

def validation(model,label):
    before=_fingerprint(model)
    with torch.inference_mode():
        full=evaluate(model,4,metric_ring=True,geometric_return=True)
        ret,protocol=evaluate_return_practice(model,GeometricReturnLibrary('D:/DroneSoccerData/return-libraries/front-memory-development-v1'))
    assert _fingerprint(model)==before
    result=dict(full=full,returns=ret,protocol=protocol)
    save(label+'.json',result);return result

def main():
    assert not (OUT/'protocol.json').exists()
    torch.set_num_threads(1);assert torch.cuda.is_available()
    checkpoint=Path('D:/DroneSoccerData/runs/standard-source-20260928/initial_model.zip')
    source_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    save('protocol.json',dict(version=VERSION,source_sha256=source_sha,device=torch.cuda.get_device_name(),
        maximum_batches=3,episodes_per_batch=96,maximum_steps_per_batch=72*200+16*150+8*600,
        mixture='40% weight on nine FIT near-pass starts;20% centered lesson2;20% varied lesson3;20% full lesson4; eight rollouts/group',
        objective='Finite complete episode discounted reward plus entropy; practice200,single150,full600,zero terminal tail, no SAC Q gradients',
        actor_lr=3e-6,epochs=4,minibatch=512,clip=.1,mean_kl=.001,max_kl=.02,
        guard='Same whole-epoch KL/surrogate guards; at most4 half-LR retries from identical actor/optimizer/RNG before stopping',
        baseline='same-group leave-one-episode-out timewise return mean, no value critic',
        stop='no accepted epoch; centered<21/25 or varied<23/25; any fixed lap crash or return crash above source; at3batches stop regardless',
        progress='>=5/50 gain new-pass or safe-return successes; never claim full mission acceptance',
        reserved_check='never used during this trial',final_7M_used=False))
    model=StandardSourceSAC.load(checkpoint,device='cuda');critic=deepcopy(model.critic.state_dict())
    baseline=validation(model,'baseline')
    bank=SuccessHistoryLibrary('D:/DroneSoccerData/diagnostics/success-return-bank-v2')
    starts=[r for r in bank.records if r['landmark']=='near_pass']
    optimizer=torch.optim.Adam(model.actor.parameters(),lr=3e-6)
    total_steps=0;stop='bounded_trial_complete';last=baseline
    for batch in range(3):
        save('status.json',dict(mode='collecting',batch=batch,steps=total_steps))
        episodes=[]
        for group in range(12):
            for rep in range(8):
                seed=3900000+batch*10000+group*8+rep;torch.manual_seed(seed)
                if group<9:
                    env=SuccessHistoryPractice(bank);obs,info=env.reset(options={'snapshot_id':starts[group]['id']})
                else:
                    stage=group-7;env=wrap(GuidedLessonEnv(stage=stage),single=stage<4,metric_ring=True,geometric_return=True)
                    obs,info=env.reset(seed=seed)
                xx=[];aa=[];zz=[];rr=[];ll=[]
                while True:
                    with torch.no_grad():
                        mu,ls,_=model.actor.get_action_dist_params(torch.as_tensor(obs,device='cuda')[None])
                        dist=model.actor.action_dist.proba_distribution(mu,ls);a=dist.get_actions(False)
                        z=dist.gaussian_actions;lp=dist.log_prob(a,z)
                    xx.append(obs.copy());aa.append(a[0].cpu().numpy().copy());zz.append(z[0].cpu().numpy().copy());ll.append(float(lp.item()))
                    obs,r,t,tr,info=env.step(aa[-1]);rr.append(r)
                    if t or tr:break
                env.close();total_steps+=len(rr)
                episodes.append(dict(group=group,seed=seed,obs=np.asarray(xx),actions=np.asarray(aa),gaussian=np.asarray(zz),logp=np.asarray(ll,dtype=np.float32),
                    rewards=np.asarray(rr),returns=soft_returns(rr,ll,float(model.ent_coef_tensor),.995),outcome=info['outcome'],passes=info['passes']))
        arrays={f'{i}_{k}':e[k] for i,e in enumerate(episodes) for k in ('obs','actions','gaussian','logp','rewards','returns')}
        np.savez_compressed(OUT/f'rollouts_{batch}.npz',**arrays)
        save(f'episodes_{batch}.json',[{k:v for k,v in e.items() if k not in ('obs','actions','gaussian','logp','rewards','returns')} for e in episodes])
        weights=[]
        for e in episodes:
            peers=[p for p in episodes if p['group']==e['group'] and p is not e]
            base=np.asarray([np.mean([p['returns'][t] for p in peers if t<len(p['returns'])]) if any(t<len(p['returns']) for p in peers) else 0 for t in range(len(e['returns']))])
            mass=.4/9 if e['group']<9 else .2
            weights.extend((e['returns']-base)*.995**np.arange(len(base))*mass/8)
        data=[torch.as_tensor(np.concatenate([e[k] for e in episodes]),device='cuda') for k in ('obs','actions','gaussian')]
        w=torch.as_tensor(weights,dtype=torch.float32,device='cuda');w=w/w.abs().mean().clamp_min(1e-8)
        lp=torch.as_tensor(np.concatenate([e['logp'] for e in episodes]),device='cuda')
        save('status.json',dict(mode='updating',batch=batch,steps=total_steps))
        torch.manual_seed(4000000+batch)
        history=update_actor(model.actor,optimizer,*data,w,lp)
        save(f'updates_{batch}.json',history)
        torch.save(dict(version=VERSION,actor=model.actor.state_dict(),optimizer=optimizer.state_dict(),source_sha256=source_sha,batch=batch,
                        source_checkpoint=str(checkpoint),production_compatible=False),OUT/f'candidate_{batch}.pt')
        assert all(torch.equal(v,model.critic.state_dict()[k]) for k,v in critic.items())
        if not any(r['accepted'] for r in history):stop='no_accepted_epoch';break
        save('status.json',dict(mode='evaluating',batch=batch,steps=total_steps))
        last=validation(model,f'candidate_{batch}')
        a,b=last['full']['suites'],baseline['full']['suites']
        retained=a['original_centered']['successful_episodes']>=21 and a['varied_approach']['successful_episodes']>=23
        safe=all(a[k]['crash_rate']<=b[k]['crash_rate'] for k in ('lesson','full_task','full_task_delay_200ms')) and last['returns']['crash_episodes']<=baseline['returns']['crash_episodes']
        print('Batch',batch,'retained',retained,'crash gate',safe,flush=True)
        if not retained or not safe:stop='retention_or_crash_gate_failed';break
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==source_sha
    progress=any(last['returns'][k]>=baseline['returns'][k]+5 for k in ('successful_episodes','safe_front_return_episodes'))
    save('status.json',dict(mode='stopped',reason=stop,steps=total_steps,development_progress=progress,mission_accepted=False,reserved_rollouts=0))
if __name__=='__main__':main()
