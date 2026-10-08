"""Actor-frozen CUDA critic regression on complete new learned-policy episodes."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from src.clearance_training import ClearanceTrainingEnv
from src.twelve_action_history import TwelveActionObservation
from src.sim import HoopEnv
from tools.audit_critic_rankings import sample
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer


def soft_targets(rewards,logps,gamma,alpha,tail):
    """Q at each recorded action; entropy starts at the subsequent action."""
    result=np.empty(len(rewards),np.float32);running=float(tail)
    for i in range(len(rewards)-1,-1,-1):
        value=float(rewards[i])+gamma*running
        result[i]=value
        running=value-alpha*float(logps[i])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.out
    assert torch.cuda.is_available();torch.set_num_threads(1)
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        source=Path('D:/DroneSoccerData/runs/twelve-adapter-20261007/online_005000/model.zip')
        shutil.copy2(source,out/'source_model.zip')
        hashes={}
        for file in list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tools/audit_critic_rankings.py',ROOT/'tools/run_clearance_experiment.py']:
            rel=file.resolve().relative_to(ROOT);dest=out/'source'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest);hashes[str(rel)]=sha(dest)
        model=SAC.load(source,device='cuda')
        model.set_random_seed(11610000)
        before={k:v.clone() for k,v in model.actor.state_dict().items()}
        target_before={k:v.clone() for k,v in model.critic_target.state_dict().items()}
        for parameter in model.actor.parameters():parameter.requires_grad_(False)
        alpha=float(model.log_ent_coef.exp());gamma=model.gamma
        episodes=5 if args.smoke else 80;checks=[0,8] if args.smoke else [0,500,1500,3000]
        write_json(out/'protocol.json',dict(kind='Frozen actor Monte Carlo critic calibration',
            source_sha256=sha(source),source_hashes=hashes,seed_base=11210000,episodes=episodes,
            split='Every fifth complete episode held out; no older diagnostic state is fitted',
            actor_updates=0,critic_lr=3e-4,checks=checks,device=torch.cuda.get_device_name(),
            target='Recorded full-episode soft returns; fixed original target critic tail only at timeout',
            target_network='Remains frozen: no self-changing regression target',
            finite_returns_saved=True,acceptance=False,wall_seconds=1800))
        deadline=time.monotonic()+1800;pieces=[]
        env=TwelveActionObservation(ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),
            'clearance',trace_dir=out/'traces'))
        for episode in range(episodes):
            if time.monotonic()>deadline:raise RuntimeError('Collection cap reached')
            obs,_=env.reset(seed=11210000+episode);xs=[];actions=[];rewards=[];logps=[]
            torch.manual_seed(11610000+episode)
            torch.save(torch.get_rng_state(),out/f'cpu_rng_{episode:03d}.pt')
            torch.save(torch.cuda.get_rng_state_all(),out/f'cuda_rng_{episode:03d}.pt')
            while True:
                action,logp=sample(model,obs)
                xs.append(obs.copy());actions.append(action.copy());logps.append(logp)
                obs,reward,done,cut,info=env.step(action);rewards.append(reward)
                if done or cut:break
            tail=0.
            if cut and not done:
                with torch.no_grad():
                    x=torch.tensor(obs[None],dtype=torch.float32,device='cuda')
                    next_action,next_logp=model.actor.action_log_prob(x)
                    tail=float(torch.cat(model.critic_target(x,next_action),1).min()-alpha*next_logp[0])
            targets=soft_targets(rewards,logps,gamma,alpha,tail)
            finite=soft_targets(rewards,logps,gamma,alpha,0)
            arrays=dict(observations=np.asarray(xs),actions=np.asarray(actions),rewards=np.asarray(rewards),
                logps=np.asarray(logps),targets=targets,finite_targets=finite,next_observation=obs,tail=tail,
                episode=episode,seed=11210000+episode,terminated=done,truncated=cut)
            np.savez_compressed(out/f'episode_{episode:03d}.npz',**arrays)
            pieces.append(arrays)
            if (episode+1)%10==0:print('COLLECTED',episode+1,flush=True)
        env.close()
        x=np.concatenate([p['observations'] for p in pieces]);a=np.concatenate([p['actions'] for p in pieces])
        y=np.concatenate([p['targets'] for p in pieces]);finite=np.concatenate([p['finite_targets'] for p in pieces])
        groups=np.concatenate([np.full(len(p['targets']),p['episode']) for p in pieces])
        fit=np.flatnonzero(groups%5!=4);held=np.flatnonzero(groups%5==4)
        np.savez_compressed(out/'split.npz',fit=fit,heldout=held,episode_ids=groups)
        x=torch.tensor(x,device='cuda');a=torch.tensor(a,device='cuda');y=torch.tensor(y,device='cuda')
        optimizer=torch.optim.Adam(model.critic.parameters(),lr=3e-4)
        model.critic.optimizer=optimizer;rng=np.random.default_rng(11610000);results=[]
        for update in range(max(checks)+1):
            if update in checks:
                record=dict(update=update)
                with torch.no_grad():
                    for name,ids in [('fit',fit),('heldout',held)]:
                        pred=torch.cat(model.critic(x[ids],a[ids]),1).min(1).values
                        error=pred-y[ids]
                        record[name]=dict(samples=len(ids),rmse=float(error.square().mean().sqrt()),
                            bias=float(error.mean()),target_mean=float(y[ids].mean()),
                            finite_rmse=float((pred-torch.tensor(finite[ids],device='cuda')).square().mean().sqrt()))
                assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in before.items())
                assert all(torch.equal(v,model.critic_target.state_dict()[k]) for k,v in target_before.items())
                model.save(out/f'critic_{update:06d}.zip')
                results.append(record);write_json(out/'results.json',results);print(record,flush=True)
            if update==max(checks):break
            if time.monotonic()>deadline or (out/'STOP').exists():raise RuntimeError('Bounded critic cap/stop reached')
            ids=rng.choice(fit,256,replace=True)
            predictions=model.critic(x[ids],a[ids])
            loss=sum((pred.flatten()-y[ids]).square().mean() for pred in predictions)
            assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.critic.parameters(),10);optimizer.step()
        torch.save(dict(numpy=rng.bit_generator.state,torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),out/'final_rng.pt')
        write_json(out/'status.json',dict(status='complete',actor_exact=True,target_network_exact=True,actor_updates=0))
        (out/'DONE').write_text('Critic-only calibration completed; no actor update or promotion.\n')


if __name__=='__main__':main()
