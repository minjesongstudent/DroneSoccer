"""Training-only state-value diagnostic. No exported flight actor or truth inputs to actor."""
import argparse
from dataclasses import fields
from pathlib import Path
import shutil
import numpy as np
import torch
from torch import nn
from stable_baselines3 import SAC
from src.clearance_training import ClearanceTrainingEnv
from src.twelve_action_history import TwelveActionObservation
from src.sim import HoopEnv
from tools.audit_critic_rankings import sample
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer


def diagnostic_truth(env):
    sim=env.unwrapped;audit=env.env.audit
    names=[];values=[]
    def add(name,items):
        items=np.asarray(items).reshape(-1)
        for i,value in enumerate(items):names.append(f'{name}.{i}');values.append(float(value))
    for name in ['pos','vel','hoop_centre','hoop_normal','_half_box']:
        add(name,getattr(sim,name))
    add('heading',[np.sin(sim.heading),np.cos(sim.heading)])
    add('motion_time',[sim.yaw_rate,sim.steps/600])
    add('ring',[sim._line_h,sim._line_v,sim.hanger_half_width])
    add('audit',[audit.armed,audit.crossed,audit.waiting_return,audit.count,audit.returns,sim.wall_steps])
    for name in ['cage','hoop','camera','dyn']:
        item=getattr(sim,name)
        for field in fields(item):
            value=getattr(item,field.name)
            if isinstance(value,(int,float)):add(name+'.'+field.name,[value])
    pending=np.zeros((4,4),np.float32)
    assert 1<=len(sim._pending)<=4
    pending[:len(sim._pending)]=np.asarray(sim._pending)
    add('pending_oldest_first',pending)
    return np.asarray(values,np.float32),names


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out;torch.set_num_threads(1)
    assert torch.cuda.is_available()
    parent=Path('D:/DroneSoccerData/runs/mc-critic-20261007')
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(__file__,out/'audit_source.py')
        model=SAC.load(parent/'source_model.zip',device='cuda');model.set_random_seed(11610000)
        frozen={k:v.clone() for k,v in model.actor.state_dict().items()}
        source_hashes={str(parent/'source_model.zip'):sha(parent/'source_model.zip')}
        public=[];truth=[];targets=[];groups=[];feature_names=None
        env=TwelveActionObservation(ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),'clearance'))
        for episode,path in enumerate(sorted(parent.glob('episode_*.npz'))):
            with np.load(path) as saved:
                obs,_=env.reset(seed=int(saved['seed']));torch.manual_seed(11610000+episode)
                episode_truth=[]
                for step,expected in enumerate(saved['observations']):
                    np.testing.assert_array_equal(obs,expected)
                    action,logp=sample(model,obs)
                    np.testing.assert_array_equal(action,saved['actions'][step])
                    values,names=diagnostic_truth(env)
                    if feature_names is None:feature_names=names
                    assert names==feature_names
                    episode_truth.append(values)
                    obs,reward,done,cut,_=env.step(action)
                    np.testing.assert_allclose(reward,saved['rewards'][step],rtol=0,atol=1e-10)
                    assert (done or cut)==(step==len(saved['actions'])-1)
                public.append(np.c_[saved['observations'],saved['actions']])
                truth.append(np.array(episode_truth));targets.append(saved['targets'].copy())
                groups.extend([episode]*len(saved['targets']))
            source_hashes[str(path)]=sha(path)
            np.savez_compressed(out/f'truth_{episode:03d}.npz',truth=episode_truth)
            if (episode+1)%10==0:print('EXACT_REPLAY',episode+1,flush=True)
        env.close()
        assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in frozen.items())
        public=np.concatenate(public);truth=np.concatenate(truth);target=np.concatenate(targets);groups=np.array(groups)
        fit=np.flatnonzero(groups%5!=4);held=np.flatnonzero(groups%5==4)
        write_json(out/'protocol.json',dict(kind='Explicit training-only privileged value diagnostic',
            input_hashes=source_hashes,truth_feature_names=feature_names,actor_updates=0,
            actor_truth_inputs=False,exact_replay_verified=True,acceptance=False,
            variants=['public_zero_padding','public_and_simulator_truth'],seeds=[11630000,11630001],
            checks=[0,500,1500,3000],split='Same whole-episode split as MC calibration',
            target='Same fixed soft returns, with original frozen target-critic timeout tail',
            warning='Diagnostic weights are not deployable policies; never load as actor'))
        np.savez_compressed(out/'split.npz',fit=fit,heldout=held,groups=groups)
        ymean=target[fit].mean();yscale=max(target[fit].std(),1e-3)
        y=torch.tensor((target-ymean)/yscale,device='cuda')
        results={}
        for variant in ['public_zero_padding','public_and_simulator_truth']:
            x=np.c_[public,np.zeros_like(truth) if variant=='public_zero_padding' else truth]
            mean=x[fit].mean(0);scale=np.maximum(x[fit].std(0),.01)
            x=torch.tensor((x-mean)/scale,device='cuda');records=[]
            for seed in [11630000,11630001]:
                torch.manual_seed(seed);rng=np.random.default_rng(seed)
                network=nn.Sequential(nn.Linear(x.shape[1],256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1)).cuda()
                optimizer=torch.optim.Adam(network.parameters(),lr=3e-4)
                for update in range(3001):
                    if update in [0,500,1500,3000]:
                        row=dict(seed=seed,update=update)
                        with torch.no_grad():
                            for name,ids in [('fit',fit),('heldout',held)]:
                                pred=network(x[ids]).flatten()*yscale+ymean
                                error=pred-torch.tensor(target[ids],device='cuda')
                                row[name]=dict(rmse=float(error.square().mean().sqrt()),bias=float(error.mean()))
                        records.append(row);print(variant,row,flush=True)
                    if update==3000:break
                    ids=rng.choice(fit,256,replace=True)
                    loss=(network(x[ids]).flatten()-y[ids]).square().mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(network.parameters(),10);optimizer.step()
                torch.save(dict(weights=network.state_dict(),optimizer=optimizer.state_dict(),xmean=mean,xscale=scale,
                    ymean=ymean,yscale=yscale,truth_features=feature_names),out/f'{variant}-{seed}.pt')
            results[variant]=records;write_json(out/'results.json',results)
        (out/'DONE').write_text('Diagnostic networks fitted; no actor trained, no truth actor input.\n')


if __name__=='__main__':main()
