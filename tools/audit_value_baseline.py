"""GPU value-only fit/heldout audit on complete saved FIT episodes. No actor updates."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn
from tools.run_clearance_experiment import sha,write_json,single_trainer,ROOT
from src.camera_history import CameraHistory


LONG_LAGS=(60,50,40,32,24,20,16,12,8,4,2,1,0)


def longer_history(observations,lengths):
    result=[];cursor=0
    for length in lengths:
        history=CameraHistory(LONG_LAGS)
        for obs in observations[cursor:cursor+length]:
            extended=history.update(obs[64:72],obs[80:84])
            np.testing.assert_array_equal(extended[32:],obs)
            result.append(extended)
        cursor+=length
    assert cursor==len(observations)
    return np.asarray(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--long-history',action='store_true')
    args=p.parse_args();out=args.out;torch.set_num_threads(1);assert torch.cuda.is_available()
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        files=sorted(Path('D:/DroneSoccerData/runs/onpolicy-step-20261007').glob('rollout_*.npz'))
        xs=[];ys=[];clocks=[];groups=[];group=0;extended=[]
        for file in files:
            with np.load(file) as z:
                if args.long_history:extended.append(longer_history(z['observations'],z['episode_lengths']))
                cursor=0
                for length in z['episode_lengths']:
                    end=cursor+length;xs.append(z['observations'][cursor:end]);ys.append(z['returns'][cursor:end])
                    clocks.append(np.arange(length,dtype=np.float32)/600);groups.extend([group]*length)
                    cursor=end;group+=1
                assert cursor==len(z['observations'])
        x=np.concatenate(xs);y=np.concatenate(ys);clock=np.concatenate(clocks);groups=np.asarray(groups)
        fit=np.flatnonzero(groups%5!=4);held=np.flatnonzero(groups%5==4)
        np.savez_compressed(out/'split.npz',fit=fit,heldout=held,episode_ids=groups)
        variants=['scaled85','scaled117'] if args.long_history else ['raw85','scaled85','scaled85_clock']
        write_json(out/'protocol.json',dict(kind='value-only diagnostic',sources={str(f):sha(f) for f in files},
            variants=variants,checks=[0,100,500,1000],fit_episodes=len(set(groups[fit])),heldout_episodes=len(set(groups[held])),
            holdout_rule='Every fifth whole collected FIT episode',actor_updated=False,
            clock='Known elapsed control steps/600; baseline only, never actor',acceptance=False))
        target=torch.tensor(y,device='cuda');results={}
        for variant in variants:
            torch.manual_seed(11550000);rng=np.random.default_rng(11550000)
            features=(np.concatenate(extended) if variant=='scaled117' else
                      np.c_[x,clock] if variant.endswith('clock') else x)
            features=torch.tensor(features,device='cuda')
            model=nn.Sequential(nn.Linear(features.shape[1],128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1)).cuda()
            opt=torch.optim.Adam(model.parameters(),lr=3e-4)
            shift=float(y[fit].mean()) if variant!='raw85' else 0.
            scale=float(y[fit].std()) if variant!='raw85' else 1.
            records=[]
            for update in range(1001):
                if update in [0,100,500,1000]:
                    metrics={}
                    with torch.no_grad():
                        for name,ids in [('fit',fit),('heldout',held)]:
                            pred=model(features[ids]).flatten()*scale+shift
                            residual=target[ids]-pred
                            metrics[name]=dict(rmse=float(residual.square().mean().sqrt()),
                                explained_variance=float(1-residual.var(unbiased=False)/target[ids].var(unbiased=False)))
                    records.append(dict(update=update,**metrics));print(variant,update,json.dumps(metrics),flush=True)
                if update==1000:break
                ids=rng.choice(fit,256,replace=True)
                loss=(model(features[ids]).flatten()-(target[ids]-shift)/scale).square().mean()
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);opt.step()
            torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),shift=shift,scale=scale),out/f'{variant}.pt')
            results[variant]=records;write_json(out/'results.json',results)
        shutil.copy2(__file__,out/'audit_source.py');(out/'DONE').write_text('Value audit completed; no actor trained.\n')


if __name__=='__main__':main()
