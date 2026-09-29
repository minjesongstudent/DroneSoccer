"""Bounded action-head-only transfer with basic-output anchoring; not acceptance."""
from pathlib import Path
import sys, json, hashlib, shutil
import numpy as np
import torch

OLD=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
OUT=Path(__file__).parent/'lap-mlp-anchor10-20260928'
sys.path.insert(0,str(OUT/'source_snapshot'))
sys.path.insert(1,str(OLD/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate

def save(name,data):
    p=OUT/name
    temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2))
    temp.replace(p)

def main():
    torch.set_num_threads(1)
    torch.manual_seed(44)
    assert torch.cuda.is_available()
    OUT.mkdir(exist_ok=False)
    shutil.copytree(OLD/'source_snapshot',OUT/'source_snapshot')
    shutil.copy2(__file__,OUT/'runner_snapshot.py')
    source=OLD/'model_update_500.zip'
    seeds=json.loads((OLD/'protocol.json').read_text())['fit_lap_seeds']
    assert len(seeds)==9 and all(s%4!=3 for s in seeds)
    bank=Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3/trajectories')
    data=[]
    for seed in seeds:
        with np.load(bank/f'{seed}.npz') as d:
            data.append((d['observations'][:-1].copy(),d['actions'].copy()))
    obs=torch.as_tensor(np.stack([d[0] for d in data]),device='cuda')
    targets=torch.as_tensor(np.stack([d[1] for d in data]),device='cuda')
    model=RecurrentPPO.load(source,device='cuda')
    for n,p in model.policy.named_parameters():
        p.requires_grad_(n.startswith(('action_net.','mlp_extractor.policy_net.')))
    params=[p for p in model.policy.parameters() if p.requires_grad]
    frozen_hash=lambda: hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for n,p in model.policy.named_parameters() if not n.startswith(('action_net.','mlp_extractor.policy_net.')))).hexdigest()
    frozen_before=frozen_hash()
    with np.load('D:/DroneSoccerData/runs/recurrent-learned-transfer-20260928/data_600000.npz') as d:
        basic_obs=torch.as_tensor(d['observations'].copy(),device='cuda')
        basic_mask=torch.as_tensor(d['mask'].copy(),device='cuda')
    def means(o):
        b,t,_=o.shape
        z=torch.zeros(1,b,model.policy.lstm_actor.hidden_size,device='cuda')
        dist,_=model.policy.get_distribution(o.reshape(-1,119),(z,z.clone()),torch.zeros(b*t,device='cuda'))
        return dist.distribution.mean.reshape(b,t,4)
    model.policy.set_training_mode(False)
    with torch.inference_mode():
        reference=means(basic_obs).clone()
    reference=reference.clone()
    rng=np.random.default_rng(45)
    opt=torch.optim.Adam(params,lr=1e-4)
    save('protocol.json',dict(kind='Supervised policy-MLP anchored FIT diagnostic; not RL or acceptance',
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seeds=seeds,device=torch.cuda.get_device_name(),max_updates=100,checks=[25,50,100],
        lr=1e-4,loss='nine complete FIT histories plus10x MSE against baseline outputs on8 sampled basic histories; policy hidden layer and action head trainable; LSTM frozen',
        purpose='Preserve recurrent features and penalize basic-output drift; frozen parameter hashes checked',
        retention_stop='centered<22/25 or varied<20/25',reserved_used=False,independent7Mused=False,
        hashes={str(s):hashlib.sha256((bank/f'{s}.npz').read_bytes()).hexdigest() for s in seeds}))
    history=[]
    reason='bounded_diagnostic_complete'
    try:
        for u in range(1,101):
            model.policy.set_training_mode(True)
            opt.zero_grad()
            z=torch.zeros(1,9,model.policy.lstm_actor.hidden_size,device='cuda')
            d,_=model.policy.get_distribution(obs.reshape(-1,119),(z,z.clone()),torch.zeros(9*600,device='cuda'))
            lap_loss=(d.distribution.mean.reshape(9,600,4)-targets).square().mean()
            idx=rng.choice(80,8,replace=False)
            bm=basic_mask[idx]
            anchor=(((means(basic_obs[idx])-reference[idx]).square().mean(-1)*bm).sum(1)/bm.sum(1)).mean()
            loss=lap_loss+10*anchor
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,.5)
            opt.step()
            history.append(dict(update=u,pre_update_lap_mse=float(lap_loss.detach()),anchor_mse=float(anchor.detach())))
            save('history.json',history)
            save('status.json',dict(mode='supervised capacity diagnostic',update=u))
            if u in (25,50,100):
                assert frozen_hash()==frozen_before
                model.save(OUT/f'model_{u}.zip')
                save('status.json',dict(mode='evaluating',update=u))
                predictor=RecurrentPredictor(model)
                with torch.inference_mode():
                    report=evaluate(predictor,4,metric_ring=True,geometric_return=True,episode_reset=predictor.reset_episode)
                save(f'evaluation_{u}.json',report)
                suites=report['suites']
                print(u,{k:v['successful_episodes'] for k,v in suites.items()},flush=True)
                if suites['original_centered']['successful_episodes']<22 or suites['varied_approach']['successful_episodes']<20:
                    reason='single_pass_retention_failed'
                    break
        save('status.json',dict(mode='stopped',update=u,reason=reason,mission_accepted=False,frozen_parameters_unchanged=frozen_hash()==frozen_before))
    except BaseException as exc:
        save('status.json',dict(mode='failed',error=repr(exc),mission_accepted=False))
        raise

if __name__=='__main__':
    main()
