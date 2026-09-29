"""Bounded supervised FIT-capacity diagnostic, not an acceptance experiment."""
from pathlib import Path
import sys, json, hashlib, shutil
import numpy as np
import torch

OLD=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
OUT=Path(__file__).parent/'lap-capacity-20260928'
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
    params=[p for n,p in model.policy.named_parameters() if n.startswith(('lstm_actor.','mlp_extractor.policy_net.','action_net.'))]
    opt=torch.optim.Adam(params,lr=1e-4)
    save('protocol.json',dict(kind='Supervised FIT-capacity diagnostic; not RL or acceptance',
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        seeds=seeds,device=torch.cuda.get_device_name(),max_updates=100,checks=[25,50,100],
        lr=1e-4,loss='all nine complete 600-step learned teacher histories; no basic rehearsal',
        purpose='Joint capacity/optimization diagnostic; cannot attribute effects to one hyperparameter',
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
            loss=(d.distribution.mean.reshape(9,600,4)-targets).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,.5)
            opt.step()
            history.append(dict(update=u,pre_update_lap_mse=float(loss.detach())))
            save('history.json',history)
            save('status.json',dict(mode='supervised capacity diagnostic',update=u))
            if u in (25,50,100):
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
        save('status.json',dict(mode='stopped',update=u,reason=reason,mission_accepted=False))
    except BaseException as exc:
        save('status.json',dict(mode='failed',error=repr(exc),mission_accepted=False))
        raise

if __name__=='__main__':
    main()
