"""Four-update CUDA reward-only policy-gradient diagnostic; no promotion."""
from pathlib import Path
import sys, json, pickle, hashlib, shutil
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT/'diagnostics/recurrent-return-histories-20260928'
OUT = ROOT/'diagnostics/authentic-return-pg-guarded-20260928'
sys.path.insert(0, str(BANK/'source_snapshot'))
import src
src.__path__.append(str(ROOT/'src'))
from src.recurrent_snapshot_rollout import collect_segment, evaluate_segment
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate
from src.recurrent_update_guard import distributions, guarded_step

def write(name, value):
    def numpy_json(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError(type(item).__name__)
    (OUT/name).write_text(json.dumps(value, indent=2, default=numpy_json))

def main():
    OUT.mkdir(exist_ok=False)
    shutil.copytree(BANK/'source_snapshot', OUT/'source_snapshot')
    for name in ('recurrent_burnin.py','recurrent_snapshot_rollout.py','recurrent_update_guard.py'):
        shutil.copy2(ROOT/'src'/name, OUT/'source_snapshot/src'/name)
    shutil.copy2(__file__, OUT/'runner.py')
    source = Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    entries = [x for x in json.loads((BANK/'result.json').read_text())['snapshots'] if x['phase']=='return_side'][:8]
    entries = [dict(x, bank=str(BANK)) for x in entries]
    reapproach_bank = ROOT/'diagnostics/authentic-reapproaches-20260928'
    entries += [dict(x, bank=str(reapproach_bank)) for x in json.loads((reapproach_bank/'result.json').read_text())['snapshots']]
    write('protocol.json', dict(source=str(source),sha256=source_hash,snapshots=entries,
        algorithm='On-policy Monte Carlo reward-to-go policy gradient, one batch update per fifteen full continuations',
        max_updates=4,max_training_steps=36000,lr=1e-5,gamma=.999,
        kl_guard=dict(mean_limit=.001,max_limit=.01,reference='Source distribution on all first-batch histories, fixed throughout; own-weight memory reconstructed',rejection='Restore parameters and optimizer state; stop trial'),
        reward_scale=1.,advantage='Batch-centered and standardized discounted reward-to-go; no critic baseline',
        rationale='Prior mixed trial action RMS change .000422; test 10x learning rate with cumulative fixed-history KL bounds and unchanged retention checks',
        burnin='Detached current-weight real prefix before each likelihood evaluation',
        seed_rule='983000+100*update+snapshot_index',finite_mission_terminal=True,
        retention_stop='centered <24/25 or varied <20/25',checks=[0,1,2,3,4],
        acceptance=False,reserved_seeds_used=False))
    torch.set_num_threads(1)
    model = RecurrentPPO.load(source,device='cuda')
    assert model.device.type=='cuda'
    optimizer = torch.optim.Adam(model.policy.parameters(),lr=1e-5)
    guard_segments=None;reference=None
    reports=[];train_steps=0
    for update in range(5):
        predictor=RecurrentPredictor(model)
        report=evaluate(predictor,4,metric_ring=True,geometric_return=True,episode_reset=predictor.reset_episode)
        write(f'evaluation_{update}.json',report)
        summary={k:v['successful_episodes'] for k,v in report['suites'].items()}
        reports.append(dict(update=update,success=summary,training_steps=train_steps))
        print(json.dumps(reports[-1]),flush=True)
        model.save(OUT/f'model_{update}')
        if summary['original_centered']<24 or summary['varied_approach']<20:
            stop='single-pass retention stop';break
        if update==4:
            stop='bounded four-update budget complete';break
        segments=[];targets=[];rows=[]
        for i,entry in enumerate(entries):
            payload=(Path(entry['bank'])/'snapshots'/f"{entry['id']}.pkl").read_bytes()
            assert hashlib.sha256(payload).hexdigest()==entry['sha256']
            d=pickle.loads(payload);env=d['environment'];initial_passes=env.unwrapped.passes
            torch.manual_seed(983000+100*update+i)
            seg=collect_segment(model,env,d['observation'],d['prefix_observations'],600)
            assert seg.mission_ended
            env.close();segments.append(seg)
            returns=np.zeros(len(seg.rewards),dtype=np.float32);carry=0.
            for j in range(len(returns)-1,-1,-1):
                carry=seg.rewards[j]+.999*carry;returns[j]=carry
            targets.append(returns);train_steps+=len(returns)
            np.savez_compressed(OUT/f'rollout_{update}_{i}.npz',prefix=seg.prefix,
                observations=seg.observations,raw_actions=seg.actions,rewards=seg.rewards,
                log_probs=seg.log_probs,returns=returns,next_observation=seg.next_observation)
            rows.append(dict(id=entry['id'],steps=len(returns),initial_passes=initial_passes,final_info=seg.final_info))
        all_targets=np.concatenate(targets);mean=float(all_targets.mean());std=float(all_targets.std())+1e-8
        optimizer.zero_grad();loss_value=0.;max_log_error=0.
        for seg,returns in zip(segments,targets):
            _,logs,_=evaluate_segment(model,seg)
            max_log_error=max(max_log_error,float(np.max(np.abs(logs.detach().cpu().numpy()-seg.log_probs))))
            weights=torch.as_tensor((returns-mean)/std,device='cuda')
            loss=-(logs*weights).sum()/len(all_targets)
            assert torch.isfinite(loss)
            loss.backward();loss_value+=float(loss.item())
        assert max_log_error<1e-3,max_log_error
        assert all(torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None)
        grad=float(torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5))
        if guard_segments is None:
            guard_segments=segments
            reference=distributions(model,guard_segments)
            torch.save(reference,OUT/'guard_reference.pt')
        guard=guarded_step(model,optimizer,guard_segments,reference)
        write(f'update_{update+1}.json',dict(rows=rows,loss=loss_value,return_mean=mean,return_std=std,
            preupdate_log_error=max_log_error,grad_norm=grad,training_steps=train_steps,guard=guard))
        if not guard['accepted']:
            stop='KL guard rejected proposal and restored weights/optimizer';break
        write('status.json',dict(state='running',completed_updates=update+1,training_steps=train_steps))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    write('result.json',dict(stop=stop,reports=reports,training_steps=train_steps,source_unchanged=True,accepted=False))
    write('status.json',dict(state='finished',stop=stop,training_steps=train_steps))

if __name__=='__main__':main()
