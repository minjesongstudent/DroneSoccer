"""Four-update CUDA reward-only policy-gradient diagnostic; no promotion."""
from pathlib import Path
import sys, json, pickle, hashlib, shutil
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT/'diagnostics/latency-return-histories-20260929'
OUT = ROOT/'diagnostics/latency-bank-actor-20260929'
sys.path.insert(0, str(BANK/'source_snapshot'))
import src
src.__path__.append(str(ROOT/'src'))
from src.recurrent_snapshot_rollout import collect_segment, collect_natural_segment, evaluate_segment
from src.sim import HoopEnv
from src.params import Drone
from src.curriculum import LessonEnv
from src.return_reset_curriculum import wrap
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate
from src.recurrent_update_guard import distributions, guarded_step
from src.independent_return_baseline import leave_one_out_baselines
from src.return_objective_mask import reinforcement_coefficient

def write(name, value):
    def numpy_json(item):
        if isinstance(item,np.ndarray):return item.tolist()
        if isinstance(item,np.generic):return item.item()
        raise TypeError(type(item).__name__)
    (OUT/name).write_text(json.dumps(value, indent=2, default=numpy_json))

def main():
    OUT.mkdir(exist_ok=False)
    shutil.copytree(BANK/'source_snapshot', OUT/'source_snapshot')
    for name in ('recurrent_burnin.py','recurrent_snapshot_rollout.py','recurrent_update_guard.py','independent_return_baseline.py','return_objective_mask.py'):
        shutil.copy2(ROOT/'src'/name, OUT/'source_snapshot/src'/name)
    shutil.copy2(__file__, OUT/'runner.py')
    source = Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    entries=[dict(x,bank=str(BANK)) for x in json.loads((BANK/'result.json').read_text())['snapshots'] if x['split']=='fit']
    assert len(entries)==30 and all(e['seed']%4!=0 for e in entries)
    entries += [dict(id=f'natural_full_{i}',natural='full',delay=0 if i<4 else 2) for i in range(8)]
    entries += [dict(id=f'natural_basic_{i}',natural='basic',stage=i//2) for i in range(4)]
    entries=[dict(e,group=i,repeat=r) for i,e in enumerate(entries) for r in range(3)]
    write('protocol.json', dict(source=str(source),sha256=source_hash,snapshots=entries,
        algorithm='On-policy Monte Carlo reward-to-go policy gradient, one batch update per 126 flights/continuations with three independent repeats per start',
        max_updates=2,max_training_steps=151200,lr=1e-5,gamma=.999,
        natural_reset_seed_rule='2991000+100*update+group_id; eight full (four delayed), four basic (two centered, two varied)',
        kl_guard=dict(mean_limit=.001,max_limit=.01,reference='Source distribution on all first-batch histories, fixed throughout; own-weight memory reconstructed',rejection='Restore parameters and optimizer state; stop trial'),
        basic_reinforcement_gradient=False,basic_guard_and_validation=True,loss_denominator=126,paired_seed_comparison='New training seeds, changed authentic nominal/delayed FIT bank',reward_scale=1.,advantage='Time-indexed mean return of other two trajectories from same start; absorbing zero after peers terminate; fixed scale40; gamma^t objective weighting; baseline detached',
        rationale='Test independent-trajectory baseline after critic calibration failed; FIT authentic returns and new natural/basic starts; fresh on-policy batch per update',
        burnin='Detached current-weight real prefix before each likelihood evaluation',
        seed_rule='997000+1000*update+entry_index',finite_mission_terminal=True,
        retention_stop='centered <24/25 or varied <20/25',checks=[0,1,2],
        acceptance=False,reserved_seeds_used=False))
    torch.set_num_threads(1)
    model = RecurrentPPO.load(source,device='cuda')
    assert model.device.type=='cuda'
    optimizer = torch.optim.Adam(model.policy.parameters(),lr=1e-5)
    guard_segments=None;reference=None
    reports=[];train_steps=0
    for update in range(3):
        predictor=RecurrentPredictor(model)
        report=evaluate(predictor,4,metric_ring=True,geometric_return=True,episode_reset=predictor.reset_episode)
        write(f'evaluation_{update}.json',report)
        summary={k:v['successful_episodes'] for k,v in report['suites'].items()}
        reports.append(dict(update=update,success=summary,training_steps=train_steps))
        print(json.dumps(reports[-1]),flush=True)
        model.save(OUT/f'model_{update}')
        if summary['original_centered']<24 or summary['varied_approach']<20:
            stop='single-pass retention stop';break
        if update==2:
            stop='bounded two-update budget complete';break
        segments=[];targets=[];rows=[]
        for i,entry in enumerate(entries):
            torch.manual_seed(997000+1000*update+i)
            if 'natural' in entry:
                base=HoopEnv(level=1.,drone=Drone(delay_steps=entry['delay'])) if entry['natural']=='full' else LessonEnv(stage=entry['stage'])
                env=wrap(base,single=entry['natural']=='basic',metric_ring=True,geometric_return=True)
                initial_passes=0
                seg=collect_natural_segment(model,env,2991000+100*update+entry['group'],600)
            else:
                payload=(Path(entry['bank'])/'snapshots'/f"{entry['id']}.pkl").read_bytes()
                assert hashlib.sha256(payload).hexdigest()==entry['sha256']
                d=pickle.loads(payload);env=d['environment'];initial_passes=env.unwrapped.passes
                seg=collect_segment(model,env,d['observation'],d['prefix_observations'],600)
            assert seg.mission_ended
            env.close();segments.append(seg)
            returns=np.zeros(len(seg.rewards),dtype=np.float32);carry=0.
            for j in range(len(returns)-1,-1,-1):
                carry=seg.rewards[j]+.999*carry;returns[j]=carry
            targets.append(returns);train_steps+=len(returns)
            np.savez_compressed(OUT/f'rollout_{update}_{i}.npz',prefix=seg.prefix,
                observations=seg.observations,raw_actions=seg.actions,rewards=seg.rewards,
                log_probs=seg.log_probs,returns=returns,next_observation=seg.next_observation,natural_start=seg.natural_start)
            rows.append(dict(id=entry['id'],group=entry['group'],repeat=entry['repeat'],steps=len(returns),initial_passes=initial_passes,final_info=seg.final_info))
        all_targets=np.concatenate(targets);mean=float(all_targets.mean());std=float(all_targets.std())+1e-8
        baselines=leave_one_out_baselines(targets,[e['group'] for e in entries])
        for k in range(0,len(segments),3):
            assert all(np.array_equal(segments[k].observations[0],segments[j].observations[0]) and np.array_equal(segments[k].prefix,segments[j].prefix) for j in (k+1,k+2))
        optimizer.zero_grad();loss_value=0.;max_log_error=0.
        for entry,seg,returns,baseline in zip(entries,segments,targets,baselines):
            _,logs,_=evaluate_segment(model,seg)
            max_log_error=max(max_log_error,float(np.max(np.abs(logs.detach().cpu().numpy()-seg.log_probs))))
            weights=torch.as_tensor((returns-baseline)*(.999**np.arange(len(returns)))/40.,device='cuda',dtype=logs.dtype)
            loss=-reinforcement_coefficient(entry)*(logs*weights).sum()/len(segments)
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
            basic_gradient_count=sum(reinforcement_coefficient(e)!=0 for e in entries if e.get("natural")=="basic"),guard_episodes=len(guard_segments),preupdate_log_error=max_log_error,grad_norm=grad,training_steps=train_steps,guard=guard))
        if not guard['accepted']:
            stop='KL guard rejected proposal and restored weights/optimizer';break
        write('status.json',dict(state='running',completed_updates=update+1,training_steps=train_steps))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    write('result.json',dict(stop=stop,reports=reports,training_steps=train_steps,source_unchanged=True,accepted=False))
    write('status.json',dict(state='finished',stop=stop,training_steps=train_steps))

if __name__=='__main__':main()
