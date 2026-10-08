"""Bounded public-history system identification; no flight actor or simulator steps."""
import argparse
import json
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch
from torch import nn

from src.sim import HoopEnv
from src.approach_curriculum import ApproachCurriculumEnv
from src.camera_history import CameraHistory
from tools import run_recurrent_estimator as core

SEED = 11601000
PLAN = 'experiments/faisal/docs/DYNAMICS_IDENTIFICATION_TRIAL_20261008.md'


def reset_label(episode, env):
    """Recreate only a verified reset, never a simulated movement or future label."""
    env.reset(seed=episode.seed)
    with np.load(episode.trace, allow_pickle=False) as saved:
        first = CameraHistory(core.PROFILE.lags).update(env._frames[-1], np.zeros(4, np.float32))
        if not np.array_equal(first, episode.observations[0]):
            raise ValueError('Reset public observation differs from saved episode')
        for key, actual in [('hoop_centre', env.hoop_centre), ('hoop_normal', env.hoop_normal),
                            ('drone_half_box', env._half_box)]:
            if not np.array_equal(saved[key], actual):
                raise ValueError('Reset geometry differs from saved episode: '+key)
    return np.asarray([env.dyn.delay_steps, env.dyn.tau], np.float32)


def verify_reset_sources(run, curriculum, inputs):
    path = run/'protocol.json'
    protocol = json.loads(path.read_text()); inputs[str(path)] = core.sha(path)
    mapping = protocol['sources' if curriculum else 'source_hashes']
    required = ['src/sim.py', 'src/params.py', 'src/contract.py']
    if curriculum: required.append('src/approach_curriculum.py')
    mapping = {k.replace('\\', '/'): v for k, v in mapping.items()}
    for relative in required:
        if core.sha(core.ROOT/relative) != mapping[relative]:
            raise ValueError('Historical reset implementation changed: '+relative)


def batch(episodes, labels, ids, norm, device):
    lengths = np.asarray([len(episodes[i].observations) for i in ids])
    x = np.zeros((len(ids), int(max(lengths)), 85), np.float32)
    for j, i in enumerate(ids):
        x[j, :lengths[j]] = (episodes[i].observations-norm['x_mean'])/norm['x_scale']
    return (torch.as_tensor(x, device=device),
            torch.as_tensor((labels[ids]-norm['y_mean'])/norm['y_scale'], device=device),
            torch.as_tensor(lengths))


def episode_loss(pred, targets, lengths):
    """Equal episode weight; loss after 32 observed transitions, causal prefix only."""
    terms = [(pred[j, 32:int(n)]-targets[j]).square().mean()
             for j, n in enumerate(lengths) if int(n) > 32]
    if not terms: raise ValueError('No episode reaches the declared observation burn-in')
    return torch.stack(terms).mean()


def metrics(predictions, targets, episode_ids, steps, prior):
    ids = np.unique(episode_ids[steps >= 32])
    if not len(ids): raise ValueError('No evaluable whole episodes')
    errors, prior_errors, acc = [], [], []
    for eid in ids:
        use = (episode_ids == eid) & (steps >= 32)
        errors.append(np.mean((predictions[use]-targets[use])**2, axis=0))
        prior_errors.append(np.mean((prior-targets[use])**2, axis=0))
        acc.append(np.mean(np.rint(predictions[use, 0]) == targets[use, 0]))
    return dict(episodes=int(len(ids)), total_episodes=int(len(np.unique(episode_ids))),
                rows=int((steps >= 32).sum()), rmse=np.sqrt(np.mean(errors, axis=0)).tolist(),
                prior_rmse=np.sqrt(np.mean(prior_errors, axis=0)).tolist(),
                delay_rounded_accuracy=float(np.mean(acc)))


def gate(results, complete, smoke):
    reasons = []; comparisons = {}
    if smoke: reasons.append('Software smoke only')
    if not complete: reasons.append('Incomplete budgets or failed integrity')
    for cohort in ('heldout', 'gae_separate'):
        if any(cohort not in results.get(v, {}) for v in core.VARIANTS):
            reasons.append('Missing final cohort: '+cohort); continue
        ff, gr = (results[v][cohort] for v in core.VARIANTS)
        comparisons[cohort] = {}
        for axis, name in enumerate(('delay_steps', 'tau_seconds')):
            a, b, prior = ff['rmse'][axis], gr['rmse'][axis], gr['prior_rmse'][axis]
            valid = np.isfinite([a,b,prior]).all() and a > 0 and prior > 0 and b >= 0
            comparisons[cohort][name] = dict(feedforward=a, recurrent=b, fit_mean_prior=prior)
            if not valid or b > .85*a or b > .80*prior:
                reasons.append(cohort+': '+name+' fails 15% FF / 20% FIT-prior improvement')
    return dict(passed=not reasons, reasons=reasons, comparisons=comparisons,
                decision='supports_separate_integration_review' if not reasons else 'close_fixed_identification_trial',
                actor_updated=False, acceptance=False, physical_flight=False)


def evaluate(model, episodes, labels, ids, norm, out, check):
    values = {k: [] for k in ('predictions','targets','episode_ids','seeds','steps')}
    model.eval()
    with torch.no_grad():
        for start in range(0,len(ids),8):
            check(); selected=ids[start:start+8]
            x, _, lengths=batch(episodes,labels,selected,norm,'cuda:0')
            predicted=model(x,lengths).cpu().numpy()*norm['y_scale']+norm['y_mean']
            for j,i in enumerate(selected):
                n=len(episodes[i].observations)
                values['predictions'].append(predicted[j,:n])
                values['targets'].append(np.broadcast_to(labels[i],(n,2)))
                values['episode_ids'].append(np.full(n,i,np.int64))
                values['seeds'].append(np.full(n,episodes[i].seed,np.int64))
                values['steps'].append(np.arange(n))
    values={k:np.concatenate(v) for k,v in values.items()}
    np.savez_compressed(out,**values)
    return metrics(*(values[k] for k in ('predictions','targets','episode_ids','steps')),norm['y_mean'])


def run(args):
    out=args.out.resolve(); started=time.monotonic(); deadline=started+(180 if args.smoke else 1800)
    def check(): core.ensure_budget(deadline,out/'STOP')
    def status(phase, **kwargs): core.write_json(out/'status.json',dict(status=phase,**kwargs))
    inputs={};sources={};models={};optimizers={};done={};results={};norm=None;protocol_hash=None
    state='preparing';error=None;integrity=False
    with core.single_trainer(core.ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False);status(state)
        try:
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            torch.cuda.set_per_process_memory_fraction(.15,device=0)
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            # Bind already preserved dataset provenance; actor data are not changed or replayed.
            prior_path=Path('D:/DroneSoccerData/runs/ring-position-estimator-20261008/protocol.json')
            prior=json.loads(prior_path.read_text());inputs[str(prior_path)]=core.sha(prior_path)
            for path,h in prior['input_sha256'].items():
                check()
                if core.sha(path)!=h:raise ValueError('Historical input changed: '+path)
                inputs[path]=h
            datasets={};labels={};records={}
            for name,run,folder,curriculum in [('warmup',core.WARMUP_RUN,'clearance/training_trajectories',False),
                                              ('gae_separate',core.GAE_RUN,'trajectories',True)]:
                verify_reset_sources(run,curriculum,inputs)
                eps,rec,h=core.load_episodes(run/folder,warmup_limit=None if curriculum else 200000,
                    expected_total=240 if curriculum else None,max_episodes=(4 if curriculum else 10) if args.smoke else None,
                    deadline=deadline,stop_path=out/'STOP');inputs.update(h)
                env=(ApproachCurriculumEnv if curriculum else HoopEnv)(level=1,randomise=True,hanger=True)
                try:
                    lab=[]
                    for ep in eps:check();lab.append(reset_label(ep,env))
                finally:env.close()
                datasets[name]=eps;labels[name]=np.asarray(lab);records[name]=rec
            warm=datasets['warmup'];fit=np.asarray([i for i in range(len(warm)) if i%5!=4]);held=np.asarray([i for i in range(len(warm)) if i%5==4])
            eligible=fit[[len(warm[i].observations)>32 for i in fit]]
            if len(eligible)<4:raise ValueError('Insufficient whole FIT episodes after32 frames')
            x=np.concatenate([warm[i].observations for i in fit]);y=labels['warmup'][fit]
            norm=dict(x_mean=x.mean(0,dtype=np.float64).astype(np.float32),x_scale=np.maximum(x.std(0,dtype=np.float64),.01).astype(np.float32),
                      y_mean=y.mean(0,dtype=np.float64).astype(np.float32),y_scale=np.maximum(y.std(0,dtype=np.float64),.001).astype(np.float32))
            total=2 if args.smoke else 3000;schedule=np.random.default_rng(SEED).choice(eligible,(total,8),replace=True)
            np.save(out/'schedule.npy',schedule);np.savez_compressed(out/'normalizers.npz',**norm)
            np.savez_compressed(out/'labels.npz',**labels,fit=fit,heldout=held,eligible_fit=eligible)
            core.write_json(out/'episodes.json',records)
            files=list((core.ROOT/'src').glob('*.py'))+[Path(__file__),core.ROOT/'tools/run_recurrent_estimator.py',
                core.ROOT/'tools/run_clearance_experiment.py',core.ROOT/'tests/test_dynamics_identification.py',core.ROOT/PLAN]
            for path in files:
                relative=path.resolve().relative_to(core.ROOT);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest);sources[relative.as_posix()]=core.sha(dest)
            protocol=dict(kind='Public-history dynamics identification, no flight actor',smoke=args.smoke,seed=SEED,updates=total,
                targets=['delay_steps','tau_seconds'],burn_in_frames=32,whole_episode_loss_weighting=True,
                fitting='Original warmup four-of-five whole episodes only; no GAE/heldout fitting',
                reset_labels='Historical source hashes and exact first public observation/geometry; reset only, no step',
                inputs='Original public85; no seed/truth/action-now/future inputs',input_sha256=inputs,sources=sources,
                runtime=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
                             memory_fraction=.15,threads=1,strict_float32=True),max_seconds=180 if args.smoke else 1800)
            core.write_json(out/'protocol.json',protocol);protocol_hash=core.sha(out/'protocol.json')
            for variant in core.VARIANTS:
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
                model=core.FeedForwardEstimator() if variant=='feedforward' else core.RecurrentEstimator()
                if variant=='feedforward':model.network[-1]=nn.Linear(128,2)
                else:model.output=nn.Linear(64,2)
                model=model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
                models[variant]=model;optimizers[variant]=optimizer;done[variant]=0;model.train()
                for step,ids in enumerate(schedule,1):
                    check();x,y,lengths=batch(warm,labels['warmup'],ids,norm,'cuda:0')
                    loss=episode_loss(model(x,lengths),y,lengths)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    optimizer.zero_grad();loss.backward();g=nn.utils.clip_grad_norm_(model.parameters(),10)
                    if not torch.isfinite(g):raise ValueError('Nonfinite gradients')
                    optimizer.step();done[variant]=step
                    if step%100==0 or step==total:status('training',variant=variant,updates=step,total=total)
                core.save_state(out/f'{variant}.pt',model,optimizer,step,norm)
            for variant,model in models.items():
                results[variant]={};status('evaluating',variant=variant)
                for cohort,eps,lab,ids in [('fit',warm,labels['warmup'],fit),('heldout',warm,labels['warmup'],held),
                    ('gae_separate',datasets['gae_separate'],labels['gae_separate'],np.arange(len(datasets['gae_separate'])))]:
                    results[variant][cohort]=evaluate(model,eps,lab,ids,norm,out/f'{variant}-{cohort}.npz',check)
            core.write_json(out/'results.json',results);state='complete'
        except Exception as exc:
            state='stopped' if isinstance(exc,core.BoundedStop) else 'failed';error=repr(exc)
            (out/'error.txt').write_text(traceback.format_exc())
        finally:
            for variant,model in models.items():
                if not (out/f'{variant}.pt').exists():core.save_state(out/f'{variant}-partial.pt',model,optimizers[variant],done[variant],norm)
            try:
                if protocol_hash is None or core.sha(out/'protocol.json')!=protocol_hash:raise ValueError('Missing/changed protocol')
                for path,h in inputs.items():
                    if core.sha(path)!=h:raise ValueError('Input changed: '+path)
                for rel,h in sources.items():
                    if core.sha(core.ROOT/rel)!=h or core.sha(out/'source'/rel)!=h:raise ValueError('Source changed: '+rel)
                integrity=True
            except Exception as exc:state='failed';error=repr(exc)
            decision=gate(results,state=='complete' and integrity,args.smoke)
            decision.update(status=state,error=error,updates_completed=done,integrity_verified=integrity,elapsed_seconds=time.monotonic()-started)
            status(state,error=error,updates_completed=done)
            decision['evidence']={str(p.relative_to(out)):core.sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('decision.json','STOP')}
            core.write_json(out/'decision.json',decision)
            print(json.dumps({k:v for k,v in decision.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
