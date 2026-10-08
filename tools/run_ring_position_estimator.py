"""Offline causal ring-relative-position estimation; never a flight controller."""
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

from tools import run_recurrent_estimator as core
from tools import run_checkpoint_reset_trial as recording

PLAN='experiments/faisal/docs/RING_POSITION_ESTIMATOR_TRIAL_20261008.md'
SEED=11597000


def position_targets(trace):
    observations,_,mask,visible=core.episode_arrays(trace)
    centre=np.asarray(trace['hoop_centre'],np.float64)
    if centre.shape!=(3,) or not np.isfinite(centre).all():raise ValueError('Finite ring-centre label geometry required')
    position=np.asarray(trace['positions'],np.float64)
    heading=np.asarray(trace['headings'],np.float64)
    offset=centre-position[1:-1]
    c,s=np.cos(heading[1:-1]),np.sin(heading[1:-1])
    x,y,z=offset.T
    targets=np.zeros((len(observations),3),np.float32)
    targets[2:]=np.column_stack((c*x+s*y,s*x-c*y,z))*1000
    return observations,targets,mask,visible


def retarget(episodes,deadline,stop):
    for ep in episodes:
        core.ensure_budget(deadline,stop)
        with np.load(ep.trace,allow_pickle=False) as trace:
            obs,targets,mask,visible=position_targets(trace)
        if not np.array_equal(obs,ep.observations) or not np.array_equal(mask,ep.mask):raise ValueError('Retargeting altered inputs or split')
        ep.targets=targets


def duration_metrics(path):
    with np.load(path,allow_pickle=False) as z:
        durations=[];span=0
        for step,visible in zip(z['steps'],z['visible']):
            if step==0:span=0
            span=0 if visible else span+1;durations.append(span)
        durations=np.asarray(durations);error=z['predictions']-z['targets'];valid=z['mask']
        bins={'visible':durations==0,'blind_1_8':(durations>=1)&(durations<=8),
              'blind_9_24':(durations>=9)&(durations<=24),'blind_25_plus':durations>=25}
        return {name:dict(samples=int((valid&mask).sum()),rmse_mm_per_axis=float(np.sqrt(np.mean(error[valid&mask]**2)))
                         if (valid&mask).any() else None) for name,mask in bins.items()}


def run(args):
    out=args.out.resolve();started=time.monotonic();seconds=180 if args.smoke else 1800
    deadline=started+seconds;stop=out/'STOP';models={};optimizers={};completed={};saved=set();normalizers=None
    results={};inputs={};sources={};status='preparing';error=None;protocol_hash=None
    source=core.ROOT/'trainedModels'/f'{core.PROFILE.name}.zip'
    if core.sha(source)!=core.PROFILE.sha256:raise ValueError('Original source changed')
    with core.single_trainer(core.ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        try:
            recording.status(out,'preparing')
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():raise RuntimeError('Local CUDA required')
            torch.cuda.set_per_process_memory_fraction(.15,device=0)
            torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
            # Reuse the previously verified dataset provenance, but verify each consumed trace again.
            old=Path('D:/DroneSoccerData/runs')
            provenance=old/'recurrent-estimator-20261008/protocol.json'
            previous=json.loads(provenance.read_text());inputs[str(provenance)]=core.sha(provenance)
            if previous['source_policy']['sha256']!=core.PROFILE.sha256:raise ValueError('Wrong historical source')
            for path,digest in previous['input_sha256'].items():
                core.ensure_budget(deadline,stop)
                if core.sha(path)!=digest:raise ValueError('Historical dataset changed: '+path)
                inputs[path]=digest
            warm,wr,hashes=core.load_episodes(core.WARMUP_RUN/'clearance/training_trajectories',warmup_limit=200000,
                max_episodes=10 if args.smoke else None,deadline=deadline,stop_path=stop)
            inputs.update(hashes)
            gae,gr,hashes=core.load_episodes(core.GAE_RUN/'trajectories',expected_total=240,
                max_episodes=4 if args.smoke else None,deadline=deadline,stop_path=stop)
            inputs.update(hashes)
            retarget(warm,deadline,stop);retarget(gae,deadline,stop)
            groups=np.repeat(np.arange(len(warm)),[int(ep.mask.sum()) for ep in warm])
            with np.load(core.PRIOR_SPLIT,allow_pickle=False) as z:
                expected=z['episode_ids'][:len(groups)] if args.smoke else z['episode_ids']
                if not np.array_equal(groups,expected):raise ValueError('Historical split changed')
            if set(ep.seed for ep in warm)&set(ep.seed for ep in gae):raise ValueError('Overlapping cohorts')
            if not args.smoke and (len(warm)!=558 or [ep.seed for ep in gae]!=list(range(11241000,11241240))):raise ValueError('Unexpected data cohort')
            fit=np.asarray([i for i in range(len(warm)) if i%5!=4]);held=np.asarray([i for i in range(len(warm)) if i%5==4])
            normalizers=core.fit_normalizers([warm[i] for i in fit]);np.savez_compressed(out/'normalizers.npz',**normalizers)
            total=2 if args.smoke else 3000
            schedule=np.random.default_rng(SEED).choice(fit,size=(total,8),replace=True);np.save(out/'schedule.npy',schedule)
            np.savez_compressed(out/'split.npz',fit=fit,heldout=held,episode_ids=groups)
            core.write_json(out/'episodes.json',dict(warmup=wr,gae=gr))
            paths=list((core.ROOT/'src').glob('*.py'))+[Path(__file__).resolve(),core.ROOT/'tools/run_recurrent_estimator.py',
                core.ROOT/'tools/run_clearance_experiment.py',core.ROOT/'tools/run_checkpoint_reset_trial.py',
                core.ROOT/'tests/test_ring_position_estimator.py',core.ROOT/'tests/test_recurrent_estimator.py',core.ROOT/PLAN]
            for path in paths:
                rel=path.relative_to(core.ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest);sources[str(rel)]=core.sha(dest)
            protocol=dict(kind='Offline pre-action ring-relative position',smoke=args.smoke,seed=SEED,updates=total,
                whole_episode_batch_size=8,full_episode_unroll=True,adam_lr=.0003,gradient_norm_clip=10,
                inputs='Actual public85 only; no geometry, truth, current/future action',target='Body-frame hoop_centre - positions[t-1] at headings[t-1], mm, t>=2',
                normalizers='FIT-only',input_sha256=inputs,sources=sources,source_policy_sha256=core.PROFILE.sha256,
                runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),device='cuda:0',
                strict_float32=True,memory_fraction=.15,threads=1),max_seconds=seconds,
                actor_updated=False,acceptance=False,physical_flight=False)
            core.write_json(out/'protocol.json',protocol);protocol_hash=core.sha(out/'protocol.json')
            for variant in core.VARIANTS:
                random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
                model=(core.FeedForwardEstimator() if variant=='feedforward' else core.RecurrentEstimator()).to('cuda:0')
                optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
                models[variant]=model;optimizers[variant]=optimizer;completed[variant]=0;model.train()
                for update,ids in enumerate(schedule,1):
                    core.ensure_budget(deadline,stop)
                    x,y,mask,lengths=core.make_batch(warm,ids,normalizers,'cuda:0')
                    loss=core.masked_mse(model(x,lengths),y,mask)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),10)
                    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
                    optimizer.step();completed[variant]=update
                    if update%100==0 or update==total:recording.status(out,'training',variant=variant,updates=update,prescribed=total)
                core.save_state(out/f'{variant}.pt',model,optimizer,completed[variant],normalizers);saved.add(variant)
            for variant in core.VARIANTS:
                results[variant]={};recording.status(out,'evaluating',variant=variant)
                for label,episodes,ids in [('fit',warm,fit),('heldout',warm,held),('gae_separate',gae,np.arange(len(gae)))]:
                    results[variant][label]=core.evaluate_estimator(models[variant],episodes,ids,normalizers,
                        out/f'{variant}-{label}-predictions.npz','cuda:0',deadline=deadline,stop_path=stop)
                    results[variant][label]['blind_duration_bins']=duration_metrics(out/f'{variant}-{label}-predictions.npz')
                core.write_json(out/'results.json',results)
            status='complete';core.ensure_budget(deadline,stop)
        except core.BoundedStop as exc:status='stopped';error=str(exc)
        except Exception as exc:status='failed';error=repr(exc);(out/'error.txt').write_text(traceback.format_exc())
        finally:
            for variant,model in models.items():
                if variant not in saved:core.save_state(out/f'{variant}-partial.pt',model,optimizers[variant],completed[variant],normalizers)
            try:
                if protocol_hash is None or core.sha(out/'protocol.json')!=protocol_hash:raise ValueError('Missing/changed protocol')
                for path,digest in inputs.items():
                    if core.sha(path)!=digest:raise ValueError('Changed input '+path)
                for path,digest in sources.items():
                    if core.sha(core.ROOT/path)!=digest or core.sha(out/'source'/path)!=digest:raise ValueError('Changed source '+path)
                integrity=True
            except Exception as exc:status='failed';error=repr(exc);integrity=False
            decision=core.gate_estimator(results,complete=status=='complete' and integrity,
                source_unchanged=core.sha(source)==core.PROFILE.sha256,smoke=args.smoke)
            decision.update(status=status,error=error,updates_completed=completed,elapsed_seconds=time.monotonic()-started,
                target='ring-relative position in body frame, millimetres',integrity_verified=integrity,actor_updated=False)
            recording.status(out,status,updates_completed=completed,error=error)
            decision['evidence']={str(p.relative_to(out)):core.sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('decision.json','STOP')}
            core.write_json(out/'decision.json',decision)
            print(json.dumps({k:v for k,v in decision.items() if k!='evidence'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())
