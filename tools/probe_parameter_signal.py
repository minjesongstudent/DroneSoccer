"""Finite FIT-only paired policy-parameter signal check. No automatic promotion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from src.sim import HoopEnv
from tools import run_estimated_motion_trial as shared
from tools import run_checkpoint_reset_trial as checkpoints

PLAN = 'experiments/faisal/docs/PARAMETER_SIGNAL_TRIAL_20261008.md'
GUARD_RUN = shared.RUNS/'authentic-late-reset-20261008'


def score_endpoint(e):
    if (not 1 <= e['absolute_steps'] <= 600 or e['terminated'] == e['truncated']
            or (e['truncated'] and e['absolute_steps'] != 600)
            or e['full_body_passes'] < 0 or e['wall_steps'] < 0):
        raise ValueError('Complete original-horizon endpoint required')
    crash = bool(e['terminated'] or e['wall_steps'] > 0)
    first = e['full_body_passes'] >= 1
    repeat = e['full_body_passes'] >= 2 and e['absolute_steps'] == 600 and not crash
    return dict(score=100*int(repeat)+10*int(first)+min(e['full_body_passes'],5)-20*int(crash),
                first=int(first), repeat=int(repeat), crash=int(crash))


def scaled_direction(direction, bilinear, log_std):
    """Equal-variance Gaussian KL is quadratic in mean displacement."""
    delta = bilinear.double() @ direction.double().T
    kl = (.5*(delta / log_std.double().exp()).square()).sum(dim=1)
    if not torch.isfinite(kl).all() or kl.mean() <= 0 or kl.max() <= 0:
        raise ValueError('Degenerate/nonfinite exploration direction')
    scale = .99*min((.001/float(kl.mean()))**.5, (.05/float(kl.max()))**.5)
    return direction*scale, scale


def signal_decision(records, seeds, directions):
    keys = [(r['head'], r['seed']) for r in records]
    expected = {(h,s) for h in ['zero']+[f'd{d:02d}{sign}' for d in range(directions) for sign in ('p','m')] for s in seeds}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, duplicate or unexpected head/layout outcome')
    by = {(r['head'],r['seed']):r for r in records}
    halves = (seeds[::2], seeds[1::2])
    if not halves[0] or len(halves[0]) != len(halves[1]):
        raise ValueError('Equal nonempty predeclared FIT halves required')
    def counts(head, half):
        return {k: sum(by[head,s][k] for s in half) for k in ('score','first','repeat','crash')}
    pairs=[]; consistent=0; nonzero=0; useful=[]
    for d in range(directions):
        contrasts=[]
        for half in halves:
            contrasts.append((counts(f'd{d:02d}p',half)['score']-counts(f'd{d:02d}m',half)['score'])/len(half))
        if all(v != 0 for v in contrasts):
            nonzero += 1; consistent += int(np.sign(contrasts[0]) == np.sign(contrasts[1]))
        pairs.append(dict(direction=d, half_contrasts=contrasts))
        for sign in ('p','m'):
            head=f'd{d:02d}{sign}'; deltas=[]
            for half in halves:
                base, now=counts('zero',half),counts(head,half)
                deltas.append({k:now[k]-base[k] for k in base})
            if all(v['repeat'] >= 1 and v['first'] >= 0 and v['crash'] <= 0 for v in deltas):
                useful.append(dict(head=head,deltas=deltas))
    agreement=consistent/nonzero if nonzero else None
    return dict(pairs=pairs,nonzero_both_halves=nonzero,agreeing_directions=consistent,
                sign_agreement=agreement,useful_heads=useful,
                signal_passed=bool(nonzero>=6 and agreement>=.75 and len(useful)>=2),
                baseline=[counts('zero',h) for h in halves])


def config(smoke):
    return dict(smoke=smoke, training_seeds=list(range(11245920,11245922) if smoke else range(11246000,11246032)),
                directions=2 if smoke else 12, algorithm_seed=11596900 if smoke else 11596000,
                seconds=300 if smoke else 1800, memory_fraction=.15)


def freeze(out, budget):
    paths=list((shared.ROOT/'src').glob('*.py'))+list((shared.ROOT/'tools').glob('*.py'))+[
        shared.ROOT/PLAN,shared.ROOT/'tests/test_parameter_signal.py']
    hashes={}
    for path in paths:
        budget.check(); rel=path.relative_to(shared.ROOT); dest=out/'source'/rel
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dest)
        hashes[rel.as_posix()]=shared.sha(dest)
    return hashes


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv); cfg=config(args.smoke); out=args.out.resolve()
    started=time.monotonic(); budget=shared.Budget(out,cfg['seconds'],started=started)
    result=dict(status='preparing',completed_flights=0,acceptance=False,policy_promoted=False,physical_flight=False)
    policy=None; protocol=None; protocol_digest=None; records=[]
    with shared.single_trainer(shared.ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        try:
            checkpoints.status(out,'preparing',completed_flights=0)
            allocation=checkpoints.scan_allocation([shared.RUNS,shared.ROOT.parent/'lab-data'],cfg['training_seeds'],budget.check)
            shared.write_json(out/'seed_allocation.json',allocation)
            source_hashes=freeze(out,budget)
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
            torch.cuda.set_per_process_memory_fraction(.15,device=0)
            init=shared.verify_initialization(shared.RUNS/'estimated-motion-initialization-20261008/decision.json',budget.check)
            old=json.loads((GUARD_RUN/'protocol.json').read_text())
            guard_path=GUARD_RUN/'guard.npz'
            if shared.sha(guard_path) != old['prepared_sha256']['guard.npz']: raise ValueError('Cached FIT guard changed')
            shutil.copy2(guard_path,out/'guard.npz')
            with np.load(out/'guard.npz',allow_pickle=False) as saved: guard=saved['features'].copy()
            if guard.shape != (8192,89) or not np.isfinite(guard).all(): raise ValueError('Full frozen FIT guard required')
            policy=shared.EstimatedMotionPolicy(shared.SOURCE,shared.ESTIMATOR,device='cuda:0')
            estimator_digest=shared.tensor_digest(policy.motion._model)
            shared.seed_everything(cfg['algorithm_seed'])
            trust=torch.as_tensor(guard,device='cuda')
            with torch.no_grad():
                source_mean,source_std,_=policy.actor.get_action_dist_params(trust)
                context=torch.cat([torch.ones_like(trust[:,:1]),trust[:,:85]],dim=1)
                bilinear=(context[:,:,None]*trust[:,None,85:88]).flatten(1)*trust[:,88:89]
            protocol=dict(kind='paired-deterministic-parameter-signal',configuration=cfg,training_seeds=cfg['training_seeds'],
                shared_fit_seeds='Matched whole-policy comparison only; both halves FIT',sources=source_hashes,
                manifest=shared.checkpoint_manifest(),initialization=init,
                input_sha256={str(guard_path):shared.sha(guard_path),str(GUARD_RUN/'protocol.json'):shared.sha(GUARD_RUN/'protocol.json'),
                              str(shared.SOURCE):shared.sha(shared.SOURCE),str(shared.ESTIMATOR):shared.sha(shared.ESTIMATOR)},
                runtime=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                             precision='strict_float32',threads=1,memory_fraction=.15),
                acceptance=False,policy_promoted=False,physical_flight=False)
            shared.write_json(out/'protocol.json',protocol); protocol_digest=shared.sha(out/'protocol.json')
            (out/'heads').mkdir(); heads={'zero':np.zeros((4,258),np.float32)}; guards={}; raw=[]; scales=[]
            rng=np.random.default_rng(cfg['algorithm_seed'])
            for d in range(cfg['directions']):
                budget.check(); direction=rng.standard_normal((4,258)).astype(np.float32); raw.append(direction)
                adjusted,scale=scaled_direction(torch.as_tensor(direction,device='cuda'),bilinear,source_std)
                weight=adjusted.cpu().numpy(); scales.append(scale)
                heads[f'd{d:02d}p']=weight; heads[f'd{d:02d}m']=-weight
            np.savez_compressed(out/'directions.npz',raw=np.asarray(raw),scales=scales)
            for label,weight in heads.items():
                budget.check()
                with torch.no_grad():
                    policy.actor.residual_weight.copy_(torch.as_tensor(weight,device='cuda'))
                    mean,std,_=policy.actor.get_action_dist_params(trust)
                    kl=shared.gaussian_kl(source_mean,source_std,mean,std)
                metrics=dict(mean_kl=float(kl.mean()),max_kl=float(kl.max()))
                if not np.isfinite(list(metrics.values())).all() or metrics['mean_kl']>.001 or metrics['max_kl']>.05:
                    raise ValueError('Exploration direction failed predeclared KL bounds')
                digest=policy.save(out/'heads'/f'{label}.npz')
                # Decode serialization once, before flight; do not unpickle models.
                with np.load(out/'heads'/f'{label}.npz',allow_pickle=False) as saved:
                    if not np.array_equal(saved['residual_weight'],weight): raise ValueError('Saved head mismatch')
                guards[label]=dict(**metrics,sha256=digest)
            shared.write_json(out/'head_guards.json',guards)
            for label,weight in heads.items():
                with torch.no_grad(): policy.actor.residual_weight.copy_(torch.as_tensor(weight,device='cuda'))
                for seed in cfg['training_seeds']:
                    budget.check(); checkpoints.status(out,'collecting',head=label,seed=seed,completed_flights=len(records),
                        expected_flights=len(heads)*len(cfg['training_seeds'])+2)
                    env=shared.ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),'clearance',seed_base=seed,trace_dir=None)
                    public,_=env.reset(seed=seed);policy.reset()
                    try:
                        _,endpoint,path=checkpoints.run_suffix(policy,env,public,out/'flights'/label,str(seed),
                            dict(seed=seed,head=label,head_sha256=guards[label]['sha256']),budget,deterministic=True)
                    finally: env.close()
                    row=dict(head=label,seed=seed,endpoint=endpoint,trace=str(path),**score_endpoint(endpoint))
                    records.append(row);result['completed_flights']=len(records)
                    with (out/'outcomes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            repeat_checks=[]
            with torch.no_grad():policy.actor.residual_weight.copy_(torch.as_tensor(heads['d00p'],device='cuda'))
            for seed in cfg['training_seeds'][:2]:
                budget.check();env=shared.ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),'clearance',seed_base=seed,trace_dir=None)
                public,_=env.reset(seed=seed);policy.reset()
                try:
                    repeated,_,_=checkpoints.run_suffix(policy,env,public,out/'repeats',str(seed),dict(seed=seed,head='d00p'),budget,deterministic=True)
                finally:env.close()
                with np.load(out/'flights/d00p'/f'{seed}.npz',allow_pickle=False) as saved:
                    original={k:saved[k] for k in checkpoints.PARITY_KEYS}
                repeat_checks.append(dict(seed=seed,**checkpoints.exact_suffix_parity(original,repeated,0)))
                result['completed_flights']+=1
            shared.write_json(out/'repeat_checks.json',repeat_checks)
            result['signal']=signal_decision(records,cfg['training_seeds'],cfg['directions'])
            shared.assert_frozen(policy,estimator_digest)
            result['status']='complete';budget.check()
        except (shared.TrialStopped,KeyboardInterrupt) as exc:
            result['status']='stopped';result['error']=str(exc) or 'Interrupted'
        except Exception as exc:
            result['status']='failed';result['error']=repr(exc);(out/'error.txt').write_text(traceback.format_exc())
        finally:
            try:
                if protocol is None or shared.sha(out/'protocol.json')!=protocol_digest:raise ValueError('Missing/changed protocol')
                for relative,digest in protocol['sources'].items():
                    if shared.sha(shared.ROOT/relative)!=digest or shared.sha(out/'source'/relative)!=digest:raise ValueError('Source changed: '+relative)
                for path,digest in protocol['input_sha256'].items():
                    if shared.sha(path)!=digest:raise ValueError('Input changed: '+path)
                if policy is not None:shared.assert_frozen(policy,estimator_digest)
                result['integrity_verified']=True
            except Exception as exc:
                result['integrity_verified']=False;result['status']='failed';result['integrity_error']=repr(exc)
            result['elapsed_seconds']=time.monotonic()-started
            shared.write_json(out/'result.json',result)
            checkpoints.status(out,result['status'],completed_flights=result['completed_flights'],error=result.get('error'))
            evidence,errors=shared.available_evidence(out)
            complete=result['status']=='complete' and result['completed_flights']==(1+2*cfg['directions'])*len(cfg['training_seeds'])+2 and not errors
            passed=bool(complete and not args.smoke and result['signal']['signal_passed'])
            decision=dict(decision='software_only' if args.smoke else 'prepare_separate_parameter_training_plan' if passed else 'close_signal_variant',
                software_complete=complete,signal_passed=passed,selected_head=None,policy_promoted=False,acceptance=False,physical_flight=False,
                result_sha256=shared.sha(out/'result.json'),evidence=evidence,hash_errors=errors)
            shared.write_json(out/'decision.json',decision)
            print(json.dumps({k:v for k,v in decision.items() if k!='evidence'}),flush=True)


if __name__=='__main__':main()
