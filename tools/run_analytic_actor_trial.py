"""Bounded mean-head learning through verified shadow physics; simulation only.

Every real command is SAC.predict(public85). Truth appears only in the local
fixed-history loss. No differentiated trajectory is flown or scored as success.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import random
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import SAC

from src.approach_curriculum import ApproachCurriculumEnv
from src.clearance_training import ClearanceTrainingEnv, PROFILE
from src.constrained_sac import gaussian_kl, boundary_penalty, guarded_optimizer_step
from src.policy_runtime import LearnedPolicy, PolicySpec
from src.shadow_dynamics import rollout
from src.shadow_clearance import geometry_from_env, margins, clearance_loss
from tools.run_clearance_experiment import (ROOT, sha, write_json, single_trainer,
    summarize, retention_failures, preserved_rng, rng_state)
from tools.run_onpolicy_trial import (prepare_gae_screen_reference,
    counted_development_episodes, gae_screen_reasons)
from tools.run_parallel_estimated_motion import existing_allocation
from tools.validate_flight_policies import SUITES, episode

RUNS = Path('D:/DroneSoccerData/runs')
SOURCE = ROOT/'trainedModels/seenflag-s1-level1.zip'
BASELINE = ROOT.parent/'lab-data/policy-validation-20261006'
GUARD = RUNS/'trainable-memory-actor-20261008/guard.npz'
GUARD_SHA = '87dcd21364d0037386556e09926fb95c92727401a3371b2ba099e2c0915ceefb'
MAIN_SEEDS = tuple(range(11244240, 11244400))
SMOKE_SEEDS = tuple(range(11244790, 11244794))
PLAN = 'experiments/faisal/docs/ANALYTIC_ACTOR_PROTOCOL_20261008.md'


def check_budget(out, started, now):
    if (Path(out)/'STOP').exists() or now-started>=3600:
        raise RuntimeError('STOP or fixed 3600-second total budget')


def freeze_mean_head(model):
    if model.observation_space.shape != (85,) or model.action_space.shape != (4,):
        raise ValueError('Original public85/four-command SAC required')
    if not (np.array_equal(model.action_space.low, [-1]*4)
            and np.array_equal(model.action_space.high, [1]*4)):
        raise ValueError('Original normalized command bounds required')
    model.policy.set_training_mode(False)
    for p in model.policy.parameters(): p.requires_grad_(False)
    for p in model.actor.mu.parameters(): p.requires_grad_(True)
    return torch.optim.Adam(model.actor.mu.parameters(), lr=1e-4)


def commands(actor, public85):
    if public85.ndim != 2 or public85.shape[1] != 85 or public85.dtype != torch.float32:
        raise ValueError('Only float32 public85 enters the actor')
    mean, _, _ = actor.get_action_dist_params(public85)
    # Preserve SB3 unscale_action arithmetic, including float32 rounding.
    return -1.0 + 0.5 * (torch.tanh(mean) + 1.0) * 2.0


def sequence_commands(actor, public, lengths, check):
    """Use actual single-frame GEMM arithmetic, retaining its mean-head graph.

    Padded rows never enter the network. Padding exists only for shadow batching.
    """
    sequences=[]
    for rows,n in zip(public,lengths.tolist()):
        issued=[]
        for index in range(n):
            if index%64==0:check()
            issued.append(commands(actor,rows[index:index+1]))
        sequences.append(torch.nn.functional.pad(torch.cat(issued),(0,0,0,len(rows)-n)))
    return torch.stack(sequences)


def collect(model, seed, folder, check):
    """Actual learned flight and separate loss-only scene/dt evidence."""
    env = ClearanceTrainingEnv(ApproachCurriculumEnv(level=1, randomise=True, hanger=True),
                               'clearance', trace_dir=folder)
    observations=[]; actions=[]; positions=[]; headings=[]; dt=[]
    try:
        public, _ = env.reset(seed=seed); e=env.unwrapped
        g=geometry_from_env(e, device='cuda')
        initial=dict(initial_position=e.pos.copy(), initial_heading=float(e.heading),
            half_box=e._half_box.copy(), cage=g['cage'].cpu().numpy(), speed=e.dyn.max_speed,
            yaw_degrees=e.dyn.max_yaw_rate_deg, tau=e.dyn.tau, delay=e.dyn.delay_steps)
        if e.vel.any() or e.yaw_rate or any(np.any(x) for x in e._pending):
            raise ValueError('Shadow requires an actual zero-motion reset')
        # Independent copy records the exact dt sequence without advancing env RNG.
        clock_rng=deepcopy(e.np_random); before_crossing=False
        while True:
            check(); observations.append(public.copy())
            with torch.inference_mode(): action, _ = model.predict(public, deterministic=True)
            actions.append(action.copy()); dt.append(1/clock_rng.uniform(*e.ranges.rate_hz))
            public, _, done, cut, info = env.step(action)
            positions.append(e.pos.copy()); headings.append(e.heading)
            if not env.audit.count and (done or e.wall_steps): before_crossing=True
            if done or cut: break
        if clock_rng.bit_generator.state != e.np_random.bit_generator.state:
            raise ValueError('Unexpected per-frame RNG consumption')
        row=dict(seed=seed, steps=e.steps, outcome=info['outcome'], wall_steps=e.wall_steps,
            full_body_passes=env.audit.count, returns_to_front=env.audit.returns,
            crashed=bool(done), clean_first_pass=bool(env.audit.count and not before_crossing),
            clean_repeated_episode=bool(env.audit.count>=2 and not done and not e.wall_steps and e.steps==600))
        np.savez_compressed(folder/'loss_inputs.npz', **initial, dt=np.array(dt),
            **{'geometry_'+k: v.detach().cpu().numpy() if torch.is_tensor(v) else np.array(v)
               for k,v in g.items()})
        write_json(folder/'outcome.json',row)
        return dict(public=np.array(observations), actions=np.array(actions), positions=np.array(positions),
                    headings=np.array(headings), dt=np.array(dt), initial=initial, geometry=g, outcome=row)
    finally: env.close()


def batch_tensors(records, device='cuda'):
    lengths=np.array([len(r['actions']) for r in records],np.int64); width=int(lengths.max())
    public=np.zeros((len(records),width,85),np.float32)
    actual=np.zeros((len(records),width,4),np.float64); dt=np.full((len(records),width),.1)
    for j,r in enumerate(records):
        n=lengths[j];public[j,:n]=r['public'];actual[j,:n]=r['actions'];dt[j,:n]=r['dt']
    kw={key:torch.as_tensor(np.array([r['initial'][key] for r in records]),
        dtype=torch.int64 if key=='delay' else torch.float64,device=device) for key in records[0]['initial']}
    kw.update(dt=torch.as_tensor(dt,device=device),lengths=torch.as_tensor(lengths,device=device))
    return torch.as_tensor(public,device=device), torch.as_tensor(actual,device=device), kw


def local_loss(actor, records, check):
    public, actual, kw=batch_tensors(records)
    action=sequence_commands(actor,public,kw['lengths'],check).double()
    valid=torch.arange(public.shape[1],device=public.device)[None,:]<kw['lengths'][:,None]
    # Keep the original strict ceiling; single-frame inference removes batching drift.
    error=float((action.detach()[valid]-actual[valid]).abs().max())
    if error>2e-6: raise ValueError(f'Batch/runtime action error {error} exceeds 2e-6')
    state=rollout(actions=action,**kw,check=check)
    losses=[]; pe=he=0.
    for j,r in enumerate(records):
        n=len(r['actions']); p=state['positions'][j,:n]; h=state['headings'][j,:n]
        pe=max(pe,float((p.detach()-torch.as_tensor(r['positions'],device=p.device)).abs().max()))
        delta=h.detach()-torch.as_tensor(r['headings'],device=h.device)
        he=max(he,float(((delta+np.pi)%(2*np.pi)-np.pi).abs().max()))
        losses.append(clearance_loss(margins(p,h,**r['geometry']))/.02**2)
    if pe>1e-4 or he>1e-4: raise ValueError('Batch motion exceeds fixed local replay tolerances')
    return torch.stack(losses).mean(), dict(action_error=error,position_error=pe,heading_error=he)


class CheckedPolicy(LearnedPolicy):
    def __init__(self,path,spec,check):
        super().__init__(path,spec);self.check=check
    def predict(self,*args):
        self.check();return super().predict(*args)


def evaluate(checkpoint, folder, count, check):
    folder.mkdir(parents=True,exist_ok=False)
    spec=PolicySpec(checkpoint.stem,sha(checkpoint),PROFILE.lags)
    write_json(folder/'model.json',asdict(spec));result={}
    with preserved_rng():
        # Same CPU deterministic evaluation arithmetic as the preserved baseline.
        policy=CheckedPolicy(checkpoint,spec,check)
        for suite,base in SUITES.items():
            target=folder/suite;target.mkdir();rows=[]
            for seed in range(base,base+count):
                check();row=episode(policy,suite,seed,target/f'{seed}.npz');rows.append(row)
                with (target/'episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            result[suite]=summarize(rows)
    write_json(folder/'summary.json',result);return result


def final_decision(reference, candidate, *, smoke, complete, accepted, changed, failures):
    reasons=list(failures)
    if smoke:reasons.append('Software-only smoke')
    if not complete:reasons.append('Prescribed training budget did not complete')
    if not accepted or not changed:reasons.append('No verified mean-head learning')
    if candidate is None:reasons.append('No completed final screen')
    elif not smoke:reasons.extend(gae_screen_reasons(reference,candidate))
    return dict(passed=not reasons,reasons=reasons,
        decision='eligible_for_one_fresh_development_screen' if not reasons else 'reject_close_variant',
        acceptance=False,policy_promoted=False,physical_flight=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def check():
        check_budget(out,started,time.monotonic())
    model=optimizer=None;rounds=[];accepted=rejected=0;inputs={}
    def bind(path,expected=None):
        path=Path(path);value=sha(path)
        if expected is not None and value!=expected:raise ValueError('Input identity changed: '+str(path))
        inputs[str(path.resolve())]=value;return path
    try:
        with single_trainer(ROOT.parent/'.clearance-training.lock'):
            seeds=SMOKE_SEEDS if args.smoke else MAIN_SEEDS
            write_json(out/'seed_allocation.json',existing_allocation(RUNS,list(seeds)))
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.cuda.set_per_process_memory_fraction(.25)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            random.seed(11607000);np.random.seed(11607000);torch.manual_seed(11607000);torch.cuda.manual_seed_all(11607000)
            bind(SOURCE,PROFILE.sha256);bind(GUARD,GUARD_SHA)
            for name,expected,components in (
                ('shadow-dynamics-parity-20261008','516d16e2faeb8462e1fe6a5b3d07513fe04e0be00e4137271406715a3e493751',['src/shadow_dynamics.py']),
                ('shadow-clearance-parity-20261008','ca0a80b61401447e44a5b6b47baab0e8fc764caf226c8611c9e04c3a4d930f3c',['src/shadow_clearance.py','src/sim.py','src/params.py','src/contract.py'])):
                bind(RUNS/name/'result.json',expected)
                previous=json.loads(bind(RUNS/name/'protocol.json').read_text())
                old={k.replace('\\','/'):v for k,v in previous['sources'].items()}
                for key in components:bind(ROOT/key,old[key])
            sources={}
            files=list((ROOT/'src').glob('*.py'))+[ROOT/x for x in (
                'tools/run_analytic_actor_trial.py','tools/run_clearance_experiment.py','tools/run_onpolicy_trial.py',
                'tools/run_parallel_estimated_motion.py','tools/parallel_training.py','tools/validate_flight_policies.py',
                'tests/test_analytic_actor_trial.py',PLAN)]
            for source in files:
                rel=source.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,dest);sources[rel.as_posix()]=sha(dest)
            reference_record=prepare_gae_screen_reference(BASELINE,out/'screen_reference',SOURCE)
            reference=json.loads(Path(reference_record['path']).read_text())['metrics']
            per_round=4 if args.smoke else 32;total_rounds=1 if args.smoke else 5
            write_json(out/'protocol.json',dict(kind='Analytic mean-head trial; fixed-history local gradient',smoke=args.smoke,
                sources=sources,inputs=inputs,training_seeds=list(seeds),rounds=total_rounds,episodes_per_round=per_round,
                batch_size=4,epochs=1,updates=total_rounds*per_round//4,actor_lr=1e-4,algorithm_seed=11607000,
                loss='equal episode mean clearance_loss/.02^2 + .1*source KL mean + .1*KL boundary penalty',
                trainable='actor.mu only; feature backbone and log_std frozen',actor_inputs='unchanged public85 only',
                mean_kl_limit=.01,max_kl_limit=.1,backtracks=4,gradient_norm_limit=.5,guard_rows=8192,
                rejected_update_stop=8,
                max_seconds=3600,cuda_allocator_fraction=.25,gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,
                precision='CUDA float32 actor/noTF32; CUDA float64 loss; source-compatible CPU development evaluation',
                action_replay_tolerance=2e-6,position_heading_replay_tolerance=1e-4,
                retention='25 per suite each round: first drop<=2, repeat drop<=3, crash rise<=3',
                final_gate='final only100/suite; random repeat>=53 crash<=47; all first/repeat drop and crash rise<=2',
                failure='Close on failed final, retention stop, nonfinite, replay mismatch, STOP or budget; no automatic extension',
                limitation='No camera, policy-feedback, terminal or passage gradient; real outcomes alone decide',
                acceptance=False,physical_flight=False))
            with np.load(GUARD,allow_pickle=False) as z:guard=torch.as_tensor(z['features'][:,:85].copy(),device='cuda')
            if guard.shape!=(8192,85) or not torch.isfinite(guard).all():raise ValueError('Invalid original FIT guard')
            model=SAC.load(SOURCE,device='cuda');optimizer=freeze_mean_head(model)
            original={k:v.detach().clone() for k,v in model.actor.state_dict().items()}
            with torch.no_grad():ref_mean,ref_std,_=model.actor.get_action_dist_params(guard)
            ref_mean=ref_mean.detach().clone();ref_std=ref_std.detach().clone()
            for obs in guard[:64]:
                check()
                with torch.no_grad():a=commands(model.actor,obs[None])[0].cpu().numpy()
                b,_=model.predict(obs.cpu().numpy(),deterministic=True)
                if not np.array_equal(a,b):raise ValueError('Initialization action conversion not exact')
            model.save(out/'initial.zip')
            reloaded=SAC.load(out/'initial.zip',device='cuda')
            if any(not torch.equal(v,reloaded.actor.state_dict()[k]) for k,v in original.items()):
                raise ValueError('Initialization checkpoint mismatch')
            del reloaded
            torch.save(dict(optimizer=optimizer.state_dict(),rng=rng_state()),out/'initial-state.pt')
            write_json(out/'initialization.json',dict(source_sha256=PROFILE.sha256,exact_conversion_rows=64,
                exact_checkpoint=True,trainable=[k for k,v in model.actor.named_parameters() if v.requires_grad]))
            gradient_norms={}
            def record_gradient(name):
                def hook(value):
                    if not torch.isfinite(value).all():raise ValueError('Nonfinite mean-head gradient')
                    gradient_norms[name]=float(torch.linalg.vector_norm(value))
                    return value
                return hook
            for name,param in model.actor.mu.named_parameters():param.register_hook(record_gradient(name))
            def kl_values():
                mean,std,_=model.actor.get_action_dist_params(guard)
                return gaussian_kl(ref_mean,ref_std,mean,std)
            def trust_check():
                check()
                with torch.no_grad():kl=kl_values()
                finite=bool(torch.isfinite(kl).all())
                if not finite:raise ValueError('Nonfinite actor distribution')
                values=dict(mean_kl=float(kl.mean()),max_kl=float(kl.max()))
                return values['mean_kl']<=.01 and values['max_kl']<=.1,values
            base_guard={}
            for suite,base in SUITES.items():
                rows=[json.loads(s) for s in (BASELINE/PROFILE.name/suite/'episodes.jsonl').read_text().splitlines()]
                base_guard[suite]=summarize(rows[:25])
            failures=[];checkpoint=out/'initial.zip'
            for iteration in range(total_rounds):
                updates=[]
                for batch in range(per_round//4):
                    check();idx=iteration*per_round+batch*4;batch_records=[]
                    write_json(out/'status.json',dict(status='running',phase='collecting',round=iteration+1,
                        episodes_complete=idx,total_episodes=len(seeds),accepted=accepted,rejected=rejected))
                    for seed in seeds[idx:idx+4]:
                        batch_records.append(collect(model,seed,out/'trajectories'/str(seed),check))
                    write_json(out/'status.json',dict(status='running',phase='gradient',round=iteration+1,
                        episodes_complete=idx+4,total_episodes=len(seeds),accepted=accepted,rejected=rejected))
                    cost,parity=local_loss(model.actor,batch_records,check)
                    kl=kl_values();loss=cost+.1*kl.mean()+.1*boundary_penalty(kl,.01,.1)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    before={k:v.detach().clone() for k,v in model.actor.mu.state_dict().items()}
                    ok,trust=guarded_optimizer_step(model.actor,optimizer,loss,trust_check,max_backtracks=4)
                    accepted+=int(ok);rejected+=int(not ok)
                    changed=any(not torch.equal(v,model.actor.mu.state_dict()[k]) for k,v in before.items())
                    if any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in original.items() if not k.startswith('mu.')):
                        raise ValueError('Frozen backbone or variance changed')
                    if any(not torch.isfinite(p).all() for p in model.actor.parameters()):raise ValueError('Nonfinite weights')
                    update=dict(batch=idx//4+1,accepted=ok,changed=changed,cost=float(cost.detach()),
                        loss=float(loss.detach()),gradient_norms=dict(gradient_norms),**trust,**parity,
                        outcomes=summarize([r['outcome'] for r in batch_records]))
                    updates.append(update)
                    with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(update)+'\n')
                    del loss,cost,kl,batch_records
                    if rejected>=8:raise RuntimeError('Eight rejected updates; close without extension')
                checkpoint=out/f'policy_{iteration+1:02d}.zip';model.save(checkpoint)
                torch.save(dict(optimizer=optimizer.state_dict(),rng=rng_state()),out/f'policy_{iteration+1:02d}-state.pt')
                reloaded=SAC.load(checkpoint,device='cuda')
                if any(not torch.equal(v,reloaded.actor.state_dict()[k]) for k,v in model.actor.state_dict().items()):
                    raise ValueError('Updated checkpoint mismatch')
                with torch.no_grad():
                    if not torch.equal(commands(model.actor,guard[:64]),commands(reloaded.actor,guard[:64])):
                        raise ValueError('Updated checkpoint actions mismatch')
                del reloaded
                write_json(out/'status.json',dict(status='running',phase='retention',round=iteration+1))
                metrics=evaluate(checkpoint,out/f'guard_{iteration+1:02d}',2 if args.smoke else 25,check)
                failures=[] if args.smoke else retention_failures(base_guard,metrics)
                rounds.append(dict(iteration=iteration+1,updates=updates,retention=metrics,failures=failures))
                write_json(out/'rounds.json',rounds)
                if failures:break
            complete=len(rounds)==total_rounds and not failures
            candidate=None
            if complete:
                write_json(out/'status.json',dict(status='running',phase='final_screen'))
                candidate=evaluate(checkpoint,out/'final_development',2 if args.smoke else 100,check)
                for suite,base in SUITES.items():
                    counted=counted_development_episodes(out/'final_development'/suite/'episodes.jsonl',base,2 if args.smoke else 100)
                    if any(candidate[suite][k]!=v for k,v in counted.items()):raise ValueError('Final recount mismatch')
            check()
            for path,digest in inputs.items():
                if sha(path)!=digest:raise ValueError('Input changed')
            for rel,digest in sources.items():
                if sha(ROOT/rel)!=digest or sha(out/'source'/rel)!=digest:raise ValueError('Source changed')
            changed=any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in original.items())
            decision=final_decision(reference,candidate,smoke=args.smoke,complete=complete,accepted=accepted,
                changed=changed,failures=failures)
            decision.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),candidate=candidate,
                reference=reference,source_unchanged=True,backbone_variance_unchanged=True,
                prescribed_budget_complete=complete,accepted=accepted,rejected=rejected,actor_changed=changed)
            write_json(out/'decision.json',decision)
            write_json(out/'result.json',dict(status='complete',rounds=rounds,elapsed_seconds=time.monotonic()-started))
            write_json(out/'evidence.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
                if p.is_file() and p.name not in ('status.json','evidence.json')})
            write_json(out/'status.json',dict(status='complete',decision=decision['decision'],accepted=accepted,rejected=rejected))
            print(json.dumps(decision),flush=True)
    except Exception as exc:
        if model is not None:
            model.save(out/'interrupted.zip')
            if optimizer is not None:torch.save(dict(optimizer=optimizer.state_dict(),rng=rng_state()),out/'interrupted-state.pt')
        write_json(out/'decision.json',dict(passed=False,decision='incomplete_no_qualification',error=repr(exc),
            acceptance=False,policy_promoted=False,physical_flight=False))
        write_json(out/'status.json',dict(status='failed',error=repr(exc),accepted=accepted,rejected=rejected))
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
