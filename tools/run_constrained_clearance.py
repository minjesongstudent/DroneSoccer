"""Bounded critic adaptation followed by source-anchored CUDA policy updates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.monitor import Monitor

from src.clearance_training import ClearanceTrainingEnv, PROFILE
from src.constrained_sac import ConstrainedSAC, episode_partition
from src.risk_replay import RiskReplayBuffer, collision_approaches
from src.sim import HoopEnv
from tools.run_clearance_experiment import (ROOT, BudgetAndHealth, evaluate,
    preserved_rng, retention_failures, save_checkpoint, sha, single_trainer, write_json)


PARENT = Path('D:/DroneSoccerData/runs/clearance-reward-20261007')


def subset_buffer(old, indices, capacity, device, risk=False):
    cls = RiskReplayBuffer if risk else ReplayBuffer
    buffer = cls(capacity, old.observation_space, old.action_space, device=device,
                          handle_timeout_termination=True)
    for name in ('observations','next_observations','actions','rewards','dones','timeouts'):
        getattr(buffer,name)[:len(indices)] = getattr(old,name)[indices]
    buffer.pos = len(indices)
    assert buffer.pos < capacity
    if risk:
        buffer.risk_indices = collision_approaches(buffer.dones[:buffer.pos], buffer.timeouts[:buffer.pos])
    return buffer


def heldout_residual(model, holdout):
    # A moving-target Bellman residual is a diagnostic, not independent value accuracy.
    with preserved_rng(), torch.no_grad():
        np.random.seed(11520001);torch.manual_seed(11520001)
        data=holdout.sample(4096)
        target=model.critic_target_values(data)
        values=model.critic(data.observations,data.actions)
        error=torch.cat([v-target for v in values])
        return dict(td_rmse=float(error.square().mean().sqrt()),
                    target_mean=float(target.mean()), target_std=float(target.std()))


class OnlineBudget(BudgetAndHealth):
    def _on_step(self):
        keep=super()._on_step()
        if self.model.consecutive_rejections >= 100:
            self.stop_reason='100_consecutive_actor_rollbacks'
            return False
        return keep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--guard-boundary-weight',type=float,default=0.0)
    parser.add_argument('--risk-replay',action='store_true')
    parser.add_argument('--uniform-actor-replay',action='store_true')
    parser.add_argument('--four-actions',action='store_true')
    parser.add_argument('--twelve-actions',action='store_true')
    parser.add_argument('--new-command-columns-only',action='store_true')
    parser.add_argument('--guard-size',type=int,choices=[1024,8192],default=1024)
    parser.add_argument('--guard-alignment',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if args.four_actions and args.twelve_actions:
        parser.error('Choose one explicit action-history extension')
    if args.new_command_columns_only and not args.twelve_actions:
        parser.error('New-command-only update requires the twelve-command profile')
    extended=args.four_actions or args.twelve_actions
    inputs=121 if args.twelve_actions else 89 if args.four_actions else 85
    action_history=12 if args.twelve_actions else 4 if args.four_actions else 3
    if args.uniform_actor_replay and not args.risk_replay:
        parser.error('Uniform actor ablation requires risk replay')
    if not 0 <= args.guard_boundary_weight <= 1:
        parser.error('Boundary weight must be in [0,1]')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        source=ROOT/'trainedModels/seenflag-s1-level1.zip'
        assert sha(source)==PROFILE.sha256
        hashes={}
        files=list((ROOT/'src').glob('*.py')) + [Path(__file__),
            ROOT/'tools/run_clearance_experiment.py',ROOT/'tools/validate_flight_policies.py',
            ROOT/'tests/test_constrained_sac.py',ROOT/'tests/test_risk_replay.py',
            ROOT/'tests/test_action_history_extension.py',ROOT/'tools/evaluate_four_actions.py',
            ROOT/'tests/test_twelve_action_history.py',ROOT/'tools/evaluate_twelve_actions.py',source]
        for file in files:
            rel=file.relative_to(ROOT);dest=out/'source'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest)
            hashes[str(rel)]=sha(dest)
        replay_path=PARENT/'clearance/step_200000/replay_buffer.pkl'
        protocol=dict(kind='software smoke' if args.smoke else 'bounded critic adaptation and constrained actor trial',
            inputs=inputs,action_history=action_history,
            trace_contract='Base85 observations plus every actual action; reconstruct explicit4/12 command history with src/action_history_extension.py or src/twelve_action_history.py. Replay/model use declared actor dimension.',
            smoke=args.smoke,source=asdict(PROFILE),source_hashes=hashes,
            replay=str(replay_path),replay_sha256=sha(replay_path),
            dataset='Only original-policy warmup, 200000 real learned transitions; full-clearance rewards',
            split='Every fifth complete reset family held out (episode index mod5==4); partial final family stays grouped',
            actor_lr=1e-5 if args.new_command_columns_only else 1e-6,
            actor_update_scope='added36 input columns only' if args.new_command_columns_only else 'all actor parameters',
            critic_lr=3e-4,mean_kl_limit=.01,max_kl_limit=.1,kl_penalty=1.,
            guard_boundary_weight=args.guard_boundary_weight,
            risk_replay=args.risk_replay,
            uniform_actor_replay=args.uniform_actor_replay,
            risk_sampling='Half uniform, half last16 transitions before FIT non-timeout terminals; unchanged rewards',
            guard_boundary_penalty='Squared normalized excess above half the unchanged mean/max hard guard',
            entropy='fixed source coefficient',critic_clip=10.,actor_clip=.5,
            critic_checks=[0,8] if args.smoke else [0,1000,3000,5000],
            online_checks=[8,16] if args.smoke else [100,500,1000,2000,5000],
            fit_guard_size=args.guard_size,guard_episodes=2 if args.smoke else 25,
            final_episodes=2 if args.smoke else 100,wall_seconds=120 if args.smoke else 3600,
            online_seed_base=11100000,algorithm_seed=11520000,
            stops='Nonfinite/error, actor trust rollback,100 consecutive rejections, unchanged development retention guards',
            acceptance=False,physical_flight=False,promotion=False,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
        write_json(out/'protocol.json',protocol)
        if args.guard_alignment:
            protocol['guard_alignment']=dict(max_penalty=.2,band_m=[-.4,0],peak_m=-.1,
                formula='Negative .2 times continuous front band times projected noodle/guard penetration fraction, at pre-action pose',
                observation_change=False,physics_change=False,prepared='D:/DroneSoccerData/runs/guard-rewards-20261007')
            write_json(out/'protocol.json',protocol)
        print('PROTOCOL',out/'protocol.json',flush=True)
        # Loading a preserved local replay is not a new flight or behavior-policy substitution.
        with replay_path.open('rb') as f: old=pickle.load(f)
        assert not old.full and old.size()==200000 and old.n_envs==1
        if args.guard_alignment:
            prepared=Path('D:/DroneSoccerData/runs/guard-rewards-20261007')
            report=json.loads((prepared/'report.json').read_text())
            assert report['replay_sha256']==sha(replay_path)
            assert report['penalties_sha256']==sha(prepared/'penalties.npy')
            assert sha(prepared/'reward_source.py')==sha(ROOT/'src/guard_alignment_reward.py')
            penalties=np.load(prepared/'penalties.npy')
            assert penalties.shape==(200000,) and np.isfinite(penalties).all()
            old.rewards[:200000,0]+=penalties
            shutil.copytree(prepared,out/'reward_reconstruction')
        count=2000 if args.smoke else old.size()
        if extended:
            if args.twelve_actions:
                from src.twelve_action_history import extend_twelve_replay as extend_replay_observations
            else:
                from src.action_history_extension import extend_replay_observations
            obs,nxt=extend_replay_observations(old.observations[:old.pos,0],old.next_observations[:old.pos,0],old.dones[:old.pos])
            old.observations=obs[:,None,:];old.next_observations=nxt[:,None,:]
            import gymnasium as gym
            old.observation_space=gym.spaces.Box(-np.inf,np.inf,(inputs,),np.float32)
        fit,held,seeds=episode_partition(old.dones[:count])
        assert set(seeds[fit]).isdisjoint(seeds[held])
        np.savez_compressed(out/'partition.npz',fit_indices=fit,heldout_indices=held,reset_seeds=seeds)
        base_env=ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),
                                       'clearance',seed_base=11100000,trace_dir=out/'online_trajectories',
                                       guard_alignment=args.guard_alignment)
        evaluator=evaluate
        if extended:
            if args.twelve_actions:
                from src.twelve_action_history import TwelveActionObservation as ObservationExtension,extend_twelve_model as extend_model
                from tools.evaluate_twelve_actions import evaluate_twelve_actions as extended_evaluator
            else:
                from src.action_history_extension import FourActionObservation as ObservationExtension,extend_model
                from tools.evaluate_four_actions import evaluate_four_actions as extended_evaluator
            env=Monitor(ObservationExtension(base_env))
            model=extend_model(ConstrainedSAC.load(source,device='cuda'))
            model.set_env(env);model.set_random_seed(11520000)
            evaluator=extended_evaluator
            model.buffer_size=250000
        else:
            env=Monitor(base_env)
            model=ConstrainedSAC.load(source,env=env,device='cuda',seed=11520000,
            buffer_size=250000,learning_starts=0,train_freq=1,gradient_steps=1,
            tensorboard_log=None,verbose=0)
        model.get_env().seed(11100000)
        assert next(model.actor.parameters()).device.type=='cuda'
        rng=np.random.default_rng(11520000)
        guard_indices=rng.choice(fit,size=min(args.guard_size,len(fit)),replace=False)
        np.save(out/'fit_guard_indices.npy',guard_indices)
        model.configure(old.observations[guard_indices,0], boundary_weight=args.guard_boundary_weight,
                        uniform_actor_replay=args.uniform_actor_replay)
        if args.new_command_columns_only:
            from src.twelve_action_history import restrict_actor_to_new_columns,assert_original_actor_unchanged
            restrict_actor_to_new_columns(model.actor,learning_rate=protocol['actor_lr'])
        model.replay_buffer=subset_buffer(old,fit,250000,'cuda',risk=args.risk_replay)
        if args.risk_replay:
            np.save(out/'risk_fit_indices.npy',model.replay_buffer.risk_indices)
        holdout=subset_buffer(old,held,len(held)+1,'cuda')
        del old
        actor_before={k:v.clone() for k,v in model.actor.state_dict().items()}
        alpha_before=model.log_ent_coef.detach().clone()
        initial_updates=model._n_updates
        baseline=json.loads((PARENT/'reference_guard/summary.json').read_text())
        if extended:
            initial_path=out/f'initial_{inputs}.zip'
            model.save(initial_path)
            initial=evaluator(initial_path,out/'initial_parity_guard',2 if args.smoke else 25)
            if not args.smoke and initial!=baseline:
                raise RuntimeError('Expanded policy initial guard differs from original')
            if args.twelve_actions:
                # Require exact learned actions and physical trajectories, not merely equal counts.
                for candidate_path in (out/'initial_parity_guard').glob('*/*.npz'):
                    reference=PARENT/'reference_guard'/candidate_path.relative_to(out/'initial_parity_guard')
                    with np.load(candidate_path) as candidate, np.load(reference) as original:
                        for key in ('actions','positions','headings'):
                            np.testing.assert_array_equal(candidate[key],original[key])
                write_json(out/'initial_trajectory_parity.json',dict(exact=True,episodes=sum(v['episodes'] for v in initial.values())))
        result=dict(status='running',critic_checks=[],online_checks=[],acceptance=False,physical_flight=False,
                    fit_transitions=len(fit),heldout_transitions=len(held))
        deadline=time.monotonic()+protocol['wall_seconds']
        callback=OnlineBudget(out,out,'constrained',deadline,initial_updates)
        final=None
        try:
            previous=0
            for checkpoint in protocol['critic_checks']:
                for update in range(previous,checkpoint):
                    if time.monotonic()>=deadline or (out/'STOP').exists():
                        raise RuntimeError('Stopped during critic calibration; no actor change')
                    model.train(1,batch_size=256)
                previous=checkpoint
                assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in actor_before.items())
                assert torch.equal(alpha_before,model.log_ent_coef)
                metrics=heldout_residual(model,holdout)
                if not np.isfinite(list(metrics.values())).all(): raise FloatingPointError('Invalid heldout residual')
                record=dict(critic_updates=checkpoint,actor_exact=True,**metrics)
                result['critic_checks'].append(record)
                final=save_checkpoint(model,env,out,f'critic_{checkpoint:06d}',initial_updates)
                write_json(out/'status.json',record)
                write_json(out/'result.json',result)
                print('CRITIC',json.dumps(record),flush=True)
            model.actor_enabled=True
            for i,target in enumerate(protocol['online_checks']):
                count=target-(0 if i==0 else model.num_timesteps)
                model.learn(count,callback=callback,reset_num_timesteps=(i==0),log_interval=1000)
                allowed,drift=model.trust_check()
                assert allowed
                assert torch.equal(alpha_before,model.log_ent_coef)
                if args.new_command_columns_only:
                    assert_original_actor_unchanged(model.actor,actor_before)
                final=save_checkpoint(model,env,out,f'online_{model.num_timesteps:06d}',initial_updates)
                metrics=evaluator(final,out/f'guard_{model.num_timesteps:06d}',protocol['guard_episodes'])
                reasons=[] if args.smoke else retention_failures(baseline,metrics)
                record=dict(steps=model.num_timesteps,accepted=model.accepted_updates,
                    rejected=model.rejected_updates,drift=drift,metrics=metrics,retention_failures=reasons)
                result['online_checks'].append(record);write_json(out/'result.json',result)
                print('ONLINE',json.dumps(record),flush=True)
                if reasons or callback.stop_reason or time.monotonic()>=deadline:
                    result['status']='retention_regression' if reasons else callback.stop_reason or 'wall_cap'
                    break
            else:
                result['status']='bounded_trial_complete'
            result['final_checkpoint']=str(final)
            result['final_development']=evaluator(final,out/'final_development',protocol['final_episodes'])
        except BaseException:
            result['status']='error';result['error']=traceback.format_exc()
            try: save_checkpoint(model,env,out,'error_checkpoint',initial_updates)
            except Exception: result['checkpoint_error']=traceback.format_exc()
            raise
        finally:
            env.close()
            result['accepted_updates']=model.accepted_updates;result['rejected_updates']=model.rejected_updates
            result['source_unchanged']=sha(source)==PROFILE.sha256
            write_json(out/'result.json',result)
            write_json(out/'status.json',dict(status=result['status'],source_unchanged=result['source_unchanged']))


if __name__=='__main__': main()
