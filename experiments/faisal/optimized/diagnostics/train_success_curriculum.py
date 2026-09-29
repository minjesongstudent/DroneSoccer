"""Bounded reset-distribution experiment; no final acceptance claims."""
import sys
from pathlib import Path
OUT = Path('D:/DroneSoccerData/runs/success-backward-20260928')
sys.path.insert(0, str(OUT/'source_snapshot'))
import json
import hashlib
import random
import numpy as np
import torch
from src.standard_source_policy import StandardSourceSAC
from src.success_history import SuccessHistoryLibrary, SuccessHistoryPractice, LANDMARKS
from src.success_curriculum import BackwardScheduler, SuccessMixture
from src.unified_evaluation import evaluate
from src.return_evaluation import evaluate_return_practice, _fingerprint
from src.geometric_library import GeometricReturnLibrary

SOURCE = Path('D:/DroneSoccerData/runs/standard-source-20260928/initial_model.zip')
BANK = Path('D:/DroneSoccerData/diagnostics/success-return-bank-v1')

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2))

def fit_check(model, bank, level):
    env = SuccessHistoryPractice(bank); records = []
    for row in bank.records:
        if row['landmark'] not in LANDMARKS[:level+1]: continue
        for rep in range(9):
            torch.manual_seed(3200000+row['seed']*9+rep)
            obs, info = env.reset(options={'snapshot_id': row['id']})
            while True:
                obs, _, t, tr, info = env.step(model.predict(obs, deterministic=rep==0)[0])
                if t or tr: break
            records.append(dict(seed=row['seed'], landmark=row['landmark'], replicate=rep,
                                success=bool(info['practice_success']), passes=info['passes'], outcome=info['outcome']))
    env.close()
    return records

def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '881f9d8a09deff88340a046915f24e2d9d2cbad553fc7dd2fad95e83a65135e0'
    assert not (OUT/'protocol.json').exists()
    save('protocol.json', dict(version='success-backward-trial-v1', steps=20000,
        checkpoints=[0,10000,15000,20000], treatment='authentic FIT backward reset distribution only',
        source=str(SOURCE), cuda=torch.cuda.get_device_name(), seed=41,
        episode_mix='40% single,20% full,20% current,20% earlier FIT',
        progression='all enabled stages 4/4 deterministic and >=24/32 stochastic on two checks',
        stop='centered below21/25 or varied below23/25; full or return crashes above baseline after actor learning',
        final_progress='at least5/50 gain safe returns or new-pass success on fixed development; otherwise no extension',
        reserved_internal_check='unused during training and stage selection; fixed final candidate only',
        final_7M_test_used=False, mission_accepted=False))
    bank = SuccessHistoryLibrary(BANK); scheduler = BackwardScheduler()
    env = SuccessMixture(bank, scheduler)
    model = StandardSourceSAC.load(SOURCE, env=env, device='cuda')
    assert model.num_timesteps == 0 and model.replay_buffer.size() == 0
    model.set_random_seed(41)
    development = GeometricReturnLibrary('D:/DroneSoccerData/return-libraries/front-memory-development-v1')
    baseline = None; reason = 'bounded_trial_complete'
    for step in (0,10000,15000,20000):
        if step:
            save('status.json', dict(mode='training', target=step, previous_steps=int(model.num_timesteps)))
            model.learn(step-model.num_timesteps, reset_num_timesteps=False, progress_bar=False)
        model.save(OUT/f'model_{step}.zip')
        before = _fingerprint(model)
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
        save('status.json', dict(mode='evaluating', steps=step))
        with torch.inference_mode():
            fixed = evaluate(model, stage=4, metric_ring=True, geometric_return=True)
            returns, protocol = evaluate_return_practice(model, development)
            fit = fit_check(model, bank, scheduler.level)
        assert before == _fingerprint(model)
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state_all(rng[1]); np.random.set_state(rng[2]); random.setstate(rng[3])
        scheduler.observe(step, fit)
        result = dict(step=step, fixed=fixed, returns=returns, return_protocol=protocol,
                      fit=fit, scheduler=scheduler.history, mission_accepted=False)
        save(f'evaluation_{step}.json', result)
        print('Evaluation', step, {k:v['successful_episodes'] for k,v in fixed['suites'].items()}, 'return', returns['successful_episodes'], flush=True)
        if baseline is None: baseline = result
        if step > 10000:
            suites = fixed['suites']; old = baseline['fixed']['suites']
            retention = suites['original_centered']['successful_episodes'] >= 21 and suites['varied_approach']['successful_episodes'] >= 23
            crash_ok = all(suites[k]['crash_rate'] <= old[k]['crash_rate'] for k in ('lesson','full_task','full_task_delay_200ms')) and returns['crash_episodes'] <= baseline['returns']['crash_episodes']
            if not retention or not crash_ok:
                reason = 'retention_or_crash_gate_failed'; break
    model.save_replay_buffer(OUT/'replay.pkl')
    progress = any(returns[k] >= baseline['returns'][k]+5 for k in ('successful_episodes','safe_front_return_episodes'))
    save('status.json', dict(mode='stopped', steps=int(model.num_timesteps), reason=reason,
                            development_progress=progress, mission_accepted=False,
                            reserved_check_rollouts=0))
    env.close()

if __name__ == '__main__': main()
