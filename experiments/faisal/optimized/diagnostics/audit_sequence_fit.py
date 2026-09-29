"""Offline imitation/closed-loop divergence audit; no optimization or new flights."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch

RUN = Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
sys.path.insert(0, str(RUN / 'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

OUT = Path(__file__).parent
BANK = Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3/trajectories')

def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    traces = json.loads((OUT / 'fit_trajectories.json').read_text())
    rows = []
    for checkpoint in (300, 500):
        model = RecurrentPPO.load(RUN / f'model_update_{checkpoint}.zip', device='cuda')
        model.policy.set_training_mode(False)
        for seed in protocol['fit_lap_seeds']:
            assert seed % 4 != 3
            path = BANK / f'{seed}.npz'
            with np.load(path) as data:
                obs, actions, positions = (data[k].copy() for k in ('observations', 'actions', 'positions'))
            # Authentic full teacher observation history; recurrent state starts at zero.
            state = None
            predictions = []
            with torch.inference_mode():
                for i, o in enumerate(obs[:-1]):
                    a, state = model.predict(o, state=state,
                                             episode_start=np.array([i == 0]), deterministic=True)
                    predictions.append(a)
                # Verify training's batched sequence layout against step-by-step inference.
                tensor = torch.as_tensor(obs[:-1], device='cuda')
                z = torch.zeros(1, 1, model.policy.lstm_actor.hidden_size, device='cuda')
                distribution, _ = model.policy.get_distribution(tensor, (z, z.clone()),
                    torch.zeros(len(tensor), device='cuda'))
                batch = distribution.distribution.mean.cpu().numpy().clip(-1, 1)
            predictions = np.asarray(predictions)
            batch_error = float(np.max(np.abs(batch - predictions)))
            assert batch_error < 1e-3, batch_error  # 600-step CUDA sequence accumulation; report actual error
            error = (predictions - actions) ** 2
            env = wrap(GuidedLessonEnv(stage=4), metric_ring=True, geometric_return=True)
            initial, _ = env.reset(seed=seed)
            assert np.array_equal(initial, obs[0])
            b = env.unwrapped
            rel = positions[1:] - b.hoop_centre
            teacher_xyz = np.stack((rel @ b.hoop_normal, rel @ b._axis_h, rel[:, 2]), axis=1)
            env.close()
            trace = next(t['trace'] for t in traces if t['checkpoint'] == checkpoint and t['seed'] == seed)
            student_xyz = np.array([[t['normal'], t['lateral'], t['height']] for t in trace])
            distance = np.linalg.norm(student_xyz - teacher_xyz[:len(trace)], axis=1)
            row = dict(checkpoint=checkpoint, seed=seed, dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       mse=float(error.mean()), per_action_mse=error.mean(0).tolist(),
                       action_order=['right', 'forward', 'up', 'yaw_clockwise'],
                       per_100_steps_mse=[float(error[i:i+100].mean()) for i in range(0,600,100)],
                       batch_step_max_error=batch_error,
                       divergence_first_step={str(d):next((i+1 for i, v in enumerate(distance) if v>d), None)
                                              for d in (.05, .1, .2)},
                       student_end=trace[-1],
                       teacher_action_std=actions.std(0).tolist(), predicted_action_std=predictions.std(0).tolist())
            rows.append(row)
    report = {'device':torch.cuda.get_device_name(), 'kind':'offline diagnostic, no training',
              'reserved_or_acceptance_used':False, 'rows':rows}
    (OUT / 'sequence_fit_audit.json').write_text(json.dumps(report, indent=2))
    for checkpoint in (300,500):
        r = [x for x in rows if x['checkpoint']==checkpoint]
        print(checkpoint, json.dumps({'mean_mse':np.mean([x['mse'] for x in r]),
            'per_action_mse':np.mean([x['per_action_mse'] for x in r],axis=0).tolist(),
            'per_100_steps_mse':np.mean([x['per_100_steps_mse'] for x in r],axis=0).tolist(),
            'divergence_10cm_steps':[x['divergence_first_step']['0.1'] for x in r],
            'max_batch_step_error':max(x['batch_step_max_error'] for x in r)}),flush=True)

if __name__ == '__main__':
    main()
