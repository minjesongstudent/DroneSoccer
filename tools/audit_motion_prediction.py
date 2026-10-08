"""Predict recorded one-step displacement. Diagnostic networks never control flight."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from tools.run_clearance_experiment import ROOT, sha, single_trainer, write_json


def body_frame(displacement, heading):
    c, s = np.cos(heading), np.sin(heading)
    x, y, z = np.asarray(displacement).T
    return np.column_stack((c*x+s*y, s*x-c*y, z)).astype(np.float32)


def episode_features(z):
    obs = z['actor_observations'].astype(np.float32)
    actions = z['actions'].astype(np.float32)
    pos, heading = z['positions'], z['headings']
    # Row i is before action i; positions[i-1] is its starting position.
    # Skip first two rows: neither initial position nor initial velocity was logged.
    ids = np.arange(2, len(actions))
    history = np.zeros((len(actions), 12, 4), dtype=np.float32)
    for lag in range(1, 13):
        history[lag:, 12-lag] = actions[:-lag]
    np.testing.assert_array_equal(history[:, -3:].reshape(-1, 12), obs[:, 72:84])
    target = body_frame(pos[ids]-pos[ids-1], heading[ids-1])*1000
    prior = body_frame(pos[ids-1]-pos[ids-2], heading[ids-1])*1000
    public = np.c_[obs[ids], actions[ids]]
    return dict(public=public, zero_padding=np.c_[public,np.zeros((len(ids),36),np.float32)],
                commands12=np.c_[public, history[ids, :9].reshape(-1, 36)],
                prior_motion_truth=np.c_[public, prior], target=target,
                prior=prior, visible=obs[ids, 70] > .5)


def metrics(pred, target, visible):
    result = {}
    for name, mask in [('all', np.ones(len(target), bool)), ('visible', visible), ('blind', ~visible)]:
        residual = pred[mask]-target[mask]
        result[name] = dict(samples=int(mask.sum()),
            rmse_mm_per_axis=float(np.sqrt(np.mean(residual**2))),
            mean_vector_error_mm=float(np.linalg.norm(residual, axis=1).mean()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    source = Path('D:/DroneSoccerData/runs/clearance-reward-20261007/clearance/training_trajectories')
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        args.out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(__file__, args.out/'audit_source.py')
        pieces, hashes, groups, cumulative = [], {}, [], 0
        for line in (source/'episodes.jsonl').read_text().splitlines():
            row = json.loads(line)
            cumulative += row['steps']
            if cumulative > 200000:
                break  # Never include transitions collected after source warmup.
            if row['status'] != 'complete':
                continue
            path = source/row['trace']
            with np.load(path) as z:
                piece = episode_features(z)
            if not len(piece['target']):
                continue
            groups.extend([len(pieces)]*len(piece['target']))
            pieces.append(piece)
            hashes[str(path)] = sha(path)
        data = {k: np.concatenate([v[k] for v in pieces]) for k in pieces[0]}
        groups = np.asarray(groups)
        fit, held = np.flatnonzero(groups % 5 != 4), np.flatnonzero(groups % 5 == 4)
        np.savez_compressed(args.out/'split.npz', episode_ids=groups, fit=fit, heldout=held)
        write_json(args.out/'protocol.json', dict(sources=hashes, actor_updated=False,
            acceptance=False, device=torch.cuda.get_device_name(), seeds=[11560000,11560001],
            updates=1500, target='Next actual displacement in pre-action body frame, millimetres',
            split='Every fifth complete source warmup episode held out; no validation trajectories',
            variants=['public','zero_padding','commands12','prior_motion_truth'],
            truth_variant='Diagnostic only: prior displacement, never future displacement or actor input'))
        target = data['target']
        results = dict(baselines={name: metrics(pred[held], target[held], data['visible'][held])
            for name, pred in [('zero', np.zeros_like(target)), ('prior_displacement_truth', data['prior'])]})
        y_mean, y_scale = target[fit].mean(0), np.maximum(target[fit].std(0), 1e-3)
        y_gpu = torch.tensor((target-y_mean)/y_scale, device='cuda')
        for variant in ['public', 'zero_padding', 'commands12', 'prior_motion_truth']:
            x = data[variant]
            mean, scale = x[fit].mean(0), np.maximum(x[fit].std(0), .01)
            x_gpu = torch.tensor((x-mean)/scale, device='cuda')
            records = []
            for seed in [11560000,11560001]:
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                model = nn.Sequential(nn.Linear(x.shape[1],128),nn.Tanh(),
                    nn.Linear(128,128),nn.Tanh(),nn.Linear(128,3)).cuda()
                optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
                for update in range(1501):
                    if update in (0,500,1500):
                        record = dict(seed=seed, update=update)
                        for name, indices in [('fit',fit),('heldout',held)]:
                            with torch.no_grad():
                                pred = np.concatenate([model(x_gpu[batch]).cpu().numpy()
                                    for batch in np.array_split(indices, max(1,len(indices)//8192))])
                            pred = pred*y_scale+y_mean
                            record[name] = metrics(pred,target[indices],data['visible'][indices])
                        records.append(record)
                        print(variant, json.dumps(record), flush=True)
                    if update == 1500:
                        break
                    batch = rng.choice(fit,512,replace=True)
                    loss = (model(x_gpu[batch])-y_gpu[batch]).square().mean()
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),10); optimizer.step()
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    x_mean=mean,x_scale=scale,y_mean=y_mean,y_scale=y_scale),
                    args.out/f'{variant}-{seed}.pt')
            results[variant] = records
            write_json(args.out/'results.json',results)
        (args.out/'DONE').write_text('Offline motion prediction completed; no flight actor changed.\n')


if __name__ == '__main__':
    main()
