"""Read-only actor drift on saved development observations; no flights or updates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from stable_baselines3 import SAC


def distribution(model, observations):
    model.policy.set_training_mode(False)
    means, log_stds = [], []
    with torch.no_grad():
        for chunk in np.array_split(observations, max(1, len(observations)//1024+1)):
            mean, log_std, _ = model.actor.get_action_dist_params(torch.as_tensor(chunk))
            means.append(mean.numpy()); log_stds.append(log_std.numpy())
    return np.concatenate(means), np.concatenate(log_stds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_path = args.run/'source/trainedModels/seenflag-s1-level1.zip'
    protocol = json.loads((args.run/'protocol.json').read_text())
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == protocol['source']['sha256']
    source = SAC.load(source_path, device='cpu')
    source_data = {}
    for suite in ('aligned', 'nominal', 'randomised'):
        parts = []
        for p in sorted((args.run/'reference_guard'/suite).glob('*.npz')):
            with np.load(p) as trace:
                parts.append(trace['actor_observations'])
        obs = np.concatenate(parts)
        assert obs.shape[1] == 85
        source_data[suite] = (obs, *distribution(source, obs))
    results = {}
    for path in sorted(args.run.glob('*/step_*/model.zip')):
        model = SAC.load(path, device='cpu')
        entry = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     entropy_coefficient=float(model.log_ent_coef.exp().item()), suites={})
        for suite, (obs, mu0, ls0) in source_data.items():
            mu, ls = distribution(model, obs)
            # KL of the unsquashed Gaussians equals that of their common invertible
            # tanh transform. This describes drift, not its cause or flight safety.
            kl = np.sum(ls-ls0 + (np.exp(2*ls0)+(mu0-mu)**2)/(2*np.exp(2*ls))-.5, axis=1)
            change = np.abs(np.tanh(mu)-np.tanh(mu0))
            entry['suites'][suite] = dict(observations=len(obs),
                mean_kl=float(np.mean(kl)), p95_kl=float(np.quantile(kl, .95)),
                max_kl=float(np.max(kl)), mean_absolute_action_change=float(np.mean(change)),
                max_absolute_action_change=float(np.max(change)))
        results[str(path.relative_to(args.run))] = entry
    output = dict(scope='Saved source development histories; no new flights or optimization; not causal proof',
                  source_entropy_coefficient=float(source.log_ent_coef.exp().item()), candidates=results)
    folder = args.run/'analysis'
    folder.mkdir(exist_ok=True)
    shutil.copy2(__file__, folder/Path(__file__).name)
    (folder/'actor_drift.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
