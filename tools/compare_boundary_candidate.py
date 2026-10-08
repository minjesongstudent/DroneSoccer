"""Frozen fresh development comparison; never reserved mission acceptance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import torch

from src.policy_runtime import LearnedPolicy, PolicySpec, SPECS
from tools.validate_flight_policies import ROOT, episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate',type=Path,default=Path('D:/DroneSoccerData/runs/constrained-boundary-20261007/online_005000/model.zip'))
    parser.add_argument('--candidate-label',choices=['boundary','onpolicy','adapter','guard_reward'],default='boundary')
    parser.add_argument('--candidate-profile',choices=['original85','twelve121'],default='original85')
    parser.add_argument('--episodes',type=int,choices=[100,200],default=100)
    parser.add_argument('--cohort',choices=['951','961','971','981'],default='951')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    paths = {
        'original': ROOT/'trainedModels/seenflag-s1-level1.zip',
        args.candidate_label: args.candidate,
    }
    suites = dict(aligned=9510000, nominal=9520000, randomised=9530000)
    if args.cohort=='961':suites={name:seed+100000 for name,seed in suites.items()}
    if args.cohort=='971':suites={name:seed+200000 for name,seed in suites.items()}
    if args.cohort=='981':suites={name:seed+300000 for name,seed in suites.items()}
    hashes = {}
    for name, path in paths.items():
        dest = args.out/'models'/f'{name}.zip'
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
        paths[name] = dest
    assert hashes['original'] == SPECS['seenflag-s1-level1'].sha256
    sources = list((ROOT/'src').glob('*.py')) + [Path(__file__), ROOT/'tools/validate_flight_policies.py',
        ROOT/'tools/evaluate_twelve_actions.py',ROOT/'tools/evaluate_four_actions.py',ROOT/'tools/run_clearance_experiment.py']
    source_hashes = {}
    for path in sources:
        rel = path.resolve().relative_to(ROOT)
        dest = args.out/'source'/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[str(rel)] = hashlib.sha256(dest.read_bytes()).hexdigest()
    protocol = dict(kind='fresh development, not acceptance', suites=suites, episodes_per_suite=args.episodes,
                    models=hashes, sources=source_hashes, candidate_profile=args.candidate_profile,training=False, physical_flight=False,
                    max_steps=600, primary='paired clean repeated episodes',
                    secondary=['first full passage', 'any crash', 'return to front'],
                    automatic_promotion=False, reserved_7M_seeds_used=False)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    policies = {'original':LearnedPolicy(paths['original'],SPECS['seenflag-s1-level1'])}
    if args.candidate_profile=='twelve121':
        from tools.evaluate_twelve_actions import TwelveActionPolicy
        policies[args.candidate_label]=TwelveActionPolicy(paths[args.candidate_label])
    else:
        policies[args.candidate_label]=LearnedPolicy(paths[args.candidate_label],
            PolicySpec(args.candidate_label,hashes[args.candidate_label],SPECS['seenflag-s1-level1'].lags))
    results = {}
    keys = ['clean_first_pass', 'crashed', 'clean_repeated_episode']
    for suite, base in suites.items():
        rows = {name: [] for name in policies}
        for i in range(args.episodes):
            for name, policy in policies.items():
                folder = args.out/name/suite
                folder.mkdir(parents=True, exist_ok=True)
                row = episode(policy, suite, base+i, folder/f'{base+i}.npz')
                rows[name].append(row)
                with (folder/'episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
            if (i+1) % 25 == 0:
                print(suite, i+1, flush=True)
        results[suite] = {
            'counts': {name: dict(**{key: sum(r[key] for r in values) for key in keys},
                                 returned=sum(r['returns_to_front']>0 for r in values),
                                 full_body_passes=sum(r['full_body_passes'] for r in values),
                                 endings=dict(Counter(r['outcome'] for r in values)))
                       for name, values in rows.items()},
            'paired': {key: dict(candidate_only=sum(b[key] and not a[key] for a,b in zip(rows['original'],rows[args.candidate_label])),
                                 original_only=sum(a[key] and not b[key] for a,b in zip(rows['original'],rows[args.candidate_label])))
                       for key in keys},
        }
        (args.out/'results.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results[suite]), flush=True)
    (args.out/'DONE').write_text(f'{6*args.episodes} development episodes completed; no acceptance or promotion.\n')


if __name__ == '__main__':
    main()
