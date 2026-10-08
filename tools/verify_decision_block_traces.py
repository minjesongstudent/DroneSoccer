"""Verify saved block/tail joins and recompute finite soft outcome labels."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from tools.audit_critic_rankings import soft_return
from tools.run_clearance_experiment import write_json


def main():
    folder = Path('D:/DroneSoccerData/runs/decision-block-20261007')
    torch.set_num_threads(1)
    model = SAC.load(folder / 'model.zip', device='cpu')
    alpha = float(model.log_ent_coef.detach().exp())
    rows = [json.loads(line) for line in (folder / 'continuations.jsonl').read_text().splitlines()]
    joined = empty = 0
    for row in rows:
        state, candidate, replica = row['state'], row['candidate'], row['replica']
        base = folder / f'state_{state:03d}'
        with np.load(base / f'block_{candidate}.npz') as prefix, np.load(base / f'block_{candidate}_tail_{replica}.npz') as tail:
            assert len(prefix['actions']) == row['prefix_steps']
            assert len(tail['actions']) == row['tail_steps']
            for trace in [prefix, tail]:
                for field in ['observations', 'actions', 'rewards', 'logps', 'positions', 'next_observation']:
                    assert np.isfinite(trace[field]).all()
                assert (np.abs(trace['actions']) <= 1).all()
            if len(tail['actions']):
                np.testing.assert_array_equal(prefix['next_observation'], tail['observations'][0])
                assert not row['prefix_terminal']
                joined += 1
            else:
                assert row['prefix_terminal']
                empty += 1
            rewards = np.concatenate([prefix['rewards'], tail['rewards']])
            logps = np.concatenate([prefix['logps'], tail['logps']])
            actual = soft_return(rewards, logps, model.gamma, alpha)
            assert np.isclose(actual, row['finite_soft_return'], atol=1e-8, rtol=1e-10)
    result = dict(rows=len(rows), exact_joins=joined, terminal_prefix_rows=empty,
                  finite_return_recomputations=len(rows), actor_updates=0)
    write_json(Path('D:/DroneSoccerData/runs/decision-block-analysis-20261007') / 'trace_verification.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
