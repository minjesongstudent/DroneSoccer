"""Read-only heldout terminal calibration; no optimization or simulator movement."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import torch
from stable_baselines3 import SAC

from src.constrained_sac import episode_partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    runs = Path('D:/DroneSoccerData/runs')
    replay = runs/'clearance-reward-20261007/clearance/step_200000/replay_buffer.pkl'
    with replay.open('rb') as stream:
        data = pickle.load(stream)
    fit, held, _ = episode_partition(data.dones[:data.pos])
    terminal = (data.dones[:data.pos,0] > .5) & (data.timeouts[:data.pos,0] < .5)
    models = dict(source=root/'trainedModels/seenflag-s1-level1.zip',
                  uniform=runs/'constrained-boundary-20261007/critic_005000/model.zip',
                  risk=runs/'critic-risk-20261007/critic_005000/model.zip')
    result = dict(note='Terminal realized reward, not expected value ground truth under aliased camera histories. No bootstrap or entropy term at true terminal.',
                  replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(), models={})
    for name, path in models.items():
        model = SAC.load(path, device='cpu')
        metrics = {}
        for split, ids in [('fit',fit),('heldout',held)]:
            ids = ids[terminal[ids]]
            obs = torch.as_tensor(data.observations[ids,0])
            actions = torch.as_tensor(data.actions[ids,0])
            actual = data.rewards[ids,0]
            with torch.no_grad():
                q = torch.cat(model.critic(obs,actions),dim=1).min(dim=1).values.numpy()
            error = q-actual
            metrics[split] = dict(count=len(ids), reward_mean=float(actual.mean()),
                                 q_mean=float(q.mean()), bias=float(error.mean()),
                                 rmse=float(np.sqrt(np.mean(error**2))),
                                 median_abs_error=float(np.median(abs(error))))
            np.savez_compressed(args.out/f'{name}-{split}.npz',indices=ids,q=q,reward=actual)
        result['models'][name] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metrics=metrics)
    log = runs/'clearance-reward-20261007/clearance/training_trajectories/episodes.jsonl'
    rows=[]; total=0
    for line in log.read_text().splitlines():
        row=json.loads(line)
        if total+row['steps']>200000: break
        total+=row['steps'];rows.append(row)
    result['warmup_complete_episodes']=dict(episodes=len(rows),transitions=total,
        with_full_pass=sum(r['full_body_passes']>0 for r in rows),
        with_repeat=sum(r['full_body_passes']>=2 for r in rows),
        crashed=sum(r['outcome'] in ['hit','net','wrong_way'] for r in rows))
    shutil.copy2(__file__,args.out/'audit_source.py')
    (args.out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
