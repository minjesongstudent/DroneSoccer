"""Bind guard penalties to each original warmup transition before CUDA training."""
import json
from pathlib import Path
import pickle
import shutil
import numpy as np
from src.sim import HoopEnv
from src.guard_alignment_reward import alignment_penalties
from tools.run_clearance_experiment import ROOT,sha,write_json


def main():
    parent=Path('D:/DroneSoccerData/runs/clearance-reward-20261007')
    out=Path('D:/DroneSoccerData/runs/guard-rewards-20261007');out.mkdir(exist_ok=False)
    path=parent/'clearance/step_200000/replay_buffer.pkl'
    with path.open('rb') as stream:replay=pickle.load(stream)
    assert replay.size()==200000 and not replay.full
    for name in ['sim.py','params.py','contract.py']:
        assert sha(parent/'source/src'/name)==sha(ROOT/'src'/name)
    traces=parent/'clearance/training_trajectories';offset=0;pieces=[];hashes={str(path):sha(path)}
    for line in (traces/'episodes.jsonl').read_text().splitlines():
        row=json.loads(line);file=traces/row['trace'];hashes[str(file)]=sha(file)
        env=HoopEnv(level=1,randomise=True,hanger=True);env.reset(seed=row['seed'])
        with np.load(file) as z:
            n=min(len(z['actions']),200000-offset);sl=slice(offset,offset+n)
            for key,field in [('actions','actions'),('actor_observations','observations'),('next_actor_observations','next_observations')]:
                np.testing.assert_array_equal(z[key][:n],getattr(replay,field)[sl,0])
            np.testing.assert_allclose(z['training_rewards'][:n],replay.rewards[sl,0],rtol=1e-6,atol=1e-6)
            np.testing.assert_array_equal(env.hoop_centre,z['hoop_centre'])
            np.testing.assert_array_equal(env.hoop_normal,z['hoop_normal'])
            np.testing.assert_array_equal(env._half_box,z['drone_half_box'])
            pos=np.vstack([env.pos,z['positions'][:n-1]])
            headings=np.r_[env.heading,z['headings'][:n-1]]
            pieces.append(alignment_penalties(env,pos,headings))
            offset+=n
        env.close()
        if offset==200000:break
    assert offset==200000
    penalties=np.concatenate(pieces).astype(np.float32)
    assert np.isfinite(penalties).all() and penalties.min()>=-.200001 and penalties.max()<=0
    np.save(out/'penalties.npy',penalties)
    write_json(out/'report.json',dict(transitions=offset,nonzero=int((penalties<0).sum()),
        minimum=float(penalties.min()),mean=float(penalties.mean()),total=float(penalties.sum()),
        replay_sha256=sha(path),penalties_sha256=sha(out/'penalties.npy'),input_hashes=hashes))
    shutil.copy2(__file__,out/'preparation_source.py')
    shutil.copy2(ROOT/'src/guard_alignment_reward.py',out/'reward_source.py')
    print((out/'report.json').read_text()[:500])


if __name__=='__main__':main()
