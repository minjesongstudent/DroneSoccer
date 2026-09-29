"""Authentic115D return snapshots from complete learned-policy flights."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from src.anchored_policy import AnchoredSAC
from src.front_memory import FrontMemoryObservation, FrontSightingMemory, VERSION as FRONT_VERSION
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation, VERSION as MISSION_VERSION
from src.return_practice import (ReturnSnapshotLibrary, SEED_RANGES, ALLOWED_PHASES,
                                 REWARD_VERSION, LAPS_MAX_STEPS, _sha, _code_hashes)

VERSION = 'authentic-front-memory-return-snapshots-v1'
OBSERVATION_VERSION = MISSION_VERSION+'+'+FRONT_VERSION


def code_hashes():
    values = _code_hashes()
    values['front_memory.py'] = _sha(Path(__file__).with_name('front_memory.py').read_bytes())
    return values


def restore_payload(payload):
    result = pickle.loads(payload)  # trusted local artifacts only
    env, observation = result['environment'], result['observation']
    if (not isinstance(env, FrontMemoryObservation) or not isinstance(env.env, MissionObservation)
            or not isinstance(env.unwrapped, GuidedLessonEnv)
            or not isinstance(env.front_memory, FrontSightingMemory)
            or env.env.single_pass or env.unwrapped.guided_stage != 4):
        raise ValueError('Front snapshots require the complete authentic115D wrapper stack.')
    if observation.shape != (115,) or observation.dtype != np.float32 or not np.isfinite(observation).all():
        raise ValueError('Front snapshot requires finite115D float32 observations.')
    inner = env.env
    original = inner._append(inner.env.memory.encode(env.unwrapped._policy_input()))
    reconstructed = env._append(original)
    if not np.array_equal(reconstructed, observation) or observation[99] != 0:
        raise ValueError('Front snapshot observation and preserved history disagree.')
    if env.front_memory._has_seen and not (env.front_memory._last[6]>.5 and env.front_memory._last[7]>.5):
        raise ValueError('Front memory anchor must be an actual front-marker detection.')
    return result


class FrontReturnLibrary(ReturnSnapshotLibrary):
    version = VERSION
    observation_version = OBSERVATION_VERSION
    observation_dim = 115
    code_hashes = staticmethod(code_hashes)
    restore_payload = staticmethod(restore_payload)


def generate(checkpoint, output, *, split, episodes, seed_start=None, horizon=200):
    if split not in SEED_RANGES:
        raise ValueError('Separate train/development partitions required.')
    low, high = SEED_RANGES[split]
    seed_start = low if seed_start is None else seed_start
    if (isinstance(episodes,bool) or int(episodes)!=episodes or episodes<=0
            or int(seed_start)!=seed_start or not low<=seed_start<seed_start+episodes<=high
            or isinstance(horizon,bool) or int(horizon)!=horizon or not 1<=horizon<LAPS_MAX_STEPS):
        raise ValueError('Invalid collection seed range, episode count or horizon.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback.')
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    source_hash, hashes = _sha(checkpoint.read_bytes()), code_hashes()
    model = AnchoredSAC.load(checkpoint,device='cuda')
    if model.observation_space.shape!=(100,):
        raise ValueError('Collection uses the preserved100D source policy.')
    env = FrontMemoryObservation(MissionObservation(GuidedLessonEnv(stage=4),single_pass=False))
    output.mkdir(parents=True,exist_ok=False)
    manifest = dict(version=VERSION,split=split,reward_version=REWARD_VERSION,
        observation_version=OBSERVATION_VERSION,observation_dim=115,
        source_actor_observation_dim=100,source_checkpoint=str(checkpoint),source_sha256=source_hash,
        source_steps=int(model.num_timesteps),policy_mode='deterministic',
        seed_start=int(seed_start),seed_end=int(seed_start+episodes-1),attempted_episodes=int(episodes),
        requested_phases=list(ALLOWED_PHASES),minimum_remaining_steps=int(horizon),code_sha256=hashes,
        geometry={name:asdict(getattr(env.unwrapped,name)) for name in ('cage','drone','hoop','camera','pad')},
        snapshots=[],new_demonstrations=0,
        memory_provenance='Full source-policy flight from normal reset; detections and issued commands only')
    outcomes,rejections=Counter(),Counter()
    try:
        with torch.inference_mode():
            for seed in range(int(seed_start),int(seed_start+episodes)):
                observation,_=env.reset(seed=seed)
                previous=env.unwrapped._lap_phase
                seen=set()
                while True:
                    action,_=model.predict(observation[:100],deterministic=True)
                    observation,_,terminated,truncated,info=env.step(action)
                    base=env.unwrapped
                    phase=base._lap_phase
                    if phase in ALLOWED_PHASES and phase!=previous and phase not in seen:
                        seen.add(phase)
                        if terminated or truncated or base.passes<1 or LAPS_MAX_STEPS-base.steps<horizon:
                            rejections['ineligible_phase_entry']+=1
                        else:
                            payload=pickle.dumps(dict(environment=env,observation=observation.copy(),info=deepcopy(info)),
                                                 protocol=pickle.HIGHEST_PROTOCOL)
                            restore_payload(payload)  # validate before saving
                            identifier=f'snapshot-{len(manifest["snapshots"]):04d}'
                            filename=identifier+'.pkl'
                            (output/filename).write_bytes(payload)
                            manifest['snapshots'].append(dict(id=identifier,file=filename,sha256=_sha(payload),
                                phase=phase,seed=seed,source_step=int(base.steps),historical_passes=int(base.passes)))
                    previous=phase
                    if terminated or truncated:
                        outcomes[info['outcome']]+=1
                        break
                print(json.dumps(dict(split=split,seed=seed,snapshots=len(manifest['snapshots']))),flush=True)
    finally:
        env.close()
    if _sha(checkpoint.read_bytes())!=source_hash or code_hashes()!=hashes:
        raise RuntimeError('Source changed during collection; do not use incomplete bank.')
    manifest.update(phase_counts=dict(Counter(r['phase'] for r in manifest['snapshots'])),
                    rejected_entries=dict(rejections),episode_outcomes=dict(outcomes))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    verified=FrontReturnLibrary(output)
    print(json.dumps(dict(bank=str(output),snapshots=len(verified.records),sha256=verified.bank_sha256)))
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--split',choices=('train','development'),required=True)
    parser.add_argument('--episodes',type=int,required=True)
    args=parser.parse_args()
    generate(args.checkpoint,args.output,split=args.split,episodes=args.episodes)
