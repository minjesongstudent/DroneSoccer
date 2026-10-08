"""Exact local mid-episode state, never a reset or a physical-flight interface.

Only load locally collected, reviewed pickle payloads. Hashes bind an artifact
to its manifest; they do not make an untrusted pickle safe. The collector owns
the fresh-seed collision check and proof that the named policy issued actions.
This module checks allocation membership, state/history integrity and source
identity, and does not infer authenticity merely from successful passages.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from src import contract
from src.approach_curriculum import ApproachCurriculumEnv
from src.clearance_training import ClearanceTrainingEnv, PROFILE, public_features
from src.sim import HoopEnv

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'current-group-authentic-checkpoint-v1'
STATE_SOURCES = ('__init__.py', 'authentic_checkpoints.py', 'approach_curriculum.py',
                 'camera_history.py', 'clearance_training.py', 'contract.py',
                 'params.py', 'pass_audit.py', 'policy_runtime.py', 'sim.py')


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def source_hashes():
    return {'src/'+name: _sha((ROOT/'src'/name).read_bytes()) for name in STATE_SOURCES}


def _tree(value):
    """Value-exact state fingerprint including unconsumed RNG and deque order."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {'float': value.hex()}
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays are not checkpoint state')
        return {'array': value.dtype.str, 'shape': list(value.shape), 'bytes': value.tobytes().hex()}
    if isinstance(value, np.generic):
        return _tree(np.asarray(value))
    if isinstance(value, np.dtype):
        return {'dtype': value.str}
    if isinstance(value, np.random.Generator):
        return {'generator': type(value.bit_generator).__name__, 'state': _tree(value.bit_generator.state)}
    if isinstance(value, deque):
        return {'deque': [_tree(v) for v in value], 'maxlen': value.maxlen}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [_tree(v) for v in value]}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('Only string-keyed checkpoint state dictionaries supported')
        return {k: _tree(v) for k, v in sorted(value.items())}
    cls = type(value)
    if hasattr(value, '__dict__') and cls.__module__.startswith(('src.', 'gymnasium.spaces.')):
        return {'class': cls.__module__+'.'+cls.__qualname__, 'state': _tree(vars(value))}
    raise ValueError('Unsupported checkpoint state type: '+cls.__module__+'.'+cls.__qualname__)


def state_fingerprint(env):
    return _sha(_json(_tree(env)))


def assert_state_equal(left, right):
    if state_fingerprint(left) != state_fingerprint(right):
        raise AssertionError('Full simulator/wrapper state differs')


@dataclass(frozen=True)
class Checkpoint:
    payload: bytes
    manifest: bytes

    @property
    def metadata(self):
        return json.loads(self.manifest)

    @property
    def sha256(self):
        return _sha(self.payload)


@dataclass(frozen=True)
class RestoredCheckpoint:
    env: ClearanceTrainingEnv
    current_obs: np.ndarray
    public_prefix: np.ndarray


def _allocation(seeds):
    seeds = tuple(seeds)
    if (not seeds or any(type(s) is not int or not 11_000_000 <= s < 11_250_000 for s in seeds)
            or len(seeds) != len(set(seeds))):
        raise ValueError('Unique explicit FIT allocation required; development/acceptance forbidden')
    return seeds


def _validated(env, fit_seeds):
    if type(env) is not ClearanceTrainingEnv or type(env.env) not in (HoopEnv, ApproachCurriculumEnv):
        raise ValueError('Exact current ClearanceTrainingEnv + supported HoopEnv stack required')
    if env.trace_dir is not None:
        raise ValueError('Collector trace_dir must be None: restored state must never overwrite source logs')
    base = env.unwrapped
    if contract.LAPS_MAX_STEPS != 600:
        raise ValueError('Original 600-step horizon required')
    if not env._active or not 1 <= base.steps < 600:
        raise ValueError('Active post-step, pre-predict checkpoint required before original step600')
    if env.episode_seed not in fit_seeds or base.np_random_seed != env.episode_seed:
        raise ValueError('Source seed must belong to the declared fresh FIT allocation')
    k = base.steps
    if any(len(v) != k for v in env._trace.values()):
        raise ValueError('Checkpoint must include the entire uninterrupted source prefix')
    if any(env._trace['terminated']) or any(env._trace['truncated']):
        raise ValueError('Terminal prefix cannot be resumed')
    history = env.history
    if history.lags != PROFILE.lags or history.action_history != 3:
        raise ValueError('Original85 history contract required')
    encoded = np.concatenate([*(history.frames[-1-lag] for lag in history.lags),
                              *history.actions, [float(history.unseen)]]).astype(np.float32)
    current = np.asarray(env.current_obs)
    prior = np.asarray(env._trace['actor_observations'])
    following = np.asarray(env._trace['next_actor_observations'])
    if (current.shape != (85,) or current.dtype != np.float32
            or prior.shape != (k,85) or prior.dtype != np.float32
            or following.shape != (k,85) or following.dtype != np.float32
            or not all(np.isfinite(v).all() for v in (current, prior, following))
            or not np.array_equal(encoded, current)
            or not np.array_equal(following[-1], current)
            or not np.array_equal(prior[1:], following[:-1])
            or not np.array_equal(history.frames[-1], public_features(base._policy_input()))):
        raise ValueError('Public85 prefix/current/history state disagree')
    if env._last_info['steps'] != k or env._centre_count != base.passes:
        raise ValueError('Episode counter state disagrees')
    if len(base._pending) != base.dyn.delay_steps:
        raise ValueError('Pending command delay state disagrees')
    # These are independent detached copies. Prefix excludes current_obs, so a
    # recurrent adapter consumes prior frames, then current once in its runner.
    return current.copy(), prior.copy()


def capture_checkpoint(env, *, source_sha256, fit_seeds, source_profile='seenflag-s1-level1'):
    allocation = _allocation(fit_seeds)
    if source_profile != PROFILE.name or source_sha256 != PROFILE.sha256:
        raise ValueError('Original85 source profile/hash required')
    source = ROOT/'trainedModels'/(source_profile+'.zip')
    if _sha(source.read_bytes()) != source_sha256:
        raise ValueError('Original source checkpoint changed')
    current, prefix = _validated(env, allocation)
    base = env.unwrapped
    payload = pickle.dumps(dict(environment=env, current_obs=current, public_prefix=prefix),
                           protocol=pickle.HIGHEST_PROTOCOL)
    cleared_now = bool(env._trace['clearance_events'][-1] == 1)
    metadata = dict(version=VERSION, trusted_local_only=True, acceptance=False,
                    source_profile=source_profile, source_sha256=source_sha256,
                    source_checkpoint=str(source), code_sha256=source_hashes(),
                    fit_seeds=list(allocation), source_seed=env.episode_seed,
                    source_step=base.steps, remaining_steps=600-base.steps, max_original_steps=600,
                    prefix_rows=len(prefix), boundary='after_env_step_before_next_policy_predict',
                    prefix_excludes_current=True, historical_centre_passes=base.passes,
                    historical_full_body_passes=env.audit.count, historical_returns=env.audit.returns,
                    clearance_this_step=cleared_now,
                    eligible_first_or_second_clearance=bool(cleared_now and env.audit.count in (1,2) and base.steps <= 450),
                    payload_sha256=_sha(payload), state_sha256=state_fingerprint(env))
    return Checkpoint(payload, _json(metadata))


def _verify(checkpoint, trusted_local):
    if trusted_local is not True:
        raise ValueError('Only explicitly trusted local checkpoints may be deserialized')
    if type(checkpoint) is not Checkpoint or type(checkpoint.payload) is not bytes or type(checkpoint.manifest) is not bytes:
        raise ValueError('Immutable checkpoint bytes required')
    meta = checkpoint.metadata
    if (meta.get('version') != VERSION or meta.get('trusted_local_only') is not True
            or meta.get('source_profile') != PROFILE.name or meta.get('source_sha256') != PROFILE.sha256
            or meta.get('max_original_steps') != 600 or meta.get('acceptance') is not False
            or meta.get('prefix_excludes_current') is not True
            or meta.get('boundary') != 'after_env_step_before_next_policy_predict'):
        raise ValueError('Unsupported checkpoint manifest')
    _allocation(meta['fit_seeds'])
    # All trust/provenance checks precede pickle.loads. No untrusted payloads.
    if _sha(checkpoint.payload) != meta['payload_sha256']:
        raise ValueError('Checkpoint payload hash mismatch')
    if meta['code_sha256'] != source_hashes():
        raise ValueError('Checkpoint source code changed; do not import incompatible state')
    source = ROOT/'trainedModels'/(PROFILE.name+'.zip')
    if Path(meta['source_checkpoint']).resolve() != source.resolve() or _sha(source.read_bytes()) != PROFILE.sha256:
        raise ValueError('Checkpoint source model changed')
    return meta


def restore_checkpoint(checkpoint, *, trusted_local=False):
    meta = _verify(checkpoint, trusted_local)
    bundle = pickle.loads(checkpoint.payload)
    env = bundle['environment']
    current, prefix = _validated(env, tuple(meta['fit_seeds']))
    cleared_now = bool(env._trace['clearance_events'][-1] == 1)
    if (state_fingerprint(env) != meta['state_sha256']
            or env.episode_seed != meta['source_seed'] or env.unwrapped.steps != meta['source_step']
            or 600-env.unwrapped.steps != meta['remaining_steps']
            or len(prefix) != meta['prefix_rows']
            or env.unwrapped.passes != meta['historical_centre_passes']
            or env.audit.count != meta['historical_full_body_passes']
            or env.audit.returns != meta['historical_returns']
            or cleared_now != meta['clearance_this_step']
            or bool(cleared_now and env.audit.count in (1,2) and env.unwrapped.steps <= 450)
                != meta['eligible_first_or_second_clearance']
            or not np.array_equal(current, bundle['current_obs'])
            or not np.array_equal(prefix, bundle['public_prefix'])):
        raise ValueError('Restored checkpoint state/prefix does not match its manifest')
    return RestoredCheckpoint(env, current, prefix)


def write_checkpoint(path, checkpoint):
    """Create payload and manifest once; never overwrite existing evidence."""
    _verify(checkpoint, True)
    path = Path(path).resolve()
    manifest_path = path.with_suffix(path.suffix+'.json')
    if path.exists() or manifest_path.exists():
        raise FileExistsError('Fresh checkpoint paths required')
    with path.open('xb') as f:
        f.write(checkpoint.payload)
    with manifest_path.open('xb') as f:
        f.write(checkpoint.manifest)
    return dict(path=str(path), manifest_path=str(manifest_path),
                manifest_sha256=_sha(checkpoint.manifest), payload_sha256=checkpoint.sha256)


def read_checkpoint(path, *, expected_manifest_sha256, trusted_local=False):
    if trusted_local is not True:
        raise ValueError('Only explicitly trusted local checkpoints may be loaded')
    path = Path(path).resolve()
    metadata = path.with_suffix(path.suffix+'.json').read_bytes()
    if _sha(metadata) != expected_manifest_sha256:
        raise ValueError('Checkpoint manifest hash mismatch')
    checkpoint = Checkpoint(path.read_bytes(), metadata)
    _verify(checkpoint, True)
    return checkpoint
