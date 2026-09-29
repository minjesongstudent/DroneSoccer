"""Versioned, trusted-local successful-flight continuation bank.

Pickles must come from the locally verified collection, never outside sources.
Reserved layouts are available only through an explicit evaluation loader.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from src.return_practice import ReturnSnapshotLibrary, ReturnPracticeEnv
from src.return_geometry import GeometricReturnProgress
from src.ring_estimate import MetricRingObservation
from src.front_memory import FrontMemoryObservation
from src.mission_observation import MissionObservation

VERSION = 'success-history-return-bank-v1'
LANDMARKS = ('near_pass', 'after_cross', 'pre_cross', 'early_return', 'postpass')
FIT = (2810249, 2810252, 2810296, 2810312)
CHECK = (2810380, 2810482)
BANK_SPLITS = {
    VERSION: (FIT, CHECK),
    'success-history-return-bank-v2': (
        (2820026, 2820664, 2820750, 2820805, 2820828, 2821928, 2821958, 2822002, 2822042),
        (2821035, 2821163)),
}
STATE_FILES = ('sim.py', 'params.py', 'contract.py', 'curriculum.py',
               'guided_curriculum.py', 'mission_observation.py',
               'observation_memory.py', 'front_memory.py', 'ring_estimate.py',
               'return_geometry.py')


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SuccessHistoryLibrary(ReturnSnapshotLibrary):
    def __init__(self, directory, *, split='fit', evaluation_only=False):
        if split not in ('fit', 'internal_check'):
            raise ValueError('Unknown bank split')
        if split == 'internal_check' and not evaluation_only:
            raise ValueError('Reserved layouts require evaluation_only=True')
        self.directory = Path(directory)
        raw = (self.directory / 'bank_manifest.json').read_bytes()
        manifest = json.loads(raw)
        protocol = json.loads((self.directory / 'protocol.json').read_bytes())
        version = manifest['version']
        if version not in BANK_SPLITS or protocol['version'] != version:
            raise ValueError('Bank version mismatch')
        fit, check = BANK_SPLITS[version]
        self.version = version
        self.fit_seeds, self.check_seeds = fit, check
        if tuple(protocol['fit_seeds']) != fit or tuple(protocol['internal_check_seeds']) != check:
            raise ValueError('Preregistered layout split changed')
        source = Path(manifest['snapshot_source']) / 'src'
        for name in STATE_FILES:
            if (source / name).read_bytes() != (Path(__file__).parent / name).read_bytes():
                raise ValueError('Snapshot simulator source differs: ' + name)
        expected = {(seed, mark) for seed in fit + check for mark in LANDMARKS}
        rows = manifest['snapshots']
        if len(rows) != len(expected) or {(r['seed'], r['landmark']) for r in rows} != expected:
            raise ValueError('Incomplete or duplicate bank records')
        self.records, self.payloads = [], {}
        self.split, self.evaluation_only = split, evaluation_only
        self.manifest = dict(minimum_remaining_steps=200, source_sha256=sha(raw))
        for row in rows:
            assigned = 'fit' if row['seed'] in fit else 'internal_check'
            filename = row['file']
            if row['split'] != assigned or Path(filename).name != filename:
                raise ValueError('Invalid split or snapshot path')
            if assigned != split:
                continue  # Never deserialize reserved layouts for fitting.
            payload = (self.directory / 'snapshots' / filename).read_bytes()
            if sha(payload) != row['sha256']:
                raise ValueError('Snapshot checksum mismatch')
            self._validate(payload, row)
            if filename in self.payloads:
                raise ValueError('Duplicate snapshot path')
            self.payloads[filename] = payload
            self.records.append(dict(row, id=filename, phase=row['landmark'],
                                     source_step=row['step'], historical_passes=1))
        self.phases = LANDMARKS

    @staticmethod
    def _validate(payload, row):
        saved = pickle.loads(payload)
        env, obs = saved['environment'], saved['observation']
        if not isinstance(env, GeometricReturnProgress) or not isinstance(env.env, MetricRingObservation):
            raise ValueError('Wrong reward/observation wrappers')
        front = env.env.env
        if not isinstance(front, FrontMemoryObservation) or not isinstance(front.env, MissionObservation):
            raise ValueError('Missing authentic camera history')
        mission = front.env
        base = env.unwrapped
        if (saved['source_seed'] != row['seed'] or saved['source_step'] != row['step']
                or saved['split'] != row['split'] or saved['landmark'] != row['landmark']
                or base.np_random_seed != row['seed'] or base.steps != row['step']
                or not 0 < base.steps <= 400 or base.passes != 1
                or base.guided_stage != 4 or mission.single_pass):
            raise ValueError('Snapshot provenance or remaining horizon mismatch')
        rebuilt = mission._append(mission.env.memory.encode(base._policy_input()))
        rebuilt = env.env.observation(front._append(rebuilt))
        if obs.shape != (119,) or not np.isfinite(obs).all() or obs[99] != 0 or not np.array_equal(obs, rebuilt):
            raise ValueError('Observation and authentic history disagree')
        return saved

    def restore(self, identifier):
        row = next(r for r in self.records if r['id'] == identifier)
        saved = self._validate(self.payloads[identifier], row)
        saved['info'] = dict(outcome='running', stage=4)
        return saved


class SuccessHistoryPractice(ReturnPracticeEnv):
    """Score only new passages followed by survival of the entire local horizon."""
    def __init__(self, library, horizon=200, *, evaluation=False):
        if not isinstance(library, SuccessHistoryLibrary):
            raise TypeError('Validated success-history library required')
        if library.evaluation_only and not evaluation:
            raise ValueError('Evaluation bank cannot supply training resets')
        super().__init__(library, horizon)

    def _info(self, info):
        result = super()._info(info)
        result['practice_success'] = bool(result['practice_success'] and self.elapsed == self.horizon)
        result['bank_version'] = self.library.version
        result['bank_split'] = self.library.split
        return result
