"""Versioned recent/all lap replay mixture; all flight actions remain learned."""
import json
from pathlib import Path
import zipfile

import numpy as np

from src.retrace_policy import RetraceSAC, start_from_checkpoint as transfer_actor
from src.retrace_replay import RetraceReplayBuffer

VERSION = 'recent-and-all-lap-soft-retrace-v1'
RECENT_LAP_COUNT = 2048


class RecentLapReplay(RetraceReplayBuffer):
    """Half single-pass; lap half splits equally between all and recent laps.

    Recency is defined by insertion order, never reward, phase or outcome.
    Selecting anchors does not change their recorded successor sequences.
    All older retained lap transitions remain eligible for uniform sampling.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_version = VERSION
        self.recent_lap_count = RECENT_LAP_COUNT

    def sample(self, batch_size, env=None):
        if env is not None:
            raise ValueError('Recent Retrace requires raw observations.')
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('Positive integer batch size required.')
        count = self.size()
        if not count:
            raise ValueError('Cannot sample empty replay.')
        labels = self._single_pass_slots[:count]
        single, laps = np.flatnonzero(labels), np.flatnonzero(~labels)
        n_single = batch_size//2 if len(single) and len(laps) else (batch_size if len(single) else 0)
        n_lap = batch_size-n_single
        chunks = []
        if n_single:
            chunks.append(np.random.choice(single, n_single))
        if n_lap:
            recent = laps[np.argsort(self.insertion_ids[laps])[-self.recent_lap_count:]]
            n_recent = n_lap//2
            chunks.extend([np.random.choice(laps, n_lap-n_recent), np.random.choice(recent, n_recent)])
        indices = np.concatenate(chunks)
        np.random.shuffle(indices)
        return self._get_samples(indices, env=None)


class RecentRetraceSAC(RetraceSAC):
    def __init__(self, *args, **kwargs):
        self.recent_sampling_version = VERSION
        kwargs.setdefault('replay_buffer_class', RecentLapReplay)
        super().__init__(*args, **kwargs)

    def _validate_recent_replay(self):
        replay = self.replay_buffer
        if (self.recent_sampling_version != VERSION or not isinstance(replay, RecentLapReplay)
                or replay.sampling_version != VERSION or replay.recent_lap_count != RECENT_LAP_COUNT):
            raise ValueError('Recent replay contract differs; use a fresh experiment.')

    def _setup_model(self):
        super()._setup_model()
        self._validate_recent_replay()

    @classmethod
    def load(cls, path, *args, **kwargs):
        position = path.tell() if hasattr(path, 'tell') else None
        source = path
        if isinstance(path, (str, Path)) and not Path(path).exists():
            source = Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                if json.loads(archive.read('data')).get('recent_sampling_version') != VERSION:
                    raise ValueError('Recent replay model requires its saved sampling version.')
        finally:
            if position is not None:
                path.seek(position)
        return super().load(path, *args, **kwargs)

    def load_replay_buffer(self, *args, **kwargs):
        previous = self.replay_buffer
        try:
            super().load_replay_buffer(*args, **kwargs)
            self._validate_recent_replay()
        except Exception:
            self.replay_buffer = previous
            raise


def start_from_checkpoint(path, env, *, device, seed, buffer_size, tensorboard_log):
    model, metadata = transfer_actor(path, env, device=device, seed=seed,
                                    buffer_size=buffer_size, tensorboard_log=tensorboard_log)
    assert model.num_timesteps == 0 and model.replay_buffer.size() == 0
    model.__class__ = RecentRetraceSAC
    model.recent_sampling_version = VERSION
    model.replay_buffer_class = RecentLapReplay
    model.replay_buffer = RecentLapReplay(buffer_size, model.observation_space, model.action_space,
        device=model.device, n_envs=1, optimize_memory_usage=False, sequence_length=model.trace_length)
    model._validate_recent_replay()
    metadata.update(policy_update_rule=VERSION, recent_sampling_version=VERSION,
                    recent_lap_count=RECENT_LAP_COUNT,
                    replay_sampling='50% all single-pass; 25% all lap; 25% latest2048 lap insertions')
    return model, metadata
