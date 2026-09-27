"""Recency preserves coverage, real sequence boundaries and saved contracts."""
import pickle
import numpy as np
import pytest
import torch
from src.recent_retrace import RecentLapReplay, RecentRetraceSAC, RECENT_LAP_COUNT
from src.retrace_replay import RetraceReplayBuffer
from test_retrace_replay import add, OBSERVATION_SPACE, ACTION_SPACE
from test_retrace_policy import CountingEnv


def make_buffer(capacity=4096):
    return RecentLapReplay(capacity, OBSERVATION_SPACE, ACTION_SPACE, device='cpu')


def test_mixture_keeps_all_old_laps_and_single_passes_with_exact_recent_population(monkeypatch):
    replay = make_buffer()
    # Circular overwrite: physical slot order must not be mistaken for age.
    for serial in range(5000):
        add(replay, serial, task=int(serial % 10 == 0))
    populations = []
    original = np.random.choice
    def capture(population, size):
        populations.append((population.copy(), size))
        return original(population, size)
    monkeypatch.setattr(np.random, 'choice', capture)
    data = replay.sample(256)
    assert [size for _, size in populations] == [128, 64, 64]
    single, all_laps, recent = [p for p, _ in populations]
    assert replay._single_pass_slots[single].all()
    assert not replay._single_pass_slots[all_laps].any()
    expected = all_laps[np.argsort(replay.insertion_ids[all_laps])[-RECENT_LAP_COUNT:]]
    np.testing.assert_array_equal(recent, expected)
    assert replay.insertion_ids[all_laps].min() < replay.insertion_ids[recent].min()
    assert int((data.observations[:,99]>.5).sum()) == 128
    for row in range(256):
        ids = data.seq_insertion_ids[row][data.seq_valid[row]].numpy()
        np.testing.assert_array_equal(np.diff(ids), np.ones(len(ids)-1))


@pytest.mark.parametrize('task', [0,1])
def test_one_task_and_short_population_are_supported(task):
    replay = make_buffer(8)
    for serial in range(3):
        add(replay, serial, task=task, done=serial==1, timeout=serial==1)
    data = replay.sample(17)
    assert data.observations.shape == (17,100)
    assert (data.observations[:,99]==task).all()
    assert data.seq_valid.sum(1).max() <= 2


def test_invalid_sampling_is_rejected():
    replay = make_buffer(8)
    with pytest.raises(ValueError, match='empty'):
        replay.sample(1)
    for batch in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match='integer'):
            replay.sample(batch)


def test_saved_sampling_contract_rejects_old_replay_transactionally(tmp_path):
    torch.set_num_threads(1)
    model = RecentRetraceSAC('MlpPolicy', CountingEnv(), device='cpu', learning_starts=0,
        buffer_size=16, policy_kwargs={'net_arch':[8,8]})
    original = model.replay_buffer
    old = RetraceReplayBuffer(16, OBSERVATION_SPACE, ACTION_SPACE, device='cpu')
    with (tmp_path/'old.pkl').open('wb') as f:
        pickle.dump(old,f)
    with pytest.raises(ValueError, match='Recent replay contract'):
        model.load_replay_buffer(tmp_path/'old.pkl')
    assert model.replay_buffer is original
    model.save(tmp_path/'model')
    restored = RecentRetraceSAC.load(tmp_path/'model',device='cpu')
    assert isinstance(restored.replay_buffer, RecentLapReplay)
    restored.replay_buffer.recent_lap_count = 7
    with pytest.raises(ValueError, match='Recent replay contract'):
        restored._validate_recent_replay()
