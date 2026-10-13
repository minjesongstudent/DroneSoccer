import numpy as np

from src import baseline, contract

BLIND = contract.NOT_VISIBLE


def policy_input(frame, marker_seen=True, forward=0.0):
    frames = np.tile(frame, (contract.HISTORY_STEPS, 1))
    if marker_seen:
        frames[:, 7] = frame[6]
    else:
        frames[:, 7] = 0.0
    heights = np.ones(contract.HISTORY_STEPS, np.float32)
    actions = np.zeros((contract.HISTORY_STEPS, contract.ACTION_DIM), np.float32)
    actions[-1][1] = forward
    motion = np.tile(contract.NO_MOTION, (contract.HISTORY_STEPS, 1))
    return contract.policy_input(frames, heights, motion, actions, unseen=False)


def hoop_ahead():
    # Centred, level with the drone, round: straight ahead and lined up.
    return np.array([0.0, baseline.V_LEVEL, 0.3, 1.0, 0.0, 0.0, 1.0, 1.0], np.float32)


def test_search_turns_then_climbs():
    lapper = baseline.Lapper()
    ups = [lapper(policy_input(BLIND))[2] for _ in range(baseline.TURN_STEPS + baseline.CLIMB_STEPS)]
    assert not any(ups[:baseline.TURN_STEPS])
    assert all(u == 1.0 for u in ups[baseline.TURN_STEPS:])


def test_climbing_stops_after_max_climbs():
    lapper = baseline.Lapper()
    cycle = baseline.TURN_STEPS + baseline.CLIMB_STEPS
    ups = [lapper(policy_input(BLIND))[2] for _ in range(cycle * (baseline.MAX_CLIMBS + 2))]
    assert sum(u > 0 for u in ups) == baseline.CLIMB_STEPS * baseline.MAX_CLIMBS


def test_flies_at_a_hoop_with_its_marker():
    assert baseline.Lapper()(policy_input(hoop_ahead()))[1] > 0.9


def test_keeps_going_through_a_lost_hoop_when_committed():
    # The newest slot is the last command alone, not a 4-step mean.
    assert baseline.act(policy_input(BLIND, forward=1.0))[1] == 1.0


def test_holds_back_from_a_hoop_without_a_marker():
    # Blue junk, or the hoop's back: line up if you like, but do not fly at it.
    assert baseline.Lapper()(policy_input(hoop_ahead(), marker_seen=False))[1] == 0.0
