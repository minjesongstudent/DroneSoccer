import unittest
import numpy as np
from src.risk_replay import collision_approaches


class RiskReplayTests(unittest.TestCase):
    def test_boundaries_timeouts_and_partial_episode(self):
        done = [0, 1, 0, 1, 0, 0, 1, 0]
        timeout = [0, 0, 0, 1, 0, 0, 0, 0]
        np.testing.assert_array_equal(collision_approaches(done, timeout, 16), [0, 1, 4, 5, 6])

    def test_window_is_bounded(self):
        np.testing.assert_array_equal(collision_approaches([0]*20+[1], [0]*21, 3), [18,19,20])
