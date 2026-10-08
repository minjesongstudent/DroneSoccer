"""Preflight orchestration guards; actor numerics are tested in the core suite."""
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.check_recurrent_memory_initialization import (
    Budget, FIXED_SUITES, InitializationBridge, SOURCE_PROFILE, SPECS,
    StopRequested, check_baseline_protocol, compare_trace, identical, write_json)


class FakePolicy:
    """No simulator or movement; isolates bridge state/control invariants."""
    def __init__(self):
        self.ring_position = SimpleNamespace(frame_count=0)
        self.observed = []

    def reset(self):
        self.ring_position.frame_count = 0

    def observe(self, public85):
        self.observed.append(public85.copy())
        self.ring_position.frame_count += 1
        result = np.zeros(150, np.float32)
        result[:85] = public85
        result[149] = float(self.ring_position.frame_count >= 3)
        return result

    def predict(self, features, deterministic=True):
        return np.array([.25, -.5, 0, .75], np.float32), None


class FakeSource:
    def __init__(self):
        self.inputs = []

    def predict(self, observation, deterministic=True):
        self.inputs.append(observation.copy())
        return np.array([.25, -.5, 0, .75], np.float32), None


class InitializationPreflightTests(unittest.TestCase):
    def test_bridge_checks_actual_predict_and_advances_once(self):
        policy, source = FakePolicy(), FakeSource()
        checks = []
        bridge = InitializationBridge(policy, source, lambda: checks.append(True))
        public = np.zeros(8, np.float32)
        previous = np.zeros(4, np.float32)
        for _ in range(3):
            features, previous = bridge.predict(public, previous)
            self.assertTrue(identical(features[:85], source.inputs[-1]))
        self.assertEqual(policy.ring_position.frame_count, 3)
        self.assertEqual(bridge.action_parity_checks, 3)
        self.assertEqual(len(checks), 7)  # reset, before and after each frame
        self.assertTrue(identical(source.inputs[-1][80:84], previous))
        bridge.reset()
        self.assertEqual(policy.ring_position.frame_count, 0)

    def test_stop_interrupts_before_observing_or_action(self):
        with tempfile.TemporaryDirectory() as temp:
            policy, source = FakePolicy(), FakeSource()
            budget = Budget(temp)
            bridge = InitializationBridge(policy, source, budget)
            (Path(temp) / 'STOP').touch()
            with self.assertRaises(StopRequested):
                bridge.predict(np.zeros(8, np.float32), np.zeros(4, np.float32))
            self.assertEqual(policy.ring_position.frame_count, 0)
            self.assertEqual(source.inputs, [])
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(StopRequested):
                Budget(temp, seconds=0)()

    def test_action_comparison_is_bitwise_including_signed_zero(self):
        policy, source = FakePolicy(), FakeSource()
        source.predict = lambda obs, deterministic=True: (
            np.array([.25, -.5, -0., .75], np.float32), None)
        bridge = InitializationBridge(policy, source, lambda: None)
        with self.assertRaisesRegex(ValueError, 'actual SAC predict action'):
            bridge.predict(np.zeros(8, np.float32), np.zeros(4, np.float32))

    def test_rejects_external_recurrence_advance(self):
        policy = FakePolicy()
        bridge = InitializationBridge(policy, None, lambda: None)
        policy.ring_position.frame_count = 1
        with self.assertRaisesRegex(ValueError, 'outside'):
            bridge.predict(np.zeros(8, np.float32), np.zeros(4, np.float32))

    def test_trace_comparison_rejects_changed_prefix_or_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            old_path, new_path = Path(temp) / 'old.npz', Path(temp) / 'new.npz'
            original = np.zeros((3, 85), np.float32)
            observed = np.zeros((3, 150), np.float32); observed[2, 149] = 1
            common = dict(actions=np.zeros((3, 4), np.float32), positions=np.zeros((3, 3)),
                          headings=np.zeros(3), hoop_centre=np.zeros(3),
                          hoop_normal=np.ones(3), drone_half_box=np.ones(3))
            np.savez(old_path, actor_observations=original, **common)
            np.savez(new_path, actor_observations=observed, **common)
            self.assertTrue(compare_trace(new_path, old_path)['prefix85_bitwise'])
            observed[0, 0] = 1
            np.savez(new_path, actor_observations=observed, **common)
            with self.assertRaisesRegex(ValueError, 'original85 prefixes'):
                compare_trace(new_path, old_path)
            observed[0, 0] = 0; common['positions'][1, 0] = 1
            np.savez(new_path, actor_observations=observed, **common)
            with self.assertRaisesRegex(ValueError, 'saved positions'):
                compare_trace(new_path, old_path)

    def test_protocol_rejects_changed_suite_and_reports_cannot_overwrite(self):
        profile = asdict(SPECS[SOURCE_PROFILE]); profile['lags'] = list(profile['lags'])
        protocol = dict(suites=dict(FIXED_SUITES), episodes=100, max_steps=600,
                        flight_release=False, training=False,
                        pass_metric='Full body past far noodle face; collision-free transit',
                        profiles={SOURCE_PROFILE: profile})
        check_baseline_protocol(protocol)
        protocol['suites']['aligned'] += 1
        with self.assertRaisesRegex(ValueError, 'semantics'):
            check_baseline_protocol(protocol)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'protocol.json'
            write_json(path, {'original': True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_json(path, {'replacement': True})
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
