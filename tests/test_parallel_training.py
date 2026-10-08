"""Stdlib Windows lifetime/authorization fixtures; no CUDA or neural execution."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from tools.parallel_training import (ARMS, CohortProcesses, WorkerLease, _kernel, digest,
                                     process_identity, validate_manifest, worker_lease)
from tools.run_parallel_estimated_motion import existing_allocation, round_one_fairness, select_one


ROOT = Path(__file__).resolve().parents[1]


def make_manifest(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    allocation = out/'allocation.json'; allocation.write_text('{}')
    data = dict(format='dronesoccer-authorized-cohort-v1', cohort_id='lifetime-fixture', fixture=True,
        parent_pid=os.getpid(), parent_creation_time=process_identity(os.getpid()),
        max_workers=3, arm_order=list(ARMS), smoke=True, project_root=str(ROOT), source_sha256={},
        allocation_record=dict(path=str(allocation), sha256=digest(allocation)),
        arms={arm: dict(out=str(out/'arms'/arm), target_mode=arm, memory_fraction=.15,
            algorithm_seed=11592000, training_seeds=[11244900, 11244901], max_total_seconds=600) for arm in ARMS})
    path = out/'cohort_manifest.json'; path.write_text(json.dumps(data))
    return path, data


def wait_for(predicate, seconds=10):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError('Bounded fixture wait expired')


@unittest.skipUnless(os.name == 'nt', 'Windows process lifetime implementation')
class ParallelTrainingTests(unittest.TestCase):
    def test_manifest_alone_cannot_authorize_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            path, data = make_manifest(Path(temp))
            with self.assertRaisesRegex(RuntimeError, 'ARM grant'):
                with worker_lease(path, 'full_horizon', data['arms']['full_horizon']['out'], True):
                    self.fail('Unregistered worker acquired a lease')

    def test_three_slots_parent_lock_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest, _ = make_manifest(root/'cohort'); lock = root/'trainer.lock'
            with CohortProcesses(lock, deadline=time.monotonic()+30) as pool:
                children = [pool.launch(manifest, arm, fixture=True) for arm in ARMS]
                wait_for(lambda: all((root/'cohort/arms'/arm/'started.json').exists() for arm in ARMS))
                self.assertEqual(len({child.worker_pid for child in children}), 3)
                with self.assertRaises(ValueError): pool.launch(manifest, ARMS[0], fixture=True)
                with self.assertRaises(ValueError): pool.launch(manifest, 'fourth', fixture=True)
                # The exact existing byte0 lock is held against another process.
                code = ('from tools.parallel_training import project_lock; '
                        f'c=project_lock({str(lock)!r}); c.__enter__()')
                attempt = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, timeout=10)
                self.assertNotEqual(attempt.returncode, 0)
                (root/'cohort/STOP').touch()
                for child in children: self.assertEqual(child.process.wait(timeout=10), 0)
                self.assertTrue(all((root/'cohort/arms'/arm/'stopped.json').exists() for arm in ARMS))
            from tools.parallel_training import project_lock
            with project_lock(lock): pass

    def test_parent_death_kills_registered_worker_before_lock_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); output = root/'cohort'; lock = root/'trainer.lock'
            code = '\n'.join([
                'import json,time,os', 'from pathlib import Path',
                'from tests.test_parallel_training import make_manifest',
                'from tools.parallel_training import CohortProcesses',
                f'p,d=make_manifest(Path({str(output)!r}))',
                f'with CohortProcesses({str(lock)!r},deadline=time.monotonic()+60) as pool:',
                " c=pool.launch(p,'full_horizon',fixture=True)",
                f" Path({str(root/'coordinator.json')!r}).write_text(json.dumps(dict(parent=os.getpid(),child=c.worker_pid)))",
                ' time.sleep(60)'])
            helper = subprocess.Popen([sys.executable, '-c', code], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            kernel = _kernel(); child_handle = parent_handle = None
            try:
                wait_for(lambda: (root/'coordinator.json').exists())
                identities = json.loads((root/'coordinator.json').read_text())
                child_handle = kernel.OpenProcess(0x100000, False, identities['child'])
                parent_handle = kernel.OpenProcess(0x100001, False, identities['parent'])
                self.assertTrue(child_handle and parent_handle)
                kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                kernel.TerminateProcess.restype = ctypes.c_int
                self.assertTrue(kernel.TerminateProcess(parent_handle, 77))
                self.assertEqual(kernel.WaitForSingleObject(parent_handle, 10000), 0)
                self.assertEqual(kernel.WaitForSingleObject(child_handle, 10000), 0)
                helper.wait(timeout=10)
                from tools.parallel_training import project_lock
                with project_lock(lock): pass
            finally:
                if helper.poll() is None: helper.kill(); helper.wait(timeout=10)
                if child_handle: kernel.CloseHandle(child_handle)
                if parent_handle: kernel.CloseHandle(parent_handle)
                helper.stdout.close(); helper.stderr.close()

    def test_same_job_assignment_is_idempotent_and_worker_budget_is_tighter(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); manifest,data=make_manifest(root/'cohort')
            with CohortProcesses(root/'lock', deadline=time.monotonic()+30) as pool:
                child=pool.launch(manifest,'full_horizon',fixture=True)
                pool.job.assign(child.worker_pid)  # exact membership, no access-denied fallback bypass
                (root/'cohort/STOP').touch(); child.process.wait(timeout=10)
            grant=dict(parent_pid=os.getpid(), seconds_remaining=900, arm='full_horizon',
                       cohort_manifest_sha256=digest(manifest))
            (root/'cohort/STOP').unlink()
            lease=WorkerLease(grant,data,manifest)
            try: self.assertLessEqual(lease._parent_deadline-time.monotonic(),600)
            finally: lease.close()

    def test_prior_failed_cohort_reservation_blocks_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path,data=make_manifest(root/'failed')
            with self.assertRaisesRegex(ValueError,'overlaps'):
                existing_allocation(root,[11244900,11244901])

    def test_rollout_barrier_compares_semantics_and_final_tie_order(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); path,data=make_manifest(root)
            self.assertEqual(round_one_fairness(root)['status'],'waiting')
            for arm in ARMS:
                folder=root/'arms'/arm;folder.mkdir(parents=True)
                np.savez_compressed(folder/'rollout_01.npz',features89=np.zeros((3,89),np.float32),
                    actions=np.zeros((3,4),np.float32),episode_lengths=np.array([3]),
                    rewards=np.zeros(3,np.float32),old_log_prob=np.zeros(3,np.float32),
                    returns=np.full(3,ARMS.index(arm),np.float32))
                (folder/'episode_endpoints_01.json').write_text('[{"steps":3}]')
                ready=dict(arm=arm,cohort_manifest_sha256=digest(path),
                    rollout_sha256=digest(folder/'rollout_01.npz'),
                    endpoints_sha256=digest(folder/'episode_endpoints_01.json'))
                (folder/'.first_rollout_ready.json').write_text(json.dumps(ready))
            self.assertTrue(round_one_fairness(root)['passed'])  # differing targets are deliberately excluded
            folder=root/'arms/mission';(folder/'episode_endpoints_01.json').write_text('[{"steps":4}]')
            ready=json.loads((folder/'.first_rollout_ready.json').read_text())
            ready['endpoints_sha256']=digest(folder/'episode_endpoints_01.json')
            (folder/'.first_rollout_ready.json').write_text(json.dumps(ready))
            self.assertEqual(round_one_fairness(root)['status'],'confounded')
        candidates=[dict(arm=arm,counts=dict(randomised=dict(clean_repeat=55,crashes=45,first_pass=77))) for arm in reversed(ARMS)]
        self.assertEqual(select_one(candidates)['arm'],'full_horizon')
        self.assertIsNone(select_one([]))


class AllocationCompatibilityTests(unittest.TestCase):
    def test_legacy_named_arms_keep_root_seed_reservations(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'protocol.json'
            path.write_text(json.dumps(dict(arms=['control', 'clearance'],
                reset_seed_base=11244000, iterations=1, episodes_per_iteration=2)))
            self.assertEqual(existing_allocation(temp, [11244900,11244901])['conflicts'], [])
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                existing_allocation(temp, [11244001])

    def test_list_configs_reserve_seeds_and_unknown_shapes_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'protocol.json'
            path.write_text(json.dumps(dict(arms=[{'training_seeds':[11244000]}])))
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                existing_allocation(temp, [11244000])
            for arms in (42, ['control', None], {'mission':'unrecognized'}):
                path.write_text(json.dumps(dict(arms=arms)))
                with self.assertRaisesRegex(ValueError, 'Unsupported historical'):
                    existing_allocation(temp, [11244900])


if __name__=='__main__': unittest.main()
