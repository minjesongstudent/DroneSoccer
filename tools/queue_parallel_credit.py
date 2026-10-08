"""Bounded admission queue for the one reviewed three-arm comparison.

Wait for GPU headroom, run the software smoke, and launch MAIN only if all
three smoke workers complete with exact first-rollout parity. No retry,
resource-limit change, model promotion, or hardware control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.parallel_training import ARMS, KillJob, digest, process_identity, project_lock
from tools.run_parallel_estimated_motion import ROOT, gpu_snapshot, read_json, write_new, write_status

RUNS = Path('D:/DroneSoccerData/runs')
SMOKE = RUNS/'parallel-credit-smoke-20261008-v2'
MAIN = RUNS/'parallel-credit-20261008'
INITIALIZATION = RUNS/'estimated-motion-initialization-20261008/decision.json'


def smoke_ready(path):
    decision = read_json(path/'decision.json')
    manifest = read_json(path/'cohort_manifest.json')
    if (decision['status'] != 'complete' or decision['errors'] or not decision['smoke']
            or decision['selected'] is not None or not decision['fairness']['matched']
            or digest(path/'cohort_manifest.json') != decision['cohort_manifest_sha256']
            or manifest['smoke'] is not True or set(decision['workers']) != set(ARMS)):
        raise ValueError('Smoke coordinator did not complete the reviewed comparison')
    for arm in ARMS:
        worker = decision['workers'][arm]
        folder = path/'arms'/arm
        result = read_json(folder/'result.json')
        closure = read_json(folder/'decision.json')
        if (worker['exit_code'] != 0 or worker['decision'] != 'software_only'
                or worker['passed'] or digest(folder/'decision.json') != worker['decision_sha256']
                or result['status'] != 'complete' or not result['frozen_unchanged']
                or not result['training_budget_complete'] or not result['final_pipeline_complete']
                or result['completed_seeds'] != [11244900,11244901]
                or closure['reasons'] != ['Software-only smoke can never pass']):
            raise ValueError('Smoke worker incomplete or failed: '+arm)
        for artifact, expected in closure['evidence'].items():
            if digest(artifact) != expected:
                raise ValueError('Smoke evidence changed: '+artifact)
    return digest(path/'decision.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    started = time.monotonic()
    status = {'phase':'preparing', 'training_started':False, 'acceptance':False, 'physical_flight':False}
    with project_lock(ROOT.parent/'.parallel-credit-queue.lock'):
        out.mkdir(parents=True, exist_ok=False)
        files = list((ROOT/'src').glob('*.py')) + [ROOT/p for p in (
            'tools/queue_parallel_credit.py', 'tools/parallel_training.py',
            'tools/run_parallel_estimated_motion.py', 'tools/run_estimated_motion_trial.py',
            'tools/check_estimated_motion_initialization.py', 'tools/run_clearance_experiment.py',
            'tools/run_onpolicy_trial.py', 'tools/validate_flight_policies.py',
            'tests/test_parallel_credit_queue.py', 'tests/test_parallel_training.py', 'tests/test_estimated_motion_trial.py',
            'experiments/faisal/docs/PARALLEL_CREDIT_COMPARISON_20261008.md')]
        sources = {}
        for path in files:
            relative = path.relative_to(ROOT)
            target = out/'source'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            sources[str(path)] = digest(target)
        write_new(out/'launch_protocol.json', dict(smoke=str(SMOKE), main=str(MAIN),
            sources=sources, wait_seconds_per_admission=900, retries=0,
            note='First smoke failed before any allocation or worker; preserved separately. Corrected scanner passed85 tests.',
            initialization=str(INITIALIZATION), acceptance=False, physical_flight=False))

        def check():
            if (out/'STOP').exists(): raise RuntimeError('Queue STOP requested')

        def unchanged():
            for path, expected in sources.items():
                if digest(path) != expected: raise RuntimeError('Reviewed source changed while queued: '+path)

        def admission(stage):
            deadline = time.monotonic()+900
            consecutive = 0
            while time.monotonic() < deadline:
                check()
                gpu = gpu_snapshot()
                minimum = int(3*.15*gpu['total_mib'])+2048
                status.update(phase='waiting_for_gpu_'+stage, gpu=gpu, required_free_mib=minimum,
                    admission_seconds_remaining=round(deadline-time.monotonic(),1))
                write_status(out/'status.json',status)
                consecutive = consecutive+1 if gpu['free_mib'] >= minimum else 0
                if consecutive >= 2:
                    unchanged()
                    return
                time.sleep(10)
            raise RuntimeError('GPU admission timed out after900seconds; no automatic retry')

        def run_stage(path, smoke):
            if path.exists(): raise FileExistsError('Fresh stage output required: '+str(path))
            args = [sys.executable,'-m','tools.run_parallel_estimated_motion','--out',str(path),
                    '--initialization-report',str(INITIALIZATION)] + (['--smoke'] if smoke else [])
            job = KillJob()
            try:
                with (out/('smoke.stdout.log' if smoke else 'main.stdout.log')).open('xb') as stdout, \
                     (out/('smoke.stderr.log' if smoke else 'main.stderr.log')).open('xb') as stderr:
                    child = subprocess.Popen(args,cwd=ROOT,stdout=stdout,stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    job.assign(child.pid)
                    actual_attached = False
                    deadline = time.monotonic()+(930 if smoke else 3930)
                    status.update(phase='running_smoke' if smoke else 'running_main',
                        coordinator_launcher_pid=child.pid, active_run=str(path), training_started=True)
                    write_status(out/'status.json',status)
                    while child.poll() is None:
                        check()
                        manifest_path = path/'cohort_manifest.json'
                        if not actual_attached and manifest_path.exists():
                            try: manifest = read_json(manifest_path)
                            except json.JSONDecodeError: manifest = None
                            if manifest is not None:
                                pid = manifest['parent_pid']
                                if process_identity(pid) != manifest['parent_creation_time']:
                                    raise RuntimeError('Coordinator process identity changed')
                                job.assign(pid)
                                actual_attached = True
                        if time.monotonic() >= deadline:
                            raise RuntimeError('Coordinator exceeded bounded stage deadline')
                        time.sleep(1)
                    if child.returncode != 0: raise RuntimeError('Coordinator failed with exit '+str(child.returncode))
            finally:
                job.close()

        try:
            admission('smoke')
            run_stage(SMOKE, True)
            status['smoke_decision_sha256'] = smoke_ready(SMOKE)
            admission('main')
            run_stage(MAIN, False)
            status.update(phase='complete', main_decision_sha256=digest(MAIN/'decision.json'))
        except Exception as exc:
            status.update(phase='stopped', error=repr(exc))
        finally:
            status['elapsed_seconds'] = time.monotonic()-started
            write_status(out/'status.json',status)
            write_new(out/'result.json', status)
    return 0 if status['phase']=='complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
