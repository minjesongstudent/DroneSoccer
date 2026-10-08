"""Windows cohort lifetime and registered training leases; stdlib-only bootstrap.

Standalone trainers retain the existing byte-zero lock. A cohort supervisor owns
that same lock and a kill-on-close Job Object. Children receive a one-use ARM
grant through explicitly inherited private pipes only after job assignment.
An on-disk manifest or CLI option alone cannot acquire a worker lease.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import subprocess
import sys
import time


ARMS = ('full_horizon', 'mission', 'local_credit')
MAX_WORKERS = 3
_GRANT = None
_CONSUMED = False


class LeaseStopped(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _windows():
    if os.name != 'nt':
        raise RuntimeError('Controlled cohort lifetime currently requires Windows')


def _kernel():
    _windows()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                   ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.PeekNamedPipe.restype = wintypes.BOOL
    return kernel


def _require(ok, operation):
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error(), operation)


def process_identity(pid):
    kernel = _kernel()
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, int(pid))
    _require(handle, 'OpenProcess')
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        _require(kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)), 'GetProcessTimes')
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


class _BasicLimits(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount', 'WriteOperationCount',
                'OtherOperationCount', 'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', _BasicLimits), ('IoInfo', _IoCounters),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]


class KillJob:
    def __init__(self):
        self.kernel = _kernel()
        self.handle = self.kernel.CreateJobObjectW(None, None)
        _require(self.handle, 'CreateJobObjectW')
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            _require(self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                         ctypes.sizeof(limits)), 'SetInformationJobObject')
        except BaseException:
            self.close()
            raise

    def assign(self, pid):
        handle = self.kernel.OpenProcess(0x0100 | 0x0001 | 0x100000 | 0x1000, False, int(pid))
        _require(handle, 'OpenProcess for job assignment')
        try:
            member = wintypes.BOOL()
            _require(self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(member)), 'IsProcessInJob')
            if member.value:
                return
            _require(self.kernel.AssignProcessToJobObject(self.handle, handle), 'AssignProcessToJobObject')
        finally:
            self.kernel.CloseHandle(handle)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@contextmanager
def project_lock(path):
    """Same msvcrt byte-zero exclusion as existing single_trainer, no imports."""
    _windows()
    import msvcrt
    with Path(path).open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another trainer or cohort owns the project training lock') from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _send(fd, payload):
    data = (canonical(payload) + '\n').encode('utf-8')
    while data:
        data = data[os.write(fd, data):]


def _receive(fd, timeout=15):
    import msvcrt
    kernel = _kernel(); handle = msvcrt.get_osfhandle(fd)
    deadline = time.monotonic() + timeout; data = bytearray()
    while time.monotonic() < deadline:
        available = wintypes.DWORD()
        if not kernel.PeekNamedPipe(handle, None, 0, None, ctypes.byref(available), None):
            raise RuntimeError('Cohort control pipe closed')
        if available.value:
            chunk = os.read(fd, min(available.value, 65536))
            if not chunk:
                raise RuntimeError('Cohort control pipe EOF')
            data.extend(chunk)
            if len(data) > 65536:
                raise ValueError('Cohort control message too large')
            if b'\n' in data:
                line, trailing = data.split(b'\n', 1)
                if trailing:
                    raise ValueError('Unexpected extra cohort control data')
                return json.loads(line)
        time.sleep(.01)
    raise TimeoutError('Cohort control handshake timed out')


def validate_manifest(manifest, path):
    if (manifest.get('format') != 'dronesoccer-authorized-cohort-v1'
            or manifest.get('max_workers') != 3 or manifest.get('arm_order') != list(ARMS)
            or set(manifest.get('arms', {})) != set(ARMS)
            or type(manifest.get('smoke')) is not bool):
        raise ValueError('Invalid fixed three-arm cohort manifest')
    outputs = []
    for arm in ARMS:
        config = manifest['arms'][arm]
        output = Path(config['out']).resolve(); outputs.append(output)
        if not output.is_relative_to(Path(path).resolve().parent / 'arms'):
            raise ValueError('Worker output must be inside the cohort arms directory')
        if (config['target_mode'] != arm or config['memory_fraction'] != .15
                or config['algorithm_seed'] != 11592000
                or config['training_seeds'] != (list(range(11244900, 11244902)) if manifest['smoke']
                                                 else list(range(11244000, 11244240)))
                or config['max_total_seconds'] != (600 if manifest['smoke'] else 3600)):
            raise ValueError('Worker configuration differs from the declared comparison')
    if len(set(outputs)) != 3:
        raise ValueError('Three separate worker output directories required')


class WorkerLease:
    def __init__(self, grant, manifest, manifest_path):
        self._grant = grant
        self._kernel = _kernel()
        self._parent = self._kernel.OpenProcess(0x100000 | 0x1000, False, grant['parent_pid'])
        _require(self._parent, 'OpenProcess live cohort coordinator')
        self._manifest_path = Path(manifest_path).resolve()
        self._manifest_hash = grant['cohort_manifest_sha256']
        self._parent_deadline = time.monotonic() + max(0, grant['seconds_remaining'])
        self.config = dict(manifest['arms'][grant['arm']])
        self.config.update(cohort_id=manifest['cohort_id'],
            cohort_manifest_path=str(self._manifest_path),
            cohort_manifest_sha256=self._manifest_hash,
            allocation_record=manifest['allocation_record'])
        self._parent_deadline = min(self._parent_deadline,
                                    time.monotonic() + self.config['max_total_seconds'])
        self._bound_config = canonical(self.config)
        self.check()

    def check(self):
        if self._kernel.WaitForSingleObject(self._parent, 0) != 258:
            raise LeaseStopped('Cohort coordinator is no longer alive')
        if ((self._manifest_path.parent/'STOP').exists()
                or (Path(self.config['out'])/'STOP').exists()):
            raise LeaseStopped('Cohort or worker STOP requested')
        if time.monotonic() >= self._parent_deadline:
            raise LeaseStopped('Cohort total wall budget exhausted')
        if canonical(self.config) != self._bound_config:
            raise LeaseStopped('Registered worker configuration was modified')

    def wait_first_rollout(self):
        """Stop before value/actor fitting until all three saved rollouts match."""
        self.check()
        output = Path(self.config['out'])
        rollout = output/'rollout_01.npz'; endpoints = output/'episode_endpoints_01.json'
        ready = dict(worker_pid=os.getpid(), arm=self._grant['arm'],
            cohort_manifest_sha256=self._manifest_hash, rollout_sha256=digest(rollout),
            endpoints_sha256=digest(endpoints))
        with (output/'.first_rollout_ready.json').open('x', encoding='utf-8') as stream:
            stream.write(canonical(ready))
        approval = self._manifest_path.parent/'first_rollout_parity.json'
        while True:
            self.check()
            if approval.exists():
                try:
                    approved = json.loads(approval.read_text())
                except json.JSONDecodeError:
                    time.sleep(.02); continue  # coordinator is completing its exclusive write
                if (approved.get('passed') is not True
                        or approved.get('cohort_manifest_sha256') != self._manifest_hash):
                    raise LeaseStopped('Coordinator did not approve the matched first rollout')
                manifest = json.loads(self._manifest_path.read_text())
                if digest(self._manifest_path) != self._manifest_hash:
                    raise LeaseStopped('Cohort manifest changed at first-rollout barrier')
                for arm, config in manifest['arms'].items():
                    if digest(Path(config['out'])/'rollout_01.npz') != approved['rollout_sha256'][arm]:
                        raise LeaseStopped('First rollout changed after coordinator comparison')
                    if digest(Path(config['out'])/'episode_endpoints_01.json') != approved['endpoint_sha256'][arm]:
                        raise LeaseStopped('First rollout endpoint labels changed after coordinator comparison')
                if ready['rollout_sha256'] != approved['rollout_sha256'][self._grant['arm']]:
                    raise LeaseStopped('Own first rollout does not match coordinator approval')
                self.check()
                return dict(path=str(approval), sha256=digest(approval))
            time.sleep(.05)

    def close(self):
        if self._parent:
            self._kernel.CloseHandle(self._parent); self._parent = None


@contextmanager
def worker_lease(cohort_manifest_path, arm, out, smoke):
    """Consume a private live-parent ARM grant, never a manifest-only bypass."""
    global _CONSUMED
    if _GRANT is None or _CONSUMED:
        raise RuntimeError('No unused registered cohort ARM grant; standalone locking is required')
    grant = _GRANT
    path = Path(cohort_manifest_path).resolve()
    if (path != Path(grant['manifest_path']).resolve() or arm != grant['arm']
            or Path(out).resolve() != Path(grant['out']).resolve()
            or smoke is not grant['smoke'] or os.getpid() != grant['worker_pid']
            or digest(path) != grant['cohort_manifest_sha256']
            or process_identity(grant['parent_pid']) != grant['parent_creation_time']):
        raise ValueError('Worker request does not match its live coordinator registration')
    manifest = json.loads(path.read_text())
    validate_manifest(manifest, path)
    if manifest['parent_pid'] != grant['parent_pid'] or manifest['cohort_id'] != grant['cohort_id']:
        raise ValueError('Coordinator identity differs from the frozen manifest')
    for relative, expected in manifest['source_sha256'].items():
        if digest(Path(manifest['project_root'])/relative) != expected:
            raise ValueError(f'Bound worker source changed: {relative}')
    _CONSUMED = True
    lease = WorkerLease(grant, manifest, path)
    try:
        yield lease
    finally:
        lease.close()


@dataclass
class Child:
    arm: str
    process: subprocess.Popen
    worker_pid: int
    creation_time: int
    stdout: object
    stderr: object
    armed_at: float


class CohortProcesses:
    """Own the existing lock and a non-inherited job until every child exits."""
    def __init__(self, lock_path, *, deadline):
        self.lock_path, self.deadline = Path(lock_path), deadline
        self.children = {}
        self.job = None

    def __enter__(self):
        self._lock = project_lock(self.lock_path)
        self._lock.__enter__()
        try:
            self.job = KillJob()
        except BaseException:
            self._lock.__exit__(*sys.exc_info())
            raise
        return self

    def launch(self, manifest_path, arm, *, fixture=False):
        import msvcrt
        if self.job is None or arm in self.children or len(self.children) >= MAX_WORKERS:
            raise ValueError('Duplicate/unregistered/fourth cohort slot refused')
        path = Path(manifest_path).resolve(); manifest = json.loads(path.read_text())
        validate_manifest(manifest, path)
        if arm not in ARMS or manifest['parent_pid'] != os.getpid():
            raise ValueError('This coordinator does not own the requested slot')
        if fixture != (manifest.get('fixture') is True):
            raise ValueError('Lifetime fixtures cannot launch training')
        config = manifest['arms'][arm]
        if Path(config['out']).exists():
            raise FileExistsError(config['out'])
        if time.monotonic() >= self.deadline:
            raise LeaseStopped('Cohort expired before launch')
        parent_read, child_write = os.pipe()
        child_read, parent_write = os.pipe()
        read_handle, write_handle = msvcrt.get_osfhandle(child_read), msvcrt.get_osfhandle(child_write)
        for handle in (read_handle, write_handle):
            os.set_handle_inheritable(handle, True)
        info = subprocess.STARTUPINFO()
        info.lpAttributeList = {'handle_list': [read_handle, write_handle]}
        environment = os.environ.copy()
        environment.update(DRONESOCCER_COHORT_READ=str(read_handle), DRONESOCCER_COHORT_WRITE=str(write_handle),
            OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            PYTHONHASHSEED=str(config['algorithm_seed']), PYTHONUNBUFFERED='1')
        log_dir = path.parent/'logs'; log_dir.mkdir(exist_ok=True)
        stdout = (log_dir/f'{arm}.stdout.log').open('xb')
        stderr = (log_dir/f'{arm}.stderr.log').open('xb')
        process = None
        try:
            process = subprocess.Popen([sys.executable, '-m', 'tools.parallel_training', '--bootstrap'],
                cwd=manifest['project_root'], env=environment, stdin=subprocess.DEVNULL,
                stdout=stdout, stderr=stderr, close_fds=True, startupinfo=info,
                creationflags=subprocess.CREATE_NO_WINDOW)
            # No CUDA import/load occurs in the child before assignment and ARM.
            self.job.assign(process.pid)
            os.close(child_read); child_read = None
            os.close(child_write); child_write = None
            hello = _receive(parent_read)
            if (hello.get('kind') != 'HELLO'
                    or hello.get('parent_pid') not in (os.getpid(), process.pid)):
                raise ValueError('Unexpected bootstrap parent identity')
            worker_pid = hello['pid']
            # Windows venv redirectors may report a distinct interpreter PID.
            # The private pipe is inherited only by our spawned bootstrap, and
            # that actual interpreter is explicitly attached before any ARM.
            if worker_pid != process.pid:
                self.job.assign(worker_pid)
            worker_creation = process_identity(worker_pid)
            grant = dict(kind='ARM', nonce=secrets.token_hex(32), arm=arm, out=config['out'],
                worker_pid=worker_pid, manifest_path=str(path), smoke=manifest['smoke'],
                cohort_manifest_sha256=digest(path), cohort_id=manifest['cohort_id'],
                parent_pid=os.getpid(), parent_creation_time=process_identity(os.getpid()),
                seconds_remaining=max(0, self.deadline-time.monotonic()), fixture=fixture)
            _send(parent_write, grant)
            ready = _receive(parent_read)
            if ready != {'kind': 'ARMED', 'pid': worker_pid, 'nonce': grant['nonce']}:
                raise ValueError('Worker ARM acknowledgement mismatch')
            child = Child(arm, process, worker_pid, worker_creation, stdout, stderr, time.time())
            self.children[arm] = child
            return child
        except BaseException:
            if process is not None:
                process.terminate(); process.wait(timeout=15)
            stdout.close(); stderr.close()
            raise
        finally:
            for fd in (parent_read, parent_write, child_read, child_write):
                if fd is not None:
                    os.close(fd)

    def snapshot(self):
        return {arm: dict(launcher_pid=child.process.pid, worker_pid=child.worker_pid,
                         process_creation_time=child.creation_time, exit_code=child.process.poll(),
                         armed_at_unix=child.armed_at)
                for arm, child in self.children.items()}

    def __exit__(self, exc_type, exc, tb):
        # Closing the non-inherited job kills all surviving registered children
        # and descendants even if a coordinator exits abnormally.
        if self.job is not None:
            self.job.close()
        try:
            for child in self.children.values():
                child.process.wait(timeout=15)
                child.stdout.close(); child.stderr.close()
        finally:
            self._lock.__exit__(exc_type, exc, tb)


def _fixture(grant):
    """Only stdlib lifetime tests; no model imports or movement interfaces."""
    with worker_lease(grant['manifest_path'], grant['arm'], grant['out'], grant['smoke']) as lease:
        output = Path(grant['out']); output.mkdir(parents=True, exist_ok=False)
        (output/'started.json').write_text(canonical(dict(pid=os.getpid(), parent_pid=os.getppid())))
        try:
            while True:
                lease.check(); time.sleep(.02)
        except LeaseStopped as exc:
            (output/'stopped.json').write_text(canonical(dict(reason=str(exc))))


def _bootstrap():
    global _GRANT
    import msvcrt
    # These explicitly inherited private handles are not an environment-only
    # authorization token; the live parent must complete both pipe messages.
    read_handle = int(os.environ.pop('DRONESOCCER_COHORT_READ'))
    write_handle = int(os.environ.pop('DRONESOCCER_COHORT_WRITE'))
    reader = msvcrt.open_osfhandle(read_handle, os.O_RDONLY)
    writer = msvcrt.open_osfhandle(write_handle, os.O_WRONLY)
    try:
        _send(writer, dict(kind='HELLO', pid=os.getpid(), parent_pid=os.getppid()))
        grant = _receive(reader)
        if grant.get('kind') != 'ARM' or grant.get('worker_pid') != os.getpid():
            raise ValueError('Invalid coordinator ARM grant')
        if process_identity(grant['parent_pid']) != grant['parent_creation_time']:
            raise ValueError('Coordinator identity is stale')
        _GRANT = grant
        _send(writer, dict(kind='ARMED', pid=os.getpid(), nonce=grant['nonce']))
    finally:
        os.close(reader); os.close(writer)
    if grant['fixture']:
        _fixture(grant)
        return
    path = Path(grant['manifest_path'])
    manifest = json.loads(path.read_text())
    sys.argv = ['tools.run_estimated_motion_trial', '--out', grant['out'],
                '--initialization-report', manifest['initialization']['path'],
                '--cohort-manifest', str(path), '--arm', grant['arm']]
    if grant['smoke']:
        sys.argv.append('--smoke')
    runpy.run_module('tools.run_estimated_motion_trial', run_name='__main__')


if __name__ == '__main__':
    if sys.argv[1:] != ['--bootstrap']:
        raise SystemExit('This internal bootstrap requires a live registered supervisor')
    # The subsequently imported worker_lease must see this same private grant.
    sys.modules['tools.parallel_training'] = sys.modules[__name__]
    _bootstrap()
