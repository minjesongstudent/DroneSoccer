"""Browser camera + group detector + learned-policy suggestions. Motors off only."""
import argparse
import hashlib
import json
import queue
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from stable_baselines3 import SAC

import recorder
from src.camera_history import CameraHistory
from src.camera_calibration import default_calibration, load_calibration
from src.policy_runtime import LearnedPolicy, SPECS
from pc_runtime import PCRuntime
from src.detect import HoopDetector, draw, CONFIRM_FRAMES

ROOT = Path(__file__).resolve().parent
PAGE = '''<!doctype html><meta charset="utf-8"><title>Lab camera</title>
<style>body{margin:24px;background:#101827;color:#eee;font:17px system-ui}img{max-width:100%;max-height:70vh}p{max-width:1000px}.card{background:#1c2940;padding:12px;border-radius:8px}#error{color:#ffb1a9}</style>
<h1>Live lab camera</h1><p class="card">Motors-off preview. Policy suggestions are displayed only. No movement commands are sent.</p>
<p id="status">Connecting…</p><p id="calibration"></p><p id="error"></p><img id="camera" alt="Waiting for a decoded camera frame">
<p id="policy"></p><p>Green: accepted ring. Pink: detected front marker. Red: rejected candidate.
The model is a simulation checkpoint, not a validated flight controller.</p>
<script>async function poll(){try{let s=await(await fetch('/state')).json();
document.getElementById('status').textContent=`${s.mode||'Starting'} · camera age ${s.age_s==null?'—':s.age_s.toFixed(2)+'s'} · telemetry ${s.state_packets||0} · video ${(s.video_bytes||0).toLocaleString()} bytes · decoded ${s.decoded||0} · dropped preview packets ${s.dropped||0}`;
document.getElementById('error').textContent=s.error||(s.stale?'Video stale: policy suggestions paused.':'');
document.getElementById('calibration').textContent=`Camera: ${s.camera_calibration?.label||'Unknown'} · source: ${s.camera_calibration?.selected_by||'unknown'}`;
document.getElementById('policy').textContent=s.stale?'':`${s.model||''} · Ring ${s.ring?'visible':'not detected'} · marker ${s.marker?'detected':'not detected'} · suggested right / forward / up / yaw: ${(s.suggestion||[]).map(v=>v.toFixed(2)).join(' / ')} · actual movement commands: none`;
if(s.decoded)document.getElementById('camera').src='/frame.jpg?v='+s.frame_id;
}catch(e){document.getElementById('error').textContent='Preview disconnected'}setTimeout(poll,200)}poll();</script>'''


def select_calibration(manifest, explicit_path=None, root=None):
    """Select once; never pick up an unrecorded calibration from the working folder."""
    root = ROOT if root is None else Path(root)
    if explicit_path is not None:
        calibration = load_calibration(explicit_path)
        selected_by = 'explicit --calibration'
    else:
        declared = manifest.get('camera_calibration', {})
        bundled = declared.get('bundled_file')
        if bundled is not None:
            if bundled != 'calib.npz':
                raise ValueError('Unexpected bundled calibration path')
            calibration = load_calibration(root / bundled)
            if calibration.sha256 != declared.get('sha256'):
                raise ValueError('Bundled calibration hash does not match the manifest')
            selected_by = 'package manifest'
        else:
            if declared.get('status', 'defaults') != 'defaults':
                raise ValueError('Manifest declares calibration without a bundled file')
            calibration = default_calibration()
            selected_by = 'explicit team defaults'
    return calibration, dict(calibration.status(), selected_by=selected_by)


class Preview:
    def __init__(self, calibration_path=None):
        torch.set_num_threads(1)
        manifest = json.loads((ROOT/'preview-manifest.json').read_text())
        self.runtime = PCRuntime()
        self.spec = self.runtime.spec
        if manifest['model'] != self.spec.name or manifest['model_sha256'] != self.spec.sha256 or manifest['policy_inputs'] != 89:
            raise ValueError('Preview manifest disagrees with the selected PC checkpoint')
        self.calibration, self.calibration_status = select_calibration(manifest, calibration_path)
        self.model_hash = self.spec.sha256
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.chunks = queue.Queue(maxsize=512)
        self.sequence = 0
        self.latest = None
        self.jpg = None
        self.state = dict(mode='Connecting to camera', decoded=0, dropped=0, model=self.spec.name, policy_identity=self.runtime.identity,
                          camera_calibration=self.calibration_status)
        self.threads = []
        self.server = None

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            result['age_s'] = None if self.latest is None else time.monotonic()-self.latest[1]
        result['stale'] = result['age_s'] is None or result['age_s'] > .5 or self.stop.is_set()
        if time.monotonic()-result.get('suggestion_received', 0) > .5:
            result['stale'] = True
        if result['stale']:
            result['suggestion'] = []
        return result

    def status(self, fields):
        with self.lock:
            self.state.update(fields)

    def open_browser(self):
        preview = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                if self.path == '/':
                    data, mime = PAGE.encode(), 'text/html; charset=utf-8'
                elif self.path == '/state':
                    data, mime = json.dumps(preview.snapshot()).encode(), 'application/json'
                elif self.path.startswith('/frame.jpg'):
                    with preview.lock:
                        data = preview.jpg
                    if data is None:
                        self.send_error(404); return
                    mime = 'image/jpeg'
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type',mime)
                self.send_header('Cache-Control','no-store')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError,ConnectionResetError):
                    pass
        self.server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        url = f'http://127.0.0.1:{self.server.server_port}/'
        print('Open camera window on this Mac:',url,flush=True)
        webbrowser.open(url)

    def start(self, folder):
        self.folder = Path(folder)
        self.status({'mode':'Recording camera; motors off'})
        (self.folder/'preview_metadata.json').write_text(json.dumps(dict(
            model_sha256=self.model_hash, model=self.spec.name, policy_inputs=self.spec.inputs,
            policy_identity=self.runtime.identity,
            observation_lags=list(self.spec.lags),
            issued_actions='zero; no movement commands sent', target_rate_hz=10,
            camera_calibration=self.calibration_status,
            frame_timing='Host receipt age, not measured exposure age',
            clock_domain='Python time.monotonic on this recording host',
            log_schema_version=3,
            hand_carried=True),indent=2))
        for fn in (self.decode, self.analyse):
            t = threading.Thread(target=fn,daemon=True);t.start();self.threads.append(t)

    def feed(self, data, received):
        self.sequence += 1
        try:
            self.chunks.put_nowait((self.sequence,received,data))
        except queue.Full:
            with self.lock:
                self.state['dropped'] += 1

    def decoded(self, frame, received):
        with self.lock:
            number = self.state['decoded']+1
            self.state['decoded'] = number
            self.latest = (number,received,frame)

    def decode(self):
        codec = av.CodecContext.create('h264','r')
        previous = 0
        while not self.stop.is_set():
            try:
                number,received,data = self.chunks.get(timeout=.1)
            except queue.Empty:
                continue
            if number != previous+1:
                codec = av.CodecContext.create('h264','r')
            previous = number
            try:
                for packet in codec.parse(data):
                    for frame in codec.decode(packet):
                        self.decoded(frame.to_ndarray(format='bgr24'),received)
            except av.FFmpegError:
                with self.lock:
                    self.state['decode_errors'] = self.state.get('decode_errors',0)+1

    def analyse(self):
        self.runtime.reset()
        last_id, last_tick = -1, None
        with (self.folder/'policy_preview.jsonl').open('w') as log:
            try:
                detector = HoopDetector(*self.calibration.size, confirm=CONFIRM_FRAMES,
                                        calibration=self.calibration)
                while not self.stop.is_set():
                    tick = time.monotonic()
                    with self.lock:
                        latest = self.latest
                    if latest is None or latest[0] == last_id or tick-latest[1] > .5:
                        self.stop.wait(.02);continue
                    frame_id,received,frame = latest
                    gap = None if last_tick is None else tick-last_tick
                    reset = gap is not None and gap > .25
                    if reset:
                        self.runtime.reset()
                        detector = HoopDetector(*self.calibration.size, confirm=CONFIRM_FRAMES,
                                                calibration=self.calibration)
                    analysis_started = time.monotonic()
                    detection = detector(frame)
                    # No inferred motion from hand carrying and no unsent action history.
                    obs, action = self.runtime.predict(detection.obs,np.zeros(4,np.float32))
                    image = draw(frame,detection)
                    ok,jpg = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
                    age = time.monotonic()-received
                    suggestion = action.tolist() if age <= .5 else []
                    with self.lock:
                        if ok:self.jpg = jpg.tobytes()
                        self.state.update(frame_id=frame_id,ring=bool(detection.obs[6]),
                            marker=bool(detection.obs[7]),suggestion=suggestion,
                            suggestion_received=received)
                    log.write(json.dumps(dict(frame_id=frame_id,receipt_monotonic=received,
                        analysis_started_monotonic=analysis_started, analysis_completed_monotonic=time.monotonic(),
                        camera_features=detection.obs.tolist(),
                        raw_policy_action=action.tolist(),
                        frame_age_s=age,tick_gap_s=gap,history_reset=reset,
                        observation=obs.tolist(),suggested_action=suggestion,
                        issued_action=[0,0,0,0]))+'\n');log.flush()
                    last_id,last_tick = frame_id,tick
                    self.stop.wait(max(0,.1-(time.monotonic()-tick)))
            except Exception as exc:
                self.status({'error':'Preview analysis failed: '+str(exc),
                             'suggestion':[], 'suggestion_received':0})

    def close(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=10)
        if any(t.is_alive() for t in self.threads):
            raise RuntimeError('Preview did not finish before export; check diagnostic log')
        self.status({'mode':'Recording finished; saved in sessions'})
        if hasattr(self, 'folder') and not getattr(self, '_outcome_saved', False):
            with (self.folder/'preview_outcome.json').open('x') as outcome:
                json.dump(self.snapshot(), outcome, indent=2)
            self._outcome_saved = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--video',type=Path,help='Replay a saved video without contacting the drone')
    parser.add_argument('--calibration',type=Path,
                        help='Measured camera calibration; overrides the package selection')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('Use 1 to 300 seconds')
    preview = Preview(args.calibration)
    preview.open_browser()
    try:
        if args.video:
            folder = ROOT/'sessions'/('replay-'+str(time.time_ns()));folder.mkdir(parents=True)
            preview.start(folder)
            preview.status({'mode':'Saved video replay; motors-off policy suggestions'})
            cap = cv2.VideoCapture(str(args.video))
            try:
                start = time.monotonic()
                i = 0
                while time.monotonic()-start < args.seconds:
                    ok,frame = cap.read()
                    if not ok:break
                    time.sleep(max(0,start+i/30-time.monotonic()))
                    preview.decoded(frame,time.monotonic())
                    i += 1
            finally:
                cap.release()
            preview.close()
        else:
            recorder.record(args.seconds,observer=preview)
        print('Finished. The browser remains open. Ctrl+C closes the server.',flush=True)
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        pass
    finally:
        preview.close()
        if preview.server:
            preview.server.shutdown();preview.server.server_close()

if __name__ == '__main__':
    main()
