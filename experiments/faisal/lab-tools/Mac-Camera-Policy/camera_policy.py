"""Live Tello camera -> ring detector -> recurrent policy preview. Motors OFF."""
import argparse,json,socket,threading,time,hashlib
from pathlib import Path
from datetime import datetime
import av
import cv2
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from bridge import ObservationBridge,detect

ROOT=Path(__file__).resolve().parent
ALLOWED={'command','streamon','streamoff'}

def send(sock,command):
    if command not in ALLOWED:raise ValueError('Physical movement is disabled in this preview')
    sock.sendto(command.encode(),('192.168.10.1',8889))

class Feed:
    def __init__(self,source,live):
        self.source=source;self.live=live;self.latest=None;self.error=None;self.done=False
        self.stop=threading.Event();self.lock=threading.Lock()
        self.thread=threading.Thread(target=self.read,daemon=True)
    def read(self):
        try:
            with av.open(self.source,options={'timeout':'5000000'} if self.live else {},timeout=5) as container:
                start=time.monotonic();first=None
                for i,frame in enumerate(container.decode(video=0)):
                    if self.stop.is_set():break
                    if not self.live:
                        stamp=float(frame.time) if frame.time is not None else i/30
                        if first is None:first=stamp
                        if self.stop.wait(max(0,stamp-first-(time.monotonic()-start))):break
                    with self.lock:self.latest=(i,time.monotonic(),frame.to_ndarray(format='bgr24'))
        except Exception as exc:self.error=repr(exc)
        finally:self.done=True
    def get(self):
        with self.lock:return self.latest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',help='Optional saved video instead of live drone camera')
    args=parser.parse_args()
    torch.set_num_threads(1)
    config=json.loads((ROOT/'config.json').read_text())
    model_path=ROOT/'policy.zip'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==json.loads((ROOT/'manifest.json').read_text())['model_sha256']
    predictor=RecurrentPredictor(RecurrentPPO.load(model_path,device='cpu'))
    bridge=ObservationBridge()
    out=ROOT/'sessions'/datetime.now().strftime('%Y%m%d-%H%M%S-%f');out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'metadata.json').write_text(json.dumps({'mode':'motors-off shadow preview','actual_flight_control':False,
        'source':'live Tello' if args.video is None else str(args.video),'issued_actions':'zero: policy suggestions never sent',
        'camera_calibration_measured':config['calibration_measured'],'marker_configured':config['marker_hsv_low'] is not None,
        'warning':'Unvalidated detector and nominal camera calibration. Hand motion is not captured by command odometry.'},indent=2))
    control=None;feed=None;last_id=None;next_tick=0;previous_tick=None;pending_reset=True
    try:
        if args.video is None:
            control=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);control.bind(('',8889));control.settimeout(8)
            send(control,'command')
            response,address=control.recvfrom(1024)
            if address[0]!='192.168.10.1' or response.strip()!=b'ok':raise RuntimeError('Drone did not acknowledge SDK mode')
        feed=Feed(args.video or 'udp://0.0.0.0:11111?fifo_size=1000000&overrun_nonfatal=1',args.video is None)
        feed.thread.start()
        if control:
            send(control,'streamon')
            response,address=control.recvfrom(1024)
            if address[0]!='192.168.10.1' or response.strip()!=b'ok':raise RuntimeError('Drone did not acknowledge streamon')
        print('CAMERA POLICY PREVIEW: motors OFF. Q quits; R resets history. No movement commands sent.',flush=True)
        start=time.monotonic()
        with (out/'predictions.jsonl').open('w') as log:
            while True:
                item=feed.get();now=time.monotonic()
                if item is None:
                    if feed.done or now-start>12:raise RuntimeError(feed.error or 'No video received; check Wi-Fi and Local Network permission')
                    time.sleep(.01);continue
                frame_id,received,frame=item
                if now-received>.5:
                    pending_reset=True
                    display=frame.copy();cv2.putText(display,'STALE VIDEO - predictions paused',(15,35),0,.7,(0,0,255),2)
                    cv2.imshow('Policy preview - NO FLIGHT',display)
                    if feed.done:break
                elif now>=next_tick and frame_id!=last_id:
                    gap=None if previous_tick is None else now-previous_tick
                    if pending_reset or (gap is not None and gap>.25):
                        bridge=ObservationBridge();predictor.reset_episode();pending_reset=False
                    detection,ellipse,mask=detect(frame,config)
                    observation=bridge.update(detection,np.zeros(4,np.float32))
                    with torch.inference_mode():action,_=predictor.predict(observation)
                    display=frame.copy()
                    if ellipse is not None:cv2.ellipse(display,ellipse,(0,255,0),2)
                    cv2.putText(display,'SHADOW ONLY / UNVALIDATED INPUTS / Q quit R reset',(10,25),0,.55,(0,0,255),2)
                    cv2.putText(display,'Right Forward Up Yaw: '+' '.join(f'{a:+.2f}' for a in action),(10,55),0,.55,(255,255,255),2)
                    cv2.putText(display,f'Ring visible: {detection[6]:.0f} Front marker: {detection[7]:.0f}',(10,85),0,.55,(255,255,255),2)
                    cv2.imshow('Policy preview - NO FLIGHT',display);cv2.imshow('Blue ring mask - inspect false detections',mask)
                    log.write(json.dumps({'elapsed_s':now-start,'frame_id':frame_id,'frame_age_s':now-received,
                        'tick_gap_s':gap,'observation':observation.tolist(),'suggested_action':action.tolist(),'issued_action':[0,0,0,0]})+'\n');log.flush()
                    previous_tick=now;next_tick=now+.1;last_id=frame_id
                key=cv2.waitKey(1)&255
                if key in (ord('q'),27):break
                if key==ord('r'):pending_reset=True
                if feed.done and (item is None or last_id==frame_id):break
    finally:
        if control:
            try:send(control,'streamoff')
            finally:control.close()
        if feed:feed.stop.set();feed.thread.join(timeout=6)
        cv2.destroyAllWindows()
        print('Saved diagnostic logs:',out)

if __name__=='__main__':main()
