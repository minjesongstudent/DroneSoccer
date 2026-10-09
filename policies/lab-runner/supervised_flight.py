"""EXPERIMENTAL physical Tello policy test, started only by the local operator.

No connection occurs on import. Run explicitly with --physical-flight.
The original motors-off recorder remains a separate launcher.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import queue
import secrets
import select
import socket
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import cv2
import numpy as np
from live_preview import Preview, ROOT
from recorder import parse_state, control_reply, utc, export
from src.detect import HoopDetector, draw, CONFIRM_FRAMES
from flight_control import FlightControl
from startup_search import LIMITS
from compare_session import analyse_session

PAGE = '''<!doctype html><meta charset="utf-8"><title>Supervised policy flight</title>
<style>body{background:#101827;color:white;font:18px system-ui;margin:24px}button{font:inherit;padding:20px;margin:8px;touch-action:none}#start{background:#93e0c0}#land{background:#ffb458}#cut{background:#ec6666}img{max-height:50vh;max-width:100%}pre{white-space:pre-wrap}</style>
<h1>PC-trained policy flight + recording</h1><p>Vertical search v2</p><p id="identity">Verifying policy identity...</p><p id="phase"></p>
<p>This launcher sends REAL takeoff and flight commands. Experimental policy, not flight-qualified.
Keep the cage clear and have an operator ready. Team RC mapping: 50 / 50 / 100 / 70.
Click once to take off, find the ring, then run the policy. Startup search can rise slowly
to a measured 120 cm and turn in place for two bounded sweeps. It never searches forward
or sideways. If still unseen after turning, it lowers slowly toward 80 cm while looking.
The 150 cm ceiling remains. Search ends within 20 seconds or requests landing.
After stable ring detection, learned control keeps running until you stop it or a safety check requests landing.
Use LAND NOW or Escape to stop early. Keep this page open and focused. No automatic restart.</p>
<p><label><input type="checkbox" id="ready"> Cage clear, guards fitted; I am supervising this physical test.</label></p>
<button id="start" disabled>Take off + find ring + run policy</button>
<button id="land">LAND NOW / Escape</button>
<details><summary>Emergency motor cut - drone will drop</summary><button id="cut">CUT MOTORS NOW</button></details>
<p id="connection">Connecting to local recorder</p><p id="countdown"></p>
<pre id="status">Waiting for current state</pre><p id="error"></p><img id="camera" alt="Live recorded camera">
<script>
const token=location.hash.slice(1);
const el=id=>document.querySelector('#'+id);
let running=false,startPending=false,closedForFlight=false,timer=null,busy=false;
function endHeartbeat(){clearInterval(timer);timer=null;}
async function post(action){
  const r=await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json','X-Flight-Token':token},body:JSON.stringify({action})});
  const s=await r.json();
  if(!r.ok){const e=Error(s.error);e.controlState=s.control_state;throw e;}
  return s;
}
async function stopFlight(force=false){
  if(!running&&!force)return;
  running=false;closedForFlight=true;endHeartbeat();el('start').disabled=true;
  try{await post('land')}
  catch(e){el('error').textContent='Landing request failed; delivery unconfirmed: '+e.message}
}
el('start').onclick=async()=>{
  if(running||startPending||closedForFlight||el('start').disabled)return;
  if(!el('ready').checked){el('error').textContent='Check the supervision box, then click Take off + run policy.';return;}
  running=true;startPending=true;el('start').disabled=true;el('error').textContent='';
  try{
    await post('heartbeat');if(!running)return;
    await post('start');
    // Stop during an in-flight request must still win if the start reply arrives late.
    if(!running){await post('land');return;}
    timer=setInterval(async()=>{
      if(!running||busy)return;
      busy=true;
      try{await post('heartbeat')}catch(e){await stopFlight()}finally{busy=false}
    },100);
  }catch(e){
    el('error').textContent=e.message;
    if(e.controlState==='ground'&&!closedForFlight){running=false;endHeartbeat();}
    else{await stopFlight();}
  }finally{startPending=false;}
};
el('ready').onchange=()=>{if(!el('ready').checked)stopFlight()};
el('land').onclick=()=>stopFlight(true);
el('cut').onclick=()=>{
  running=false;closedForFlight=true;endHeartbeat();el('start').disabled=true;
  post('emergency').catch(e=>el('error').textContent=e.message);
};
addEventListener('blur',()=>{if(running)stopFlight()});
addEventListener('pagehide',()=>{running=false;closedForFlight=true;endHeartbeat()});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&running)stopFlight()});
addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();stopFlight(true)}});
async function poll(){
  try{
    const r=await fetch('/state');if(!r.ok)throw Error('HTTP '+r.status);
    const s=await r.json();if(!s.control)throw Error('Missing control state');
    el('connection').textContent='Connected to local recorder';
    el('countdown').textContent=s.preflight_remaining_s==null?'Recording continues throughout policy flight and landing.':'Start within '+Math.ceil(s.preflight_remaining_s)+' seconds (preflight window).';
    el('status').textContent=JSON.stringify(s.control,null,2);
    el('identity').textContent='PC checkpoint: '+s.control.policy_identity.id+' | '+s.control.policy_identity.sha256+' | NOT flight-qualified';
    el('phase').textContent=s.control.state==='acquiring'?'STARTUP SEARCH (scripted): '+s.control.startup_search_phase:s.control.state==='active'?'LEARNED POLICY CONTROL':s.control.state.toUpperCase()+(s.control.reason?' — '+s.control.reason:'');
    if(['landing','stopped'].includes(s.control.state)){running=false;closedForFlight=true;endHeartbeat();}
    el('start').disabled=running||startPending||closedForFlight||s.control.state!=='ground'||!!s.control.readiness;
    if(s.decoded)el('camera').src='/frame.jpg?v='+s.frame_id;
    if(s.error)el('error').textContent=s.error;
  }catch(e){
    el('connection').textContent='Local recorder unreachable; displayed state and image may be stale. Check Terminal. The preflight window may have expired before takeoff. Landing/delivery is not confirmed.';
    el('start').disabled=true;el('countdown').textContent='';
    if(running)await stopFlight();
  }
  setTimeout(poll,200);
}
poll();
</script>'''


class CaptureLink:
    def __init__(self, folder):
        self.folder=folder;self.stop=threading.Event();self.lock=threading.RLock()
        self.files={name:(folder/name).open('w') for name in
                    ('telemetry.jsonl','video_packets.jsonl','commands.jsonl','policy_events.jsonl','policy_preview.jsonl','startup_search.jsonl')}
        self.movie=(folder/'camera.h264').open('wb')
        self.sockets=[];self.replies=queue.Queue();self.thread=None
        self.preview=None;self.error=None;self.start=time.monotonic()
        self.state_packets=self.video_packets=0
        try:
            for port in (8889,8890,11111):
                sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                self.sockets.append(sock);sock.bind(('',port));sock.setblocking(False)
        except Exception:
            self.close();raise

    def write(self, name, row):
        with self.lock:
            self.files[name].write(json.dumps(row,allow_nan=False)+'\n');self.files[name].flush()

    def policy_log(self, row):
        self.write('policy_events.jsonl',row)
        if row['event']=='send_result':self.write('policy_preview.jsonl',row)
        if row['event'].startswith('startup_'):self.write('startup_search.jsonl',row)

    def send(self, command):
        if command not in ('command','streamon','streamoff','takeoff','land','emergency'):
            parts=command.split()
            if len(parts)!=5 or parts[0]!='rc' or any(abs(int(v))>100 for v in parts[1:]):
                raise ValueError('Unsupported flight command')
        stamp=time.monotonic()
        sent=False;failure=None
        try:
            self.sockets[0].sendto(command.encode('ascii'),('192.168.10.1',8889));sent=True
        except OSError as exc: failure=str(exc)
        self.write('commands.jsonl',dict(sent_monotonic=stamp,utc=utc(),command=command,
                   sent=sent,error=failure,delivery_confirmed=False))
        if failure:raise OSError(failure)

    def request(self, command):
        while not self.replies.empty():self.replies.get_nowait()
        sent=time.monotonic();self.send(command);deadline=sent+8
        while time.monotonic()<deadline:
            try: received,reply=self.replies.get(timeout=.1)
            except queue.Empty:continue
            if received<sent:continue
            if reply.lower()=='ok':return
            raise RuntimeError(command+': '+reply)
        raise TimeoutError(command+': no SDK acknowledgement')

    def run(self):
        try:
            while not self.stop.is_set():
                for sock in select.select(self.sockets,[],[],.05)[0]:
                    data,address=sock.recvfrom(65535);received=time.monotonic()
                    if address[0]!='192.168.10.1':continue
                    stamp=dict(utc=utc(),receipt_monotonic=received,elapsed_s=received-self.start)
                    if sock is self.sockets[1]:
                        raw=data.decode('utf-8',errors='replace');fields=parse_state(raw)
                        self.write('telemetry.jsonl',dict(stamp,raw=raw,fields=fields));self.state_packets+=1
                        self.preview.control.receive_telemetry(fields,received)
                    elif sock is self.sockets[2]:
                        self.write('video_packets.jsonl',dict(stamp,offset=self.movie.tell(),size=len(data)))
                        self.movie.write(data);self.movie.flush();self.video_packets+=1
                        self.preview.feed(data,received)
                    else:
                        reply=control_reply(data,address)
                        self.write('commands.jsonl',dict(stamp,response=reply,raw_hex=data.hex(),source=list(address)))
                        if reply:
                            self.replies.put((received,reply));self.preview.control.acknowledge(reply,received)
        except Exception as exc:
            self.error=str(exc)
            if self.preview:self.preview.control.stop('capture failure: '+str(exc))

    def close(self):
        self.stop.set()
        if self.thread:self.thread.join(2)
        for sock in self.sockets:sock.close()
        for f in self.files.values():f.close()
        self.movie.close()


class FlightPreview(Preview):
    def __init__(self, link):
        super().__init__()
        self.link=link
        self.control=FlightControl(self.runtime,link.send,link.policy_log)
        self.token=secrets.token_urlsafe(32)
        self.preflight_deadline=None

    def start(self, folder):
        super().start(folder)
        p=self.folder/'preview_metadata.json';meta=json.loads(p.read_text())
        meta.update(hand_carried=False,flight_control=True,
            release='vertical-search-v2',startup_search=dict(LIMITS,controller='scripted_startup_search',count_as_learned_policy=False),
            issued_actions='Quantized policy commands handed to UDP transport; delivery unconfirmed',
            ui_start='single click',stop_on_focus_loss=True,
            mode='physical supervised policy test',rc_full=[50,50,100,70],policy_limit_s=None,policy_stop='operator or safety check',
            guards=dict(camera_age_s=.25,telemetry_age_s=.5,operator_age_s=.5,
                        prediction_age_s=.3,min_battery=30,max_temperature=80,height_cm=[30,150]))
        p.write_text(json.dumps(meta,indent=2))

    def analyse(self):
        detector=HoopDetector(*self.calibration.size,confirm=CONFIRM_FRAMES,calibration=self.calibration)
        last=-1
        try:
            while not self.stop.is_set():
                tick=time.monotonic()
                with self.lock: latest=self.latest
                if latest and latest[0]!=last:
                    seq,received,frame=latest;last=seq
                    if tick-received<=.25:
                        detection=detector(frame)
                        self.control.receive_camera(received)
                        self.control.step(seq,received,detection.obs)
                        ok,jpg=cv2.imencode('.jpg',draw(frame,detection))
                        with self.lock:
                            if ok:self.jpg=jpg.tobytes()
                            self.state.update(frame_id=seq,ring=bool(detection.obs[6]))
                self.stop.wait(max(0,.1-(time.monotonic()-tick)))
        except Exception as exc:
            self.status({'error':str(exc)});self.control.stop('camera/policy failure: '+str(exc))

    def snapshot(self):
        result=super().snapshot();result['control']=self.control.snapshot()
        result['preflight_remaining_s']=None if self.preflight_deadline is None or result['control']['state']!='ground' else max(0,self.preflight_deadline-time.monotonic())
        return result

    def open_browser(self):
        preview=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def respond(self,body,mime='application/json',status=200):
                self.send_response(status);self.send_header('Content-Type',mime)
                self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)))
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
                try:self.wfile.write(body)
                except (BrokenPipeError,ConnectionResetError):pass
            def do_GET(self):
                path=urlsplit(self.path).path
                if path=='/':self.respond(PAGE.encode(),'text/html; charset=utf-8')
                elif path=='/state':self.respond(json.dumps(preview.snapshot()).encode())
                elif path=='/frame.jpg':
                    with preview.lock: body=preview.jpg
                    self.respond(body or b'','image/jpeg',200 if body else 404)
                else:self.send_error(404)
            def do_POST(self):
                if self.path!='/control' or self.headers.get('X-Flight-Token')!=preview.token:
                    self.respond(b'{"error":"unauthorized control"}',status=403);return
                try:
                    n=int(self.headers.get('Content-Length','0'))
                    if not 0<n<=1024:raise ValueError('Invalid control request')
                    action=json.loads(self.rfile.read(n))['action']
                    if action=='heartbeat':preview.control.heartbeat()
                    elif action=='start':preview.control.start()
                    elif action=='land':preview.control.stop('operator release/land')
                    elif action=='emergency':preview.control.emergency()
                    else:raise ValueError('Unknown control')
                    self.respond(b'{}')
                except Exception as exc:
                    self.respond(json.dumps(dict(error=str(exc),control_state=preview.control.snapshot()['state'])).encode(),status=400)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        url=f'http://127.0.0.1:{self.server.server_port}/#{self.token}'
        print('Local flight controls are opening. Do not share the control URL.',flush=True)
        webbrowser.open(url)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical-flight',action='store_true',required=True)
    parser.parse_args()
    folder=ROOT/'sessions'/('flight-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    folder.mkdir(parents=True,exist_ok=False)
    link=preview=None;watch_stop=threading.Event();watcher=None
    info=dict(started_utc=utc(),mode='physical supervised policy test',physical_flight_validated=False,error=None)
    try:
        from pc_runtime import PCRuntime
        PCRuntime()  # fail closed on missing/wrong components before opening sockets
        link=CaptureLink(folder);info['start_monotonic']=link.start
        preview=FlightPreview(link);link.preview=preview;preview.start(folder)
        info['policy_identity']=preview.runtime.identity
        link.thread=threading.Thread(target=link.run,daemon=True);link.thread.start()
        link.request('command');link.request('streamon')
        preview.control.capture_ready=True
        def watch():
            while not watch_stop.wait(.02):
                try:preview.control.watch()
                except Exception as exc:preview.control.stop('watchdog error: '+str(exc))
        watcher=threading.Thread(target=watch,daemon=True);watcher.start()
        idle_deadline=time.monotonic()+180
        preview.preflight_deadline=idle_deadline
        preview.open_browser();print('RECORDING:',folder,flush=True)
        print('Preflight window is three minutes. After takeoff, policy and recording run until operator/safety stop.',flush=True)
        while preview.control.state!='stopped':
            preview.control.expire_idle(idle_deadline)
            time.sleep(.05)
    except KeyboardInterrupt:
        if preview:preview.control.stop('Ctrl+C')
    except Exception as exc:
        info['error']=str(exc);print('Flight test issue:',exc,flush=True)
        if preview:preview.control.stop('launcher failure: '+str(exc))
    finally:
        # Keep neutral/land requests and capture alive during bounded landing retries.
        if preview:
            preview.control.stop('shutdown')
            try:
                while preview.control.state=='landing':
                    preview.control.watch();time.sleep(.05)
            except KeyboardInterrupt:
                preview.control.stop('second Ctrl+C');preview.control._land()
            info['flight_outcome']=preview.control.snapshot()
            print('Flight session stopped:',info['flight_outcome']['reason'],flush=True)
            watch_stop.set()
            if watcher:watcher.join(1)
            try:preview.close()
            except Exception as exc:info['preview_close_error']=str(exc)
        if link:
            try:link.send('streamoff')
            except Exception as exc:info['streamoff_error']=str(exc)
            info.update(state_packets=link.state_packets,video_packets=link.video_packets,
                        capture_error=link.error)
            link.close()
        if preview and preview.server:
            preview.server.shutdown();preview.server.server_close()
        info['ended_utc']=utc();(folder/'session.json').write_text(json.dumps(info,indent=2))
        if (folder/'policy_preview.jsonl').exists():
            try:analyse_session(folder)
            except Exception as exc:print('Comparison could not be generated:',exc)
        export(folder)
        print('Saved. Landing requests are not proof of landing; visually verify motors stopped.',flush=True)


if __name__=='__main__':main()
