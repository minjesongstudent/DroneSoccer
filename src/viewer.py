"""Local browser replay of a learned policy in the group simulator; no flight I/O."""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from src.clearance_training import public_features
from src.pass_audit import PassageAudit
from src.policy_runtime import LearnedPolicy, SPECS
from src.sim import HoopEnv

STATE = {"message": "Loading learned policy"}
LOCK = threading.Lock()
HTML = '''<!doctype html><meta charset="utf-8"><title>Group simulator</title>
<style>body{background:#101827;color:#eee;font:16px system-ui;margin:24px}canvas{background:#19263b;width:100%;height:65vh}a{color:#6ee7ef}</style>
<h1>Group simulator · learned policy replay</h1>
<p>Group simulator, nominal level-1 layouts. No real drone connection.</p>
<p>Passes count full-body clearances. Later crashes still invalidate a clean mission. This replay is not an acceptance test or a flight release.</p>
<p id="status">Loading…</p><canvas id="scene"></canvas>
<script>const c=document.getElementById('scene'),x=c.getContext('2d');
function draw(s){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;
const C=s.cage||[2.05,2.05,2.05],scale=Math.min(c.width,c.height)*.27;
function p(v){let a=v[0]-C[0]/2,b=v[1]-C[1]/2;return[c.width/2+(a*.8-b*.6)*scale,c.height*.75+((a*.6+b*.8)*.4-v[2])*scale]}
function line(a,col,w=2){if(!a.length)return;x.beginPath();a.forEach((v,i)=>{let q=p(v);i?x.lineTo(...q):x.moveTo(...q)});x.strokeStyle=col;x.lineWidth=w;x.stroke()}
let f=[[0,0,0],[C[0],0,0],[C[0],C[1],0],[0,C[1],0],[0,0,0]];line(f,'#64748b');line(f.map(v=>[v[0],v[1],C[2]]),'#64748b');f.slice(0,4).forEach(v=>line([v,[v[0],v[1],C[2]]],'#64748b'));
line(s.ring||[],'#37d8e0',5);line(s.path||[],'#8295b3');if(s.position){let q=p(s.position);x.beginPath();x.arc(...q,7,0,7);x.fillStyle='#ffba65';x.fill()}}
async function poll(){try{let s=await(await fetch('/state')).json();document.getElementById('status').textContent=s.error||`${s.model||''} · episode ${s.episode||0} · ${s.seconds||0}s · passes ${s.passes||0} · ${s.outcome||s.message||''}`;draw(s)}catch(e){document.getElementById('status').textContent='Viewer disconnected'}setTimeout(poll,200)}poll();</script>'''

def load_replay_policy(model_path):
    path=Path(model_path)
    if path.stem not in SPECS:
        raise ValueError('Select a named policy with its recorded observation history')
    return LearnedPolicy(path,SPECS[path.stem])


def worker(model_path):
    global STATE
    try:
        torch.set_num_threads(1)
        policy = load_replay_policy(model_path)
        env = HoopEnv(level=1.0)
        episode = 0
        while True:
            obs, _ = env.reset(seed=9_200_000 + episode)
            policy.reset();previous=np.zeros(4,np.float32)
            audit=PassageAudit();audit.update(env)
            path = []
            while True:
                start = time.monotonic()
                _,action = policy.predict(public_features(obs),previous)
                previous=action
                obs, _, terminated, truncated, info = env.step(action)
                audit.update(env,info['outcome'],terminated or info['wall_steps']>0)
                path.append(env.pos.tolist())
                phi = np.linspace(0, 2*np.pi, 65)
                ring = env.hoop_centre + np.outer(np.cos(phi), env._hoop_h) + np.outer(np.sin(phi), env._hoop_v)
                with LOCK:
                    STATE = dict(model=Path(model_path).name, episode=episode+1,
                                 position=env.pos.tolist(), path=list(path), ring=ring.tolist(),
                                 cage=[env.cage.length, env.cage.width, env.cage.height],
                                 seconds=round(info['steps']*env.dyn.dt, 1),
                                 passes=audit.count, centre_passes=info['passes'],
                                 returns=audit.returns, policy_inputs=policy.spec.inputs,
                                 outcome=info['outcome'])
                time.sleep(max(0, env.dyn.dt-(time.monotonic()-start)))
                if terminated or truncated:
                    break
            episode += 1
            time.sleep(1)
    except Exception as exc:
        with LOCK:
            STATE = {"error": str(exc)}

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            body, mime = HTML.encode(), 'text/html; charset=utf-8'
        elif self.path == '/state':
            with LOCK:
                body = json.dumps(STATE).encode()
            mime = 'application/json'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', mime)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='trainedModels/seenflag-s1-level1.zip')
    parser.add_argument('--port', type=int, default=8793)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=worker, args=(args.model,), daemon=True).start()
    print(f'Group simulator: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()

if __name__ == '__main__':
    main()
