"""Read-only real-time replay of the latest completed recurrent checkpoint."""
import sys,json,time,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import numpy as np
ROOT=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
sys.path.insert(0,str(ROOT/'source_snapshot'))
import torch
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.return_reset_curriculum import wrap
from src.guided_curriculum import GuidedLessonEnv

state={'message':'Loading checkpoint','checkpoint':None};lock=threading.Lock()
def read_json(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def worker():
    global state
    try:
        torch.set_num_threads(1);loaded=-1;episode=0
        while True:
            complete=sorted(int(p.stem.split('_')[-1]) for p in ROOT.glob('evaluation_update_*.json'))
            if not complete:time.sleep(1);continue
            latest=complete[-1]
            if latest!=loaded:
                model=RecurrentPPO.load(ROOT/f'model_update_{latest}.zip',device='cuda')
                predictor=RecurrentPredictor(model);loaded=latest
            env=wrap(GuidedLessonEnv(stage=4),single=False,metric_ring=True,geometric_return=True)
            obs,info=env.reset(seed=9_100_000+episode);predictor.reset_episode();path=[]
            while True:
                started=time.monotonic()
                with torch.inference_mode():action,_=predictor.predict(obs)
                obs,_,t,tr,info=env.step(action);b=env.unwrapped;path.append(b.pos.tolist())
                phi=np.linspace(0,2*np.pi,65)
                ring=b.hoop_centre+np.outer(np.cos(phi),b._hoop_h)+np.outer(np.sin(phi),b._hoop_v)
                with lock:
                    state=dict(checkpoint=loaded,episode=episode+1,position=b.pos.tolist(),ring=ring.tolist(),path=path[-600:],
                        cage=[b.cage.length,b.cage.width,b.cage.height],heading=float(b.heading),passes=int(info['passes']),
                        outcome=info['outcome'],seconds=info['steps']*b.drone.dt,trainer=read_json(ROOT/'status.json'),
                        evaluation=read_json(ROOT/f'evaluation_update_{loaded}.json'),message='Saved policy replay; not a training rollout')
                time.sleep(max(0,.1-(time.monotonic()-started)))
                if t or tr:break
            env.close();episode+=1;time.sleep(.5)
    except Exception as exc:
        with lock:state=dict(state,error=str(exc))

HTML='''<!doctype html><html><head><meta charset="utf-8"><title>Recurrent drone training</title>
<style>body{margin:0;background:#101827;color:#e8eef9;font:16px system-ui;padding:24px}h1{font-size:24px;margin:0}p{color:#adbdd3}.row{display:flex;gap:16px;flex-wrap:wrap}.card{background:#1c2940;padding:14px;border-radius:10px;flex:1;min-width:160px}canvas{background:#131f32;width:100%;height:60vh;border-radius:12px;touch-action:none}button{padding:8px 16px;background:#2d4667;color:white;border:0;border-radius:6px}#error{color:#ff9191}</style></head><body>
<h1>Recurrent PPO · drone simulator</h1><p>One learned policy with LSTM memory · training on RTX 4070 Ti</p>
<div class="row"><div class="card" id="training">Connecting…</div><div class="card" id="checkpoint">Loading…</div><div class="card" id="flight">Waiting…</div></div>
<p><b>Saved-checkpoint replay of the front-start full mission</b> — this animation is not the trainer’s current rollout. Updates after each completed checkpoint evaluation.</p>
<canvas id="scene"></canvas><p><button id="minus">−</button> <button id="plus">+</button> Drag to rotate · scroll to zoom. Cyan: ring; orange: drone; line: flight path.</p>
<div class="card" id="scores"></div><p id="error"></p>
<script>const c=document.getElementById('scene'),ctx=c.getContext('2d');let state={},zoom=1,angle=.65,drag=null;
const el=id=>document.getElementById(id);function project(p){const [x,y,z]=p,C=state.cage||[2.05,2.05,2.05],a=x-C[0]/2,b=y-C[1]/2;let u=Math.cos(angle)*a-Math.sin(angle)*b,v=Math.sin(angle)*a+Math.cos(angle)*b;let s=Math.min(c.width,c.height)*.27*zoom;return[c.width/2+u*s,c.height*.68+(v*.4-z)*s]}
function line(points,color,width=2){if(!points.length)return;ctx.beginPath();points.forEach((p,i)=>{let q=project(p);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;ctx.clearRect(0,0,c.width,c.height);let C=state.cage||[2.05,2.05,2.05];let floor=[[0,0,0],[C[0],0,0],[C[0],C[1],0],[0,C[1],0],[0,0,0]];line(floor,'#64748b');line(floor.map(p=>[p[0],p[1],C[2]]),'#45556d');floor.slice(0,4).forEach(p=>line([p,[p[0],p[1],C[2]]],'#45556d'));line(state.path||[],'#8295b3',1.5);line(state.ring||[],'#37d8e0',5);if(state.position){let p=state.position;line([[p[0]-.1,p[1],p[2]],[p[0]+.1,p[1],p[2]]],'#ffba65',5);line([[p[0],p[1]-.1,p[2]],[p[0],p[1]+.1,p[2]]],'#ffba65',5);let q=project(p);ctx.beginPath();ctx.arc(...q,5,0,Math.PI*2);ctx.fillStyle='#ffe0ac';ctx.fill()}}
c.onpointerdown=e=>{drag=e.clientX;c.setPointerCapture(e.pointerId)};c.onpointermove=e=>{if(drag!==null){angle+=(e.clientX-drag)*.01;drag=e.clientX;draw()}};c.onpointerup=()=>drag=null;c.onwheel=e=>{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};el('minus').onclick=()=>{zoom=Math.max(.4,zoom/1.2);draw()};el('plus').onclick=()=>{zoom=Math.min(4,zoom*1.2);draw()};
async function poll(){try{state=await(await fetch('/state')).json();let t=state.trainer||{};el('training').textContent='Trainer: '+(t.mode||'loading')+(t.target?' · target '+t.target.toLocaleString()+' steps':'')+(t.reason?' · '+t.reason.replaceAll('_',' '):'');el('checkpoint').textContent='Replay checkpoint: '+(state.checkpoint??'…')+' supervised updates';el('flight').textContent='Replay episode '+(state.episode||0)+' · '+(state.seconds||0).toFixed(1)+'s · passes '+(state.passes||0)+' · '+(state.outcome||'loading');let suites=state.evaluation?.suites||{};el('scores').textContent=Object.entries(suites).map(([k,v])=>k.replaceAll('_',' ')+': '+v.successful_episodes+'/25').join(' | ');el('error').textContent=state.error||'';draw()}catch(e){el('error').textContent='Viewer disconnected'}setTimeout(poll,200)}poll();</script></body></html>'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path=='/':body=HTML.encode();mime='text/html; charset=utf-8'
        elif self.path=='/state':
            with lock:payload=dict(state)
            payload['trainer']=read_json(ROOT/'status.json')
            body=json.dumps(payload).encode();mime='application/json'
        else:self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

if __name__=='__main__':
    server=ThreadingHTTPServer(('127.0.0.1',8792),Handler)
    threading.Thread(target=worker,daemon=True).start()
    print('Recurrent checkpoint viewer: http://127.0.0.1:8792/',flush=True)
    server.serve_forever()
