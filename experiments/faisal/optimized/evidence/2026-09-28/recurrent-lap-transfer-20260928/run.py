import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
R=Path(__file__).parent;sys.path.insert(0,str(R/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.unified_evaluation import evaluate

def save(n,v):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def tensor(a):return torch.as_tensor(a,dtype=torch.float32,device='cuda')
def loss(m,data):
 o,a,mask=data;b,t,_=o.shape;z=torch.zeros(1,b,128,device='cuda')
 d,_=m.policy.get_distribution(o.reshape(-1,119),(z,z.clone()),torch.zeros(b*t,device='cuda'))
 err=(d.distribution.mean.reshape(b,t,4)-a).square().mean(-1)
 return ((err*mask).sum(1)/mask.sum(1)).mean()
def main():
 torch.set_num_threads(1);torch.manual_seed(43);assert torch.cuda.is_available();assert not (R/'protocol.json').exists()
 bank=Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3');source=Path('D:/DroneSoccerData/runs/recurrent-lap-finetune-20260928/model_32768.zip')
 seeds=json.loads((bank/'result.json').read_text())['successes_by_split']['fit'];assert len(seeds)==9 and all(s%4!=3 for s in seeds)
 save('protocol.json',dict(method='Supervised complete-history transfer from rare learned-policy lap successes with basic-skill rehearsal',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),fit_lap_seeds=seeds,selected_teacher_successes=True,collection_denominator=2048,collection_total_successes=11,reserved_successes_used=False,independent7Mused=False,
  public_observation_dimensions=119,replay_verification='All9 matched current observations exactly and ended600steps2passes; no invented recurrent history',max_updates=500,checks=[0,100,300,500],lr=3e-5,batch='8 basic +8 lap entire episodes; equal loss weight per group and per episode',retention='Stop if centered<22/25 or varied<20/25; no acceptance from demonstrations'))
 d=np.load('D:/DroneSoccerData/runs/recurrent-learned-transfer-20260928/data_600000.npz');basic=tuple(tensor(d[k]) for k in ('observations','actions','mask'))
 laps=[np.load(bank/'trajectories'/f'{s}.npz') for s in seeds];lap=(tensor(np.stack([d['observations'][:-1] for d in laps])),tensor(np.stack([d['actions'] for d in laps])),torch.ones(9,600,device='cuda'))
 save('dataset_hashes.json',{str(bank/'trajectories'/f'{s}.npz'):hashlib.sha256((bank/'trajectories'/f'{s}.npz').read_bytes()).hexdigest() for s in seeds})
 m=RecurrentPPO.load(source,device='cuda');params=[p for n,p in m.policy.named_parameters() if n.startswith(('lstm_actor.','mlp_extractor.policy_net.','action_net.'))];opt=torch.optim.Adam(params,lr=3e-5);rng=np.random.default_rng(43);history=[];reason='bounded_transfer_complete'
 for u in range(501):
  if u:
   m.policy.set_training_mode(True);bi=rng.choice(80,8,replace=False);li=rng.choice(9,8,replace=False);opt.zero_grad()
   l=.5*loss(m,tuple(x[bi] for x in basic))+.5*loss(m,tuple(x[li] for x in lap));assert torch.isfinite(l);l.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step()
  if u%25==0:
   with torch.inference_mode():row=dict(supervised_updates=u,basic_mse=float(loss(m,basic)),lap_fit_mse=float(loss(m,lap)))
   history.append(row);save('fit_history.json',history);save('status.json',dict(mode='supervised lap transfer',**row))
  if u in (0,100,300,500):
   m.save(R/f'model_update_{u}.zip');save('status.json',dict(mode='evaluating',supervised_updates=u));p=RecurrentPredictor(m)
   with torch.inference_mode():report=evaluate(p,4,metric_ring=True,geometric_return=True,episode_reset=p.reset_episode)
   report['supervised_updates']=u;save(f'evaluation_update_{u}.json',report);s=report['suites'];print(u,{k:v['successful_episodes'] for k,v in s.items()},flush=True)
   if u and (s['original_centered']['successful_episodes']<22 or s['varied_approach']['successful_episodes']<20):reason='single_pass_retention_failed';break
 save('status.json',dict(mode='stopped',supervised_updates=u,reason=reason,mission_accepted=False,physical_flight_validated=False))
if __name__=='__main__':main()
