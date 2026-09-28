import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
R=Path(__file__).parent;sys.path.insert(0,str(R/'source_snapshot'))
from src.standard_source_policy import StandardSourceSAC
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
from src.unified_evaluation import evaluate
from sb3_contrib import RecurrentPPO

def save(n,v):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
def make(stage):return wrap(GuidedLessonEnv(stage=stage),single=True,metric_ring=True,geometric_return=True)
def collect(teacher,count,seed):
 obs=np.zeros((count,150,119),np.float32);act=np.zeros((count,150,4),np.float32);mask=np.zeros((count,150),np.float32);rows=[]
 for i in range(count):
  e=make(2+i%2);o,info=e.reset(seed=seed+i)
  for t in range(150):
   with torch.inference_mode():a,_=teacher.predict(o,deterministic=True)
   obs[i,t]=o;act[i,t]=a;mask[i,t]=1;o,_,done,tr,info=e.step(a)
   if done or tr:break
  rows.append(dict(seed=seed+i,stage=2+i%2,steps=t+1,outcome=info['outcome'],passes=info['passes']));e.close()
 np.savez_compressed(R/f'data_{seed}.npz',observations=obs,actions=act,mask=mask)
 save(f'flights_{seed}.json',rows)
 return tuple(torch.as_tensor(x,device='cuda') for x in (obs,act,mask))
def outputs(m,obs):
 b,t,_=obs.shape;z=torch.zeros((1,b,128),device='cuda')
 d,_=m.policy.get_distribution(obs.reshape(-1,119),(z,z.clone()),torch.zeros(b*t,device='cuda'))
 return d.distribution.mean.reshape(b,t,4)
def loss(m,data):
 o,a,mask=data;return (((outputs(m,o)-a)**2).mean(-1)*mask).sum()/mask.sum()
def main():
 torch.set_num_threads(1);torch.manual_seed(41);assert torch.cuda.is_available();assert not (R/'protocol.json').exists()
 source=Path('D:/DroneSoccerData/runs/standard-source-20260928/initial_model.zip')
 save('protocol.json',dict(method='Supervised recurrent initialization from preserved learned SAC commands; NOT pure RL or scripted demonstrations',
  source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),
  fit_episodes=80,fit_seeds='600000..600079',holdout_episodes=20,holdout_seeds='640000..640019',
  failed_teacher_flights_retained=True,observations=119,privileged_actor_inputs=False,whole_authentic_sequences=True,
  max_updates=1000,checks=[100,300,1000],batch_episodes=16,lr=.0003,success_gates='Original fixed suites unchanged; transfer is not fullmission acceptance',independent_7M_used=False))
 save('status.json',dict(mode='collecting learned-policy flights'))
 teacher=StandardSourceSAC.load(source,device='cuda');fit=collect(teacher,80,600000);holdout=collect(teacher,20,640000);del teacher
 e=make(2);m=RecurrentPPO('MlpLstmPolicy',e,device='cuda',seed=41,n_steps=256,batch_size=256,n_epochs=5,learning_rate=3e-4,gamma=.995,gae_lambda=.95,clip_range=.2,target_kl=.01,ent_coef=.01,policy_kwargs=dict(log_std_init=-1.6094379124341003,lstm_hidden_size=128,net_arch=dict(pi=[128],vf=[128])),verbose=0)
 # Verify sequence batching matches actual step-by-step inference before fitting.
 with torch.inference_mode():
  batched=outputs(m,fit[0][:2,:12]).cpu().numpy()
  for i in range(2):
   p=RecurrentPredictor(m)
   for t in range(12):
    a,_=p.predict(fit[0][i,t].cpu().numpy());assert np.allclose(a,batched[i,t].clip(-1,1),atol=1e-6)
 params=list(m.policy.lstm_actor.parameters())+list(m.policy.mlp_extractor.policy_net.parameters())+list(m.policy.action_net.parameters())
 opt=torch.optim.Adam(params,lr=3e-4);rng=np.random.default_rng(41);history=[]
 m.save(R/'initial_model.zip')
 for update in range(1,1001):
  m.policy.set_training_mode(True)
  idx=rng.choice(80,16,replace=False);data=tuple(x[idx] for x in fit);opt.zero_grad();l=loss(m,data);assert torch.isfinite(l);l.backward();torch.nn.utils.clip_grad_norm_(params,.5);opt.step()
  if update%25==0:
   with torch.inference_mode():row=dict(update=update,fit_mse=float(loss(m,fit)),holdout_mse=float(loss(m,holdout)))
   history.append(row);save('fit_history.json',history);save('status.json',dict(mode='fitting learned-policy transfer',**row))
  if update in (100,300,1000):
   m.save(R/f'model_update_{update}.zip');save('status.json',dict(mode='evaluating',updates=update));p=RecurrentPredictor(m)
   with torch.inference_mode():report=evaluate(p,4,metric_ring=True,geometric_return=True,episode_reset=p.reset_episode)
   report['supervised_updates']=update;save(f'evaluation_update_{update}.json',report)
   counts={k:v['successful_episodes'] for k,v in report['suites'].items()};print(update,counts,flush=True)
   if counts['original_centered']>=23 and counts['varied_approach']>=20:break
 save('status.json',dict(mode='stopped',supervised_updates=update,reason='bounded_transfer_complete',mission_accepted=False,physical_flight_validated=False));e.close()
if __name__=='__main__':main()
