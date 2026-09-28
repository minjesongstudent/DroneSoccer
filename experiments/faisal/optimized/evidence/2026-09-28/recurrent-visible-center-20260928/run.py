import sys,json,hashlib,time
from pathlib import Path
from collections import Counter
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT/'source_snapshot'))
import numpy as np,torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
from src.unified_evaluation import evaluate

def save(name,value):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def make():return wrap(GuidedLessonEnv(stage=2),single=True,metric_ring=True,geometric_return=True)
class Logging(BaseCallback):
 def __init__(self):super().__init__();self.completed=0;self.outcomes=Counter()
 def _on_step(self):
  for done,info in zip(self.locals['dones'],self.locals['infos']):
   if done:self.completed+=1;self.outcomes[info['outcome']]+=1
  if self.num_timesteps%2048==0:
   with (OUT/'training_outcomes.jsonl').open('a') as f:f.write(json.dumps(dict(steps=self.num_timesteps,episodes=self.completed,outcomes=dict(self.outcomes)))+'\n')
   save('status.json',dict(mode='training',steps=self.num_timesteps,target=65536,episodes=self.completed,outcomes=dict(self.outcomes)))
  return True

def centered_eval(model,step):
 p=RecurrentPredictor(model);rows=[]
 for i in range(25):
  e=make();o,info=e.reset(seed=9300000+i);p.reset_episode();reward=0
  while True:
   with torch.inference_mode():a,_=p.predict(o)
   o,r,t,tr,info=e.step(a);reward+=r
   if t or tr:break
  rows.append(dict(seed=9300000+i,outcome=info['outcome'],passes=info['passes'],steps=info['steps'],reward=reward));e.close()
 result=dict(step=step,episodes=25,successes=sum(x['outcome']=='passed' for x in rows),rows=rows)
 save(f'centered_{step}.json',result);return result

def main():
 assert not (OUT/'protocol.json').exists()
 torch.set_num_threads(1);assert torch.cuda.is_available()
 save('protocol.json',dict(algorithm='RecurrentPPO',device=torch.cuda.get_device_name(),seed=41,max_steps=65536,
  controlled_change='Natural centered stage2 instead of camera-invisible centered stage0; same low-noise baseline configuration',initial_log_std=-1.6094379124341003,
  rewards_and_geometry='unchanged',public_observations=119,scripted_actions=False,
  checkpoints=[0,16384,32768,65536],centered_development_seeds='9300000..9300024',
  fixed_evaluation='unchanged five suites25episodes; no independent7M used',
  decision='Stop at64k; no automatic promotion or extension. Assess centered-learning before advancing curriculum.'))
 env=VecNormalize(DummyVecEnv([make for _ in range(8)]),norm_obs=False,norm_reward=True,gamma=.995)
 m=RecurrentPPO('MlpLstmPolicy',env,device='cuda',seed=41,n_steps=256,batch_size=256,n_epochs=5,
  learning_rate=3e-4,gamma=.995,gae_lambda=.95,clip_range=.2,target_kl=.01,ent_coef=.01,
  policy_kwargs=dict(log_std_init=-1.6094379124341003,lstm_hidden_size=128,net_arch=dict(pi=[128],vf=[128])),verbose=0)
 assert np.allclose(m.policy.log_std.detach().cpu().exp().numpy(),.2)
 m.set_logger(configure(str(OUT/'logs'),['csv']))
 cb=Logging()
 for step in (0,16384,32768,65536):
  if step:
   save('status.json',dict(mode='training',steps=int(m.num_timesteps),target=step))
   m.learn(step-m.num_timesteps,reset_num_timesteps=False,callback=cb)
  assert m.num_timesteps==step and next(m.policy.parameters()).is_cuda
  m.save(OUT/f'model_{step}.zip');env.save(OUT/f'reward_normalizer_{step}.pkl')
  save('status.json',dict(mode='evaluating',steps=step))
  centered=centered_eval(m,step);p=RecurrentPredictor(m)
  with torch.inference_mode():report=evaluate(p,4,metric_ring=True,geometric_return=True,episode_reset=p.reset_episode)
  save(f'evaluation_{step}.json',report)
  print(step,'centered',centered['successes'],'fixed',{k:v['successful_episodes'] for k,v in report['suites'].items()},flush=True)
 save('status.json',dict(mode='stopped',steps=int(m.num_timesteps),reason='bounded_centered_trial_complete',mission_accepted=False))
 env.close()
if __name__=='__main__':main()
