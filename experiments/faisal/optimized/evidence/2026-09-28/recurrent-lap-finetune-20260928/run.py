import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,torch,gymnasium as gym
R=Path(__file__).parent;sys.path.insert(0,str(R/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
from src.unified_evaluation import evaluate

def save(n,v):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2));t.replace(p)
class Mixture(gym.Env):
 def __init__(self):
  self.envs=[wrap(GuidedLessonEnv(stage=k),single=k<4,metric_ring=True,geometric_return=True) for k in (2,3,4)]
  self.active=self.envs[0];self.observation_space=self.active.observation_space;self.action_space=self.active.action_space;self.rng=np.random.default_rng()
 def reset(self,*,seed=None,options=None):
  if options:raise ValueError('Only genuine episode starts')
  if seed is not None:self.rng=np.random.default_rng(seed)
  self.active=self.envs[self.rng.choice(3,p=[.25,.25,.5])]
  return self.active.reset(seed=int(self.rng.integers(1000000)))
 def step(self,a):return self.active.step(a)
 def close(self):
  for e in self.envs:e.close()
class Logging(BaseCallback):
 def __init__(self):super().__init__();self.outcomes=Counter()
 def _on_step(self):
  for done,info in zip(self.locals['dones'],self.locals['infos']):
   if done:self.outcomes[str(info.get('stage'))+':'+info['outcome']]+=1
  if self.num_timesteps%2048==0:
   row=dict(mode='critic calibration' if self.num_timesteps<=8192 else 'RL training',steps=self.num_timesteps,outcomes=dict(self.outcomes))
   save('status.json',row)
   with (R/'training_outcomes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  return True

def actor_params(m):return [(n,p) for n,p in m.policy.named_parameters() if n.startswith(('lstm_actor.','mlp_extractor.policy_net.','action_net.')) or n=='log_std']
def actor_hash(m):
 h=hashlib.sha256()
 for n,p in actor_params(m):h.update(n.encode());h.update(p.detach().cpu().numpy().tobytes())
 return h.hexdigest()
def main():
 assert not (R/'protocol.json').exists();torch.set_num_threads(1);assert torch.cuda.is_available()
 source=Path('D:/DroneSoccerData/runs/recurrent-learned-transfer-20260928/model_update_300.zip')
 save('protocol.json',dict(method='RecurrentPPO RL after supervised learned-policy initialization',source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),device=torch.cuda.get_device_name(),seed=42,
  max_steps=65536,critic_only_steps=8192,mixture='25% centered single,25% varied single,50% front full-lap natural episode starts',
  checks=[0,8192,16384,32768,65536],lr=.00003,clip=.1,target_kl=.003,
  retention='Stop if centered<22/25 or varied<20/25 after policy updates',
  progress='Stop at32768 if zero full successes and zero front returns across all full suites',
  unchanged='119publicobs,physics,rewards,fullbodyclearance,600step fullmission criteria',independent7M=False))
 env=VecNormalize(DummyVecEnv([Mixture for _ in range(8)]),norm_obs=False,norm_reward=True,gamma=.995)
 m=RecurrentPPO.load(source,env=env,device='cuda',seed=42,learning_rate=3e-5,clip_range=.1,target_kl=.003)
 assert m.num_timesteps==0 and next(m.policy.parameters()).is_cuda
 m.set_logger(configure(str(R/'logs'),['csv']));cb=Logging();initial=actor_hash(m);reason='bounded_budget_complete'
 for step in (0,8192,16384,32768,65536):
  if step:
   for _,p in actor_params(m):p.requires_grad_(step>8192)
   save('status.json',dict(mode='training',steps=m.num_timesteps,target=step))
   m.learn(step-m.num_timesteps,reset_num_timesteps=False,callback=cb)
   if step==8192:
    assert actor_hash(m)==initial
    save('critic_calibration_verification.json',dict(actor_unchanged=True,steps=8192,actor_sha256=initial))
  m.save(R/f'model_{step}.zip');env.save(R/f'reward_normalizer_{step}.pkl');save('status.json',dict(mode='evaluating',steps=step))
  p=RecurrentPredictor(m)
  with torch.inference_mode():report=evaluate(p,4,metric_ring=True,geometric_return=True,episode_reset=p.reset_episode)
  save(f'evaluation_{step}.json',report);s=report['suites'];print(step,{k:v['successful_episodes'] for k,v in s.items()},flush=True)
  if step>8192 and (s['original_centered']['successful_episodes']<22 or s['varied_approach']['successful_episodes']<20):reason='single_pass_retention_failed';break
  if step==32768 and not any(s[k]['successful_episodes'] or s[k]['returned_to_front_episodes'] for k in ('lesson','full_task','full_task_delay_200ms')):reason='no_return_learning_signal';break
 save('status.json',dict(mode='stopped',steps=m.num_timesteps,reason=reason,mission_accepted=False,physical_flight_validated=False));env.close()
if __name__=='__main__':main()
