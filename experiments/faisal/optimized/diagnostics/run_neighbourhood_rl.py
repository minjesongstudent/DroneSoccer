"""Bounded fresh-critic SAC trial around successful FIT reset layouts."""
from pathlib import Path
import sys,json,hashlib,shutil,time
import numpy as np
import torch
import gymnasium as gym

ROOT=Path(__file__).parent
OUT=ROOT/'neighbourhood-rl-20260928'
SOURCE=ROOT/'lap-recurrent-anchor10-20260928/source_snapshot'
sys.path.insert(0,str(OUT/'source_snapshot'))

def save(name,value):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2))
    for attempt in range(30):
        try:tmp.replace(OUT/name);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.1)

def main():
    OUT.mkdir(exist_ok=False)
    shutil.copytree(SOURCE,OUT/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,OUT/'runner.py')
    from src.guided_curriculum import GuidedLessonEnv
    from src.return_reset_curriculum import wrap
    from src.metric_policy import MetricRingSAC,MetricActorPolicy
    from src.geometric_policy import GeometricSAC
    from src.unified_evaluation import evaluate
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    seeds=[2820026,2820664,2820750,2820805,2820828,2821928,2821958,2822002,2822042]
    class LocalEnv(GuidedLessonEnv):
        def __init__(self):
            self.offset=np.zeros(3)
            super().__init__(stage=4)
        def _sample_layout(self):
            super()._sample_layout()
            p=self.pos+self.offset[0]*self._axis_h+np.array([0,0,self.offset[1]])
            self._set_layout(p,self.heading+self.offset[2],self.hoop_centre.copy(),self.hoop_normal.copy(),
                             np.linalg.norm(self._hoop_h),np.linalg.norm(self._hoop_v))
    class Mixture(gym.Env):
        def __init__(self):
            self.local=LocalEnv()
            self.envs=[wrap(GuidedLessonEnv(stage=k),single=True,metric_ring=True,geometric_return=True) for k in (2,3)]
            self.envs.append(wrap(self.local,metric_ring=True,geometric_return=True))
            self.active=self.envs[0];self.action_space=self.active.action_space;self.observation_space=self.active.observation_space
            self.rng=np.random.default_rng(61)
        def reset(self,*,seed=None,options=None):
            if seed is not None:self.rng=np.random.default_rng(seed)
            k=int(self.rng.choice(3,p=[.25,.25,.5]));self.active=self.envs[k]
            if k==2:
                layout=int(self.rng.choice(seeds))
                self.local.offset=self.rng.uniform(-1,1,3)*[.01,.01,np.pi/180]
            else:layout=int(self.rng.integers(0,1000000))
            return self.active.reset(seed=layout)
        def step(self,a):return self.active.step(a)
        def close(self):
            for e in self.envs:e.close()
    class Log(BaseCallback):
        def _on_step(self):
            if self.num_timesteps%1000==0:
                save('status.json',dict(mode=self.model.transfer_phase,steps=self.num_timesteps,actor_updates=self.model.actor_updates))
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if done:
                    with (OUT/'episodes.jsonl').open('a') as f:f.write(json.dumps({k:info[k] for k in ('stage','outcome','passes','steps')})+'\n')
            return True
    torch.set_num_threads(1);assert torch.cuda.is_available()
    path=Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip')
    source_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    old=MetricRingSAC.load(path,device='cuda');env=Mixture()
    m=GeometricSAC(MetricActorPolicy,env,device='cuda',seed=61,buffer_size=40000,learning_starts=0,
        batch_size=256,tau=old.tau,gamma=old.gamma,train_freq=(1,'step'),gradient_steps=1,
        ent_coef=float(old.ent_coef_tensor.item()),target_update_interval=old.target_update_interval,
        policy_kwargs={'net_arch':old.actor.net_arch,'activation_fn':old.actor.activation_fn})
    m.actor.load_state_dict(old.actor.state_dict());m.reference_actor.load_state_dict(old.actor.state_dict())
    # New reference is explicitly this starting actor, not the older inherited reference.
    m.set_logger(configure(str(OUT/'logs'),['csv']))
    o,_=env.reset(seed=61)
    with torch.inference_mode():
        np.testing.assert_array_equal(m.predict(o,deterministic=True)[0],old.predict(o,deterministic=True)[0])
    assert m.replay_buffer.size()==0 and m.actor_updates==0
    assert all(torch.equal(v,m.reference_actor.state_dict()[k]) for k,v in m.actor.state_dict().items())
    save('protocol.json',dict(source=str(path),source_sha256=source_hash,seed=61,max_steps=16000,
        checks=[0,10000,12000,16000],fit_seeds=seeds,mixture='25% centered,25% varied,50% nearby FIT full laps',
        perturbation='Independent uniform +/-1cm lateral, +/-1cm height, +/-1degree yaw; applied before reset histories.',
        reference='Fresh frozen copy of starting producer actor; original reference preserved only in source checkpoint.',
        fresh_critics=True,fresh_optimizers=True,fresh_replay=True,refill=5000,critic_calibration_updates=5000,
        anchor_coefficient=m.anchor_coefficient,anchor_max_kl=m.anchor_max_kl,
        actor_lr=m.actor_learning_rate,critic_lr=m.critic_learning_rate,
        retention='Stop if centered below22/25 or varied below21/25; no automatic extension.',
        truth='Reset/reward/diagnostics only; actor unchanged119public inputs; all flight actions learned.',acceptance=False))
    cb=Log();last=0;reason='bounded_budget_complete'
    for step in (0,10000,12000,16000):
        if step:m.learn(total_timesteps=step-last,reset_num_timesteps=last==0,callback=cb)
        with torch.inference_mode():report=evaluate(m,4,metric_ring=True,geometric_return=True)
        save(f'eval_{step}.json',report);m.save(OUT/f'model_{step}.zip')
        counts={k:v['successful_episodes'] for k,v in report['suites'].items()}
        print(step,counts,flush=True)
        last=step
        if counts['original_centered']<22 or counts['varied_approach']<21:
            reason='single_pass_retention_failed';break
    m.save_replay_buffer(OUT/'replay.pkl')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==source_hash
    save('status.json',dict(mode='stopped',steps=last,reason=reason,source_unchanged=True,mission_accepted=False))
    env.close()

if __name__=='__main__':main()
