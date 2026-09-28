"""Separate reset-lesson progression; cannot certify full-mission success."""
from copy import deepcopy
import gymnasium as gym
import numpy as np
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation
from src.front_memory import FrontMemoryObservation
from src.return_reset_lessons import ReturnResetLesson, LEVELS, VERSION as LESSON_VERSION
from src.return_evaluation import _fingerprint

VERSION='validated-fresh-return-curriculum-v2'
SEED_START=8_700_000
EPISODES=25
COVERAGE_VERSION='uniform-six-reset-coverage-v1'


def wrap(env,single=False,metric_ring=False,geometric_return=False):
    result=FrontMemoryObservation(MissionObservation(env,single_pass=single))
    if metric_ring:
        from src.ring_estimate import MetricRingObservation
        result=MetricRingObservation(result)
    if geometric_return:
        from src.return_geometry import GeometricReturnProgress
        result=GeometricReturnProgress(result)
    return result


class ResetScheduler:
    def __init__(self):
        self.level=0;self.streak=0;self.last_step=-1;self.history=[];self.stop_reason=None

    @property
    def should_stop(self):return self.stop_reason is not None

    def observe(self,report,centered,varied):
        if self.should_stop:raise ValueError('Reset curriculum already stopped')
        if report.get('version')!=VERSION or report.get('lesson_version')!=LESSON_VERSION or report.get('episodes')!=EPISODES or report.get('seed_start')!=SEED_START:
            raise ValueError('Reset evaluation protocol changed')
        step=report['step']
        if isinstance(step,bool) or not isinstance(step,int) or step<=self.last_step:
            raise ValueError('Evaluation steps must increase')
        scores=report['successes']
        if len(scores)!=len(LEVELS) or any(isinstance(v,bool) or not isinstance(v,int) or not 0<=v<=EPISODES for v in scores):
            raise ValueError('Invalid reset evaluation counts')
        if any(not np.isfinite(v) or not 0<=v<=1 for v in (centered,varied)):
            raise ValueError('Invalid passing retention')
        if any(v<20 for v in scores[:self.level]):
            self.stop_reason='reset_mastered_lesson_regression'
        retained=centered>=.84 and varied>=.80
        eligible=not self.should_stop and retained and all(v>=20 for v in scores[:self.level+1])
        self.streak=self.streak+1 if eligible else 0
        previous=self.level
        if self.streak>=2 and self.level<len(LEVELS)-1:
            self.level+=1;self.streak=0
        self.last_step=step
        self.history.append(dict(step=step,successes=list(scores),centered=centered,varied=varied,level_before=previous,level=self.level,streak=self.streak))
        return dict(level=self.level,advanced=self.level!=previous,passing_retained=retained,full_mission_accepted=False,should_stop=self.should_stop,stop_reason=self.stop_reason)

    def state(self):return dict(version=VERSION,history=deepcopy(self.history))

    @classmethod
    def restore(cls,state):
        if state.get('version')!=VERSION:raise ValueError('Reset scheduler version differs')
        obj=cls()
        for row in state['history']:
            obj.observe(dict(version=VERSION,lesson_version=LESSON_VERSION,episodes=EPISODES,seed_start=SEED_START,step=row['step'],successes=row['successes']),row['centered'],row['varied'])
            if obj.history[-1]!=row:raise ValueError('Reset scheduler history is inconsistent')
        return obj


class ResetMixture(gym.Env):
    """40% single-pass,20% full flights,20% current reset,20% earlier resets."""
    def __init__(self,scheduler,metric_ring=False,coverage=False,geometric_return=False):
        self.coverage=bool(coverage)
        self.scheduler=scheduler
        self.single=[wrap(GuidedLessonEnv(stage=k),True,metric_ring,geometric_return) for k in (2,3)]
        self.full=wrap(GuidedLessonEnv(stage=4),metric_ring=metric_ring,geometric_return=geometric_return)
        self.lessons=[wrap(ReturnResetLesson(k),metric_ring=metric_ring,geometric_return=geometric_return) for k in range(len(LEVELS))]
        self.active=self.single[0];self.rng=np.random.default_rng()
        self.observation_space=self.active.observation_space;self.action_space=self.active.action_space

    def set_scheduler(self,scheduler):
        if not isinstance(scheduler,ResetScheduler):raise ValueError('Reset scheduler required')
        self.scheduler=scheduler

    def set_stage(self,stage):
        if stage!=4:raise ValueError('Full-flight focus remains stage4 in reset trial')

    @property
    def unwrapped(self):return self.active.unwrapped

    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        choice=int(self.rng.integers(5))
        if choice<2:self.active=self.single[choice]
        elif choice==2:self.active=self.full
        else:
            k=int(self.rng.integers(len(LEVELS))) if self.coverage else self.scheduler.level if choice==3 else int(self.rng.integers(max(1,self.scheduler.level)))
            self.active=self.lessons[k]
        obs,info=self.active.reset(seed=int(self.rng.integers(1_000_000)),options=options)
        return obs,dict(info,focus_stage=4)

    def step(self,action):
        obs,r,t,tr,info=self.active.step(action)
        return obs,r,t,tr,dict(info,focus_stage=4)

    def close(self):
        for env in [*self.single,self.full,*self.lessons]:env.close()


def evaluate_reset_lessons(model,progress=None,metric_ring=False,geometric_return=False):
    before=_fingerprint(model);records=[]
    for level in range(len(LEVELS)):
        env=wrap(ReturnResetLesson(level),metric_ring=metric_ring,geometric_return=geometric_return)
        try:
            for i in range(EPISODES):
                obs,info=env.reset(seed=SEED_START+i)
                while True:
                    obs,_,term,trunc,info=env.step(model.predict(obs,deterministic=True)[0])
                    if term or trunc:break
                records.append(dict(level=level,seed=SEED_START+i,success=bool(info['reset_lesson_success']),passes=info['passes'],outcome=info['outcome']))
                if progress:progress(level,i+1,EPISODES)
        finally:env.close()
    if _fingerprint(model)!=before:raise RuntimeError('Evaluation modified the model')
    return dict(version=VERSION,lesson_version=LESSON_VERSION,step=int(model.num_timesteps),episodes=EPISODES,seed_start=SEED_START,successes=[sum(r['success'] for r in records if r['level']==k) for k in range(len(LEVELS))],records=records,full_mission_test=False)
