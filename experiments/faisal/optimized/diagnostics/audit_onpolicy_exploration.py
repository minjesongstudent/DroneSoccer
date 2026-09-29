"""Matched training-layout diagnostic of deterministic vs learned stochastic actions."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib
import torch
import numpy as np

ROOT=Path(__file__).parent
RUN=ROOT/'lap-onpolicy-recurrent50-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

torch.set_num_threads(1)
path=RUN/'model_4096.zip'
before=hashlib.sha256(path.read_bytes()).hexdigest()
model=RecurrentPPO.load(path,device='cuda')
report={'checkpoint_sha256':before,'seed_range':[610100,610163],
        'training_domain_only':True,'reserved_or_acceptance_used':False,'suites':{}}
for deterministic in (True,False):
    torch.manual_seed(460)
    rows=[]
    for seed in range(610100,610164):
        env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
        obs,_=env.reset(seed=seed);p=RecurrentPredictor(model)
        while True:
            with torch.inference_mode():action,_=p.predict(obs,deterministic=deterministic)
            obs,_,t,tr,info=env.step(action)
            if t or tr:break
        rows.append({'seed':seed,'outcome':info['outcome'],'passes':info['passes'],'steps':info['steps']})
        env.close()
    result=dict(episodes=64,first_pass_episodes=sum(r['passes']>0 for r in rows),
        full_missions=sum(r['outcome']=='timeout' and r['passes']>=2 for r in rows),
        outcomes=dict(Counter(r['outcome'] for r in rows)),records=rows)
    report['suites']['deterministic' if deterministic else 'stochastic']=result
    print(deterministic,{k:v for k,v in result.items() if k!='records'},flush=True)
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
(ROOT/'onpolicy_exploration_audit.json').write_text(json.dumps(report,indent=2))
