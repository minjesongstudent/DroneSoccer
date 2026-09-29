"""FIT-only parameter interpolation diagnostic; no optimization or promotion."""
from pathlib import Path
import sys,json,hashlib
import torch
ROOT=Path(__file__).parent
RUN=ROOT/'history-guard-rl-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.history_guard_policy import HistoryGuardSAC
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
torch.set_num_threads(1)
paths=[RUN/'model_0.zip',RUN/'model_10000.zip']
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
a,b=[HistoryGuardSAC.load(p,device='cuda') for p in paths]
initial={k:v.clone() for k,v in a.actor.state_dict().items()}
final=b.actor.state_dict()
rows=[]
seeds=json.loads((RUN/'protocol.json').read_text())['fit_seeds']
with torch.inference_mode():
    for fraction in (0.,.01,.05,.1,.5,1.):
        a.actor.load_state_dict({k:v+fraction*(final[k]-v) for k,v in initial.items()})
        for seed in seeds:
            env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
            obs,_=env.reset(seed=seed)
            while True:
                obs,_,t,tr,info=env.step(a.predict(obs,deterministic=True)[0])
                if t or tr:break
            rows.append(dict(fraction=fraction,seed=seed,steps=info['steps'],passes=info['passes'],outcome=info['outcome'],
                full=info['outcome']=='timeout' and info['passes']>=2 and info['steps']==600))
            env.close()
        print(fraction,'full',sum(r['full'] for r in rows if r['fraction']==fraction),flush=True)
assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
result=dict(actor_updates=b.actor_updates,actor_attempts=b.actor_attempts,history_rejected=b.history_rejected,
    history_kl=b.history_kl(),source_hashes=hashes,rows=rows,
    caveat='Parameter interpolation of one learned update only, not a learning-rate trial or independent validation; no candidate saved/promoted, no optimizer invoked.')
(RUN/'update_sensitivity.json').write_text(json.dumps(result,indent=2))
print('actor_updates',b.actor_updates,'attempts',b.actor_attempts,flush=True)
