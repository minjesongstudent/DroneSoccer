"""Paired fresh local resets and fixed-input derivative comparison, FIT only."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
R=Path(__file__).parent
RUN=R/'command-regularization-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.metric_policy import MetricRingSAC
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

class Nearby(GuidedLessonEnv):
    def __init__(self,offset):self.offset=offset;super().__init__(stage=4)
    def _sample_layout(self):
        super()._sample_layout()
        dx,dz,dy=self.offset
        self._set_layout(self.pos+dx*self._axis_h+[0,0,dz],self.heading+dy,self.hoop_centre.copy(),self.hoop_normal.copy(),np.linalg.norm(self._hoop_h),np.linalg.norm(self._hoop_v))

def main():
    OUT=RUN/'confirmation';OUT.mkdir(exist_ok=False)
    seeds=json.loads((RUN/'protocol.json').read_text())['fit_seeds']
    rng=np.random.default_rng(941)
    cases=[dict(seed=seed,offset=(rng.uniform(-1,1,3)*[.01,.01,np.pi/180]).tolist()) for seed in seeds for _ in range(4)]
    paths={'source':Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip'),'update20':RUN/'model_20.zip'}
    hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in paths.items()}
    (OUT/'protocol.json').write_text(json.dumps(dict(cases=cases,offset_seed=941,hashes=hashes,
        purpose='36 paired new local offsets around selected FIT seeds, exploratory training-domain confirmation; not independent acceptance.',training_updates=0),indent=2))
    torch.set_num_threads(1);rows=[];derivatives={}
    bank=np.concatenate([np.load(R/'teacher-neighbours-20260928'/f'original_{s}.npz')['observations'][:-1] for s in seeds])
    chosen=np.random.default_rng(942).choice(len(bank),512,replace=False)
    np.save(OUT/'derivative_indices.npy',chosen)
    for name,path in paths.items():
        m=MetricRingSAC.load(path,device='cuda')
        x=torch.tensor(bank[chosen],device='cuda',requires_grad=True)
        action=m.actor(x,deterministic=True)
        jac=torch.stack([torch.autograd.grad(action[:,k].sum(),x,retain_graph=True)[0][:,72:84] for k in range(4)],dim=1)
        derivatives[name]=dict(mean_squared_frobenius=float(jac.square().sum((1,2)).mean()),mean_spectral_norm=float(torch.linalg.matrix_norm(jac,ord=2).mean()))
        with torch.inference_mode():
            for index,case in enumerate(cases):
                e=wrap(Nearby(case['offset']),metric_ring=True,geometric_return=True)
                obs,_=e.reset(seed=case['seed']);front=False
                while True:
                    obs,_,t,tr,info=e.step(m.predict(obs,deterministic=True)[0])
                    front=front or info.get('lap_phase')=='align_front'
                    if t or tr:break
                rows.append(dict(model=name,index=index,seed=case['seed'],steps=info['steps'],passes=info['passes'],outcome=info['outcome'],front=front,full=info['outcome']=='timeout' and info['steps']==600 and info['passes']>=2))
                e.close()
        print(name,derivatives[name],'full',sum(r['full'] for r in rows if r['model']==name),flush=True)
        (OUT/'rows.json').write_text(json.dumps(rows,indent=2))
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[name] for name,p in paths.items())
    summary={name:dict(full=sum(r['full'] for r in rows if r['model']==name),first=sum(r['passes']>0 for r in rows if r['model']==name),front=sum(r['front'] for r in rows if r['model']==name),crashes=sum(r['outcome']!='timeout' for r in rows if r['model']==name)) for name in paths}
    (OUT/'result.json').write_text(json.dumps(dict(summary=summary,derivatives=derivatives,models_unchanged=True,mission_accepted=False),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
