"""Bounded offline actor regularization on authentic FIT observations."""
from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import torch
ROOT=Path(__file__).parent
OUT=ROOT/'command-regularization-20260928'
SNAP=ROOT/'lap-recurrent-anchor10-20260928/source_snapshot'
sys.path.insert(0,str(OUT/'source_snapshot'))

def main():
    OUT.mkdir(exist_ok=False)
    shutil.copytree(SNAP,OUT/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(__file__,OUT/'runner.py')
    from src.metric_policy import MetricRingSAC
    from src.unified_evaluation import evaluate
    from src.guided_curriculum import GuidedLessonEnv
    from src.return_reset_curriculum import wrap
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.manual_seed(72)
    path=Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    m=MetricRingSAC.load(path,device='cuda')
    seeds=[2820026,2820664,2820750,2820805,2820828,2821928,2821958,2822002,2822042]
    bank=np.concatenate([np.load(ROOT/'teacher-neighbours-20260928'/f'original_{s}.npz')['observations'][:-1] for s in seeds])
    basic_path=Path('D:/DroneSoccerData/runs/recurrent-learned-transfer-20260928/data_600000.npz')
    with np.load(basic_path) as d:basic=d['observations'][d['mask'].astype(bool)]
    lap=torch.as_tensor(bank,device='cuda');basic=torch.as_tensor(basic,device='cuda')
    with torch.no_grad():
        lap_target=m.actor(lap,deterministic=True).detach()
        basic_target=m.actor(basic,deterministic=True).detach()
    optimizer=torch.optim.Adam(m.actor.parameters(),lr=1e-5)
    protocol=dict(source=str(path),source_sha256=before,basic_bank_sha256=hashlib.sha256(basic_path.read_bytes()).hexdigest(),
        fit_seeds=seeds,max_updates=20,checks=[1,5,20],lr=1e-5,seed=72,
        objective='SAC actor deterministic-output MSE to starting actor on128FIT+128basic authentic frames plus0.001 times mean squared command-input Jacobian Frobenius norm on128FIT frames.',
        method='Offline regularization of RL-trained actor, not new reward-driven RL. Actual evaluated movements remain learned actor outputs.',
        retention='Stop centered<22/25 or varied<21/25. No automatic extension.',
        criteria='Full600step missions unchanged; local diagnostic not independent acceptance.',
        observations='Unmodified authentic119D public observations; autograd derivatives only, no fabricated flight observations.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    history=[];reason='bounded_complete'
    for update in range(1,21):
        li=torch.randint(len(lap),(128,),device='cuda');bi=torch.randint(len(basic),(128,),device='cuda')
        x=lap[li].detach().clone().requires_grad_(True)
        action=m.actor(x,deterministic=True)
        grads=[torch.autograd.grad(action[:,axis].sum(),x,create_graph=True,retain_graph=True)[0][:,72:84] for axis in range(4)]
        sensitivity=torch.stack(grads,dim=1).square().sum((1,2)).mean()
        imitation=.5*((action-lap_target[li]).square().mean()+(m.actor(basic[bi],deterministic=True)-basic_target[bi]).square().mean())
        loss=imitation+.001*sensitivity
        optimizer.zero_grad();loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().max()>0 for p in m.actor.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.actor.parameters())
        optimizer.step()
        history.append(dict(update=update,imitation=float(imitation.detach()),sensitivity=float(sensitivity.detach()),loss=float(loss.detach())))
        (OUT/'history.json').write_text(json.dumps(history,indent=2))
        if update not in (1,5,20):continue
        m.offline_command_regularization_updates=update
        m.save(OUT/f'model_{update}.zip')
        with torch.inference_mode():
            report=evaluate(m,4,metric_ring=True,geometric_return=True)
            local=[]
            class Neighbour(GuidedLessonEnv):
                def __init__(self,offset):self.offset=offset;super().__init__(stage=4)
                def _sample_layout(self):
                    super()._sample_layout()
                    dx,dz,dyaw=self.offset
                    self._set_layout(self.pos+dx*self._axis_h+[0,0,dz],self.heading+dyaw,self.hoop_centre.copy(),self.hoop_normal.copy(),np.linalg.norm(self._hoop_h),np.linalg.norm(self._hoop_v))
            for offset in [(0,0,0),(.01,0,0),(-.01,0,0),(0,.01,0),(0,-.01,0),(0,0,np.pi/180),(0,0,-np.pi/180)]:
                for seed in seeds:
                    e=wrap(Neighbour(offset),metric_ring=True,geometric_return=True);o,_=e.reset(seed=seed)
                    while True:
                        o,_,t,tr,info=e.step(m.predict(o,deterministic=True)[0])
                        if t or tr:break
                    local.append(dict(offset=offset,seed=seed,passes=info['passes'],steps=info['steps'],outcome=info['outcome'],full=info['outcome']=='timeout' and info['steps']==600 and info['passes']>=2));e.close()
        (OUT/f'eval_{update}.json').write_text(json.dumps(report,indent=2))
        (OUT/f'local_{update}.json').write_text(json.dumps(local,indent=2))
        counts={k:v['successful_episodes'] for k,v in report['suites'].items()}
        print(update,counts,'local_full',sum(r['full'] for r in local),flush=True)
        if counts['original_centered']<22 or counts['varied_approach']<21:reason='retention_failed';break
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    (OUT/'status.json').write_text(json.dumps(dict(mode='stopped',update=update,reason=reason,source_unchanged=True,mission_accepted=False),indent=2))

if __name__=='__main__':main()
