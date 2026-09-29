"""Bounded reset sensitivity audit: all movement from a frozen learned actor."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import torch

ROOT=Path(__file__).parent
SNAPSHOT=ROOT/'lap-recurrent-anchor10-20260928/source_snapshot'
sys.path.insert(0,str(SNAPSHOT))
from src.metric_policy import MetricRingSAC
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

OUT=ROOT/'teacher-neighbours-20260928'
SEEDS=[2820026,2820664,2820750,2820805,2820828,2821928,2821958,2822002,2822042]
CASES={'original':(0,0,0),'left1cm':(.01,0,0),'right1cm':(-.01,0,0),
       'up1cm':(0,.01,0),'down1cm':(0,-.01,0),'yawplus1deg':(0,0,np.pi/180),'yawminus1deg':(0,0,-np.pi/180)}

class NeighbourEnv(GuidedLessonEnv):
    def __init__(self, offset):
        self.offset=offset
        super().__init__(stage=4)
    def _sample_layout(self):
        super()._sample_layout()
        lateral,vertical,yaw=self.offset
        pos=self.pos+self._axis_h*lateral+np.array([0,0,vertical])
        self._set_layout(pos,self.heading+yaw,self.hoop_centre.copy(),self.hoop_normal.copy(),
                         np.linalg.norm(self._hoop_h),np.linalg.norm(self._hoop_v))

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.candidate and not args.output:parser.error('Candidate audit requires separate output')
    if args.output:OUT=args.output
    OUT.mkdir(exist_ok=False)
    path=args.candidate or Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip')
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'protocol.json').write_text(json.dumps(dict(seeds=SEEDS,cases=CASES,source_snapshot=str(SNAPSHOT),
        model=str(path),sha256=before,horizon=600,training_updates=0,
        purpose='Selected FIT reset sensitivity only, not acceptance; each flight uses one frozen learned policy.',
        observation_contract='Offsets applied in reset layout before episode/memory initialization; no truth enters actor; no action perturbations or scripted commands.'),indent=2))
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    if args.candidate:
        from src.geometric_policy import GeometricSAC
        model=GeometricSAC.load(path,device='cuda')
    else:model=MetricRingSAC.load(path,device='cuda')
    rows=[]
    with torch.inference_mode():
        for case,offset in CASES.items():
            for seed in SEEDS:
                assert seed%4!=3
                env=wrap(NeighbourEnv(offset),metric_ring=True,geometric_return=True)
                o,_=env.reset(seed=seed)
                if case=='original' and not args.candidate:
                    with np.load(ROOT/'learned-recovery-20260928'/f'teacher_{seed}.npz') as previous:
                        expected=previous['observations'].copy()
                    np.testing.assert_allclose(o,expected[0],atol=1e-6,rtol=0)
                observations=[];actions=[];positions=[];events=[];old=(0,'approach')
                while True:
                    a=model.predict(o,deterministic=True)[0]
                    observations.append(o.copy());actions.append(a.copy());positions.append(env.unwrapped.pos.copy())
                    o,r,t,tr,info=env.step(a)
                    new=(info['passes'],info['lap_phase'])
                    if new!=old:events.append(dict(step=info['steps'],passes=new[0],phase=new[1]))
                    old=new
                    if t or tr:break
                observations.append(o.copy());positions.append(env.unwrapped.pos.copy())
                if case=='original' and not args.candidate:np.testing.assert_allclose(observations,expected,atol=1e-6,rtol=0)
                np.savez_compressed(OUT/f'{case}_{seed}.npz',observations=observations,actions=actions,positions=positions)
                rows.append(dict(case=case,seed=seed,steps=info['steps'],passes=info['passes'],outcome=info['outcome'],
                    full=info['steps']==600 and info['outcome']=='timeout' and info['passes']>=2,events=events))
                env.close()
            print(case,'full',sum(r['full'] for r in rows if r['case']==case),flush=True)
            (OUT/'rows.json').write_text(json.dumps(rows,indent=2))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    summary={case:dict(full=sum(r['full'] for r in rows if r['case']==case),
        firstpass=sum(r['passes']>0 for r in rows if r['case']==case),
        crashes=sum(r['outcome']!='timeout' for r in rows if r['case']==case)) for case in CASES}
    (OUT/'result.json').write_text(json.dumps(dict(summary=summary,model_unchanged=True,original_trajectories_reproduced=not bool(args.candidate),
        mission_accepted=False,training_updates=0),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
