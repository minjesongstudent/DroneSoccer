"""CUDA motion parity on saved learned commands; no new flight or actor inference."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
from src.approach_curriculum import ApproachCurriculumEnv
from src.clearance_training import ClearanceTrainingEnv
from src.shadow_dynamics import rollout
from tools.run_clearance_experiment import ROOT,sha,single_trainer,write_json

BANK=Path('D:/DroneSoccerData/runs/policy-sampling-comparison-20261008')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    def check():
        if (out/'STOP').exists() or time.monotonic()-started>=600:raise RuntimeError('STOP or total budget reached')
    try:
        with single_trainer(ROOT.parent/'.clearance-training.lock'):
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.cuda.set_per_process_memory_fraction(.15)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            sources={}
            files=list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tools/run_clearance_experiment.py',
                ROOT/'tools/validate_flight_policies.py',ROOT/'tests/test_shadow_dynamics.py',
                ROOT/'experiments/faisal/docs/SHADOW_DYNAMICS_PROTOCOL_20261008.md']
            for path in files:
                check();rel=path.resolve().relative_to(ROOT);dest=out/'source'/rel
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);sources[str(rel)]=sha(dest)
            inputs={}
            def bind(p):
                p=Path(p);inputs[str(p)]=sha(p);return p
            raw_evidence=json.loads(bind(BANK/'evidence.json').read_text())
            evidence={name.replace('\\','/'):digest for name,digest in raw_evidence.items()}
            if len(evidence)!=len(raw_evidence):raise ValueError('Ambiguous evidence path spellings')
            def read(p):
                p=bind(p);relative=p.relative_to(BANK).as_posix()
                if relative in evidence and inputs[str(p)]!=evidence[relative]:raise ValueError('Saved input changed')
                return json.loads(p.read_text())
            old=read(BANK/'protocol.json')
            for rel in ('src/sim.py','src/params.py','src/contract.py','src/approach_curriculum.py','src/clearance_training.py','src/camera_history.py'):
                if sha(ROOT/rel)!=old['sources'][rel]:raise ValueError('Physics/reset/history differs from saved flights')
            records_path=bind(BANK/'records.jsonl')
            assert inputs[str(records_path)]==evidence['records.jsonl']
            records=[json.loads(s) for s in records_path.read_text().splitlines()]
            if args.smoke:records=records[:6]
            write_json(out/'protocol.json',dict(kind='Differentiable shadow motion parity; saved commands only',
                smoke=args.smoke,episodes=len(records),source_bank=str(BANK),sources=sources,
                tolerance_position_m=1e-8,tolerance_heading_rad=1e-8,wall_counts_exact=True,
                gradient_gate='directional absolute error <=1e-7 OR relative error <=1e-3',
                direction_seed=11606000,finite_difference_epsilon=1e-5,precision='CUDA float64',
                max_seconds=600,allocator_fraction=.15,actor_updated=False,acceptance=False,physical_flight=False,
                limitation='Fixed saved commands and dt; not gradients of camera, collision decisions or closed-loop policy'))
            write_json(out/'status.json',dict(status='preparing',episodes=len(records)))
            n=len(records);steps=max(r['steps'] for r in records)
            arrays=dict(actions=np.zeros((n,steps,4)),dt=np.full((n,steps),.1),lengths=np.zeros(n,np.int64),
                initial_position=np.zeros((n,3)),initial_heading=np.zeros(n),half_box=np.zeros((n,3)),
                cage=np.zeros((n,4)),speed=np.zeros(n),yaw_degrees=np.zeros(n),tau=np.zeros(n),delay=np.zeros(n,np.int64))
            expected_p=np.zeros((n,steps,3));expected_h=np.zeros((n,steps));valid=np.zeros((n,steps),bool)
            for i,row in enumerate(records):
                check();folder=BANK/'trajectories'/row['mode']/str(row['seed'])
                path=bind(folder/'episodes.jsonl')
                assert inputs[str(path)]==evidence[path.relative_to(BANK).as_posix()]
                logs=[json.loads(s) for s in path.read_text().splitlines()];assert len(logs)==1
                path=bind(folder/logs[0]['trace'])
                assert inputs[str(path)]==evidence[path.relative_to(BANK).as_posix()]
                with np.load(path,allow_pickle=False) as z:
                    size=row['steps'];arrays['actions'][i,:size]=z['actions']
                    expected_p[i,:size]=z['positions'];expected_h[i,:size]=z['headings'];valid[i,:size]=True
                    first_obs=z['actor_observations'][0].copy()
                # Reset only to recover the exact recorded constants and tick RNG.
                # No env.step, model load, current-action selection or new rollout.
                env=ClearanceTrainingEnv(ApproachCurriculumEnv(level=1,randomise=True,hanger=True),'clearance')
                try:
                    public,_=env.reset(seed=row['seed']);e=env.unwrapped
                    assert np.array_equal(public,first_obs)
                    assert np.array_equal(e.pos,row['initial']['position']) and e.heading==row['initial']['heading']
                    assert asdict(e.dyn)==row['initial']['dynamics']
                    assert np.array_equal(e.hoop_centre,row['initial']['centre']) and np.array_equal(e.hoop_normal,row['initial']['normal'])
                    assert not e.vel.any() and e.yaw_rate==0
                    arrays['dt'][i,:size]=1/e.np_random.uniform(*e.ranges.rate_hz,size=size)
                    arrays['lengths'][i]=size;arrays['initial_position'][i]=e.pos;arrays['initial_heading'][i]=e.heading
                    arrays['half_box'][i]=e._half_box;arrays['cage'][i]=[e.cage.length,e.cage.width,e.cage.height,e.cage.min_altitude]
                    arrays['speed'][i]=e.dyn.max_speed;arrays['yaw_degrees'][i]=e.dyn.max_yaw_rate_deg
                    arrays['tau'][i]=e.dyn.tau;arrays['delay'][i]=e.dyn.delay_steps
                finally:env.close()
            write_json(out/'inputs.json',inputs)
            np.savez_compressed(out/'diagnostic_inputs.npz',**arrays,valid=valid,expected_positions=expected_p,expected_headings=expected_h)
            tensors={k:torch.as_tensor(v,device='cuda') for k,v in arrays.items()}
            write_json(out/'status.json',dict(status='checking',episodes=n))
            with torch.no_grad():prediction=rollout(**tensors,check=check)
            predicted={k:v.cpu().numpy() for k,v in prediction.items()}
            pe=float(np.max(np.abs(predicted['positions'][valid]-expected_p[valid])))
            angle=(predicted['headings'][valid]-expected_h[valid]+np.pi)%(2*np.pi)-np.pi
            he=float(np.max(abs(angle)))
            walls=predicted['wall_contacts'].sum(1)
            wall_exact=bool(np.array_equal(walls,[r['wall_steps'] for r in records]))
            # Numerical derivative only, on the shadow arithmetic. This does not
            # move a simulator or claim correctness of closed-loop policy gradients.
            direction=np.random.default_rng(11606000).uniform(-.5,.5,arrays['actions'].shape)
            direction*=valid[:,:,None] & (abs(arrays['actions'])<.9)
            vector=torch.as_tensor(direction,device='cuda')
            mask=torch.as_tensor(valid,device='cuda')
            def loss(x):
                kw=dict(tensors);kw['actions']=x;z=rollout(**kw,check=check)
                return z['positions'][mask].square().mean()+torch.sin(z['headings'][mask]).mean()
            a=tensors['actions'].detach().clone().requires_grad_()
            gradient=torch.autograd.grad(loss(a),a)[0]
            analytic=float((gradient*vector).sum().detach())
            with torch.no_grad():numerical=float((loss(a+1e-5*vector)-loss(a-1e-5*vector))/(2e-5))
            ae=abs(analytic-numerical);re=ae/max(abs(analytic),abs(numerical),1e-12)
            gp=ae<=1e-7 or re<=1e-3
            np.savez_compressed(out/'shadow_outputs.npz',**predicted,action_gradient=gradient.cpu().numpy(),direction=direction)
            for rel,digest in sources.items():assert sha(ROOT/rel)==digest and sha(out/'source'/rel)==digest
            for path,digest in inputs.items():assert sha(path)==digest
            result=dict(status='complete',smoke=args.smoke,episodes=n,frames=int(valid.sum()),max_position_error_m=pe,
                max_heading_error_rad=he,wall_counts_exact=wall_exact,analytic_directional_derivative=analytic,
                finite_difference_derivative=numerical,gradient_absolute_error=ae,gradient_relative_error=re,
                arithmetic_checks_passed=pe<=1e-8 and he<=1e-8 and wall_exact and gp,
                supports_objective_design=not args.smoke and pe<=1e-8 and he<=1e-8 and wall_exact and gp,
                elapsed_seconds=time.monotonic()-started,actor_updated=False,policy_promoted=False,acceptance=False,physical_flight=False)
            write_json(out/'result.json',result)
            write_json(out/'evidence.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'})
            write_json(out/'status.json',dict(status='complete',arithmetic_checks_passed=result['arithmetic_checks_passed']))
            print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
