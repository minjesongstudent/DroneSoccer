"""Check loss geometry on preserved FIT flights; no policy inference or env.step."""
from pathlib import Path
import argparse
import json
import shutil
import time
import traceback
import numpy as np
import torch
from src.approach_curriculum import ApproachCurriculumEnv
from src.shadow_dynamics import rollout
from src.shadow_clearance import geometry_from_env,margins,clearance_loss
from tools.run_clearance_experiment import ROOT,sha,single_trainer,write_json

BANK=Path('D:/DroneSoccerData/runs/policy-sampling-comparison-20261008')
MOTION=Path('D:/DroneSoccerData/runs/shadow-dynamics-parity-20261008')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def check():
        if (out/'STOP').exists() or time.monotonic()-started>=600:raise RuntimeError('STOP or total 600-second budget')
    inputs={}
    def bind(path):
        path=Path(path);inputs[str(path)]=sha(path);return path
    try:
        with single_trainer(ROOT.parent/'.clearance-training.lock'):
            torch.set_num_threads(1);torch.set_num_interop_threads(1)
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            torch.cuda.set_per_process_memory_fraction(.15)
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            sources={}
            for path in list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tools/run_clearance_experiment.py',
                    ROOT/'tools/validate_flight_policies.py',ROOT/'tests/test_shadow_clearance.py',
                    ROOT/'experiments/faisal/docs/SHADOW_CLEARANCE_PROTOCOL_20261008.md']:
                check();rel=path.resolve().relative_to(ROOT);dest=out/'source'/rel
                dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest);sources[str(rel)]=sha(dest)
            evidence={k.replace('\\','/'):v for k,v in json.loads(bind(BANK/'evidence.json').read_text()).items()}
            def oldfile(path):
                path=bind(path);assert inputs[str(path)]==evidence[path.relative_to(BANK).as_posix()];return path
            old=json.loads(oldfile(BANK/'protocol.json').read_text())
            for rel in ('src/sim.py','src/params.py','src/contract.py','src/approach_curriculum.py'):
                assert sha(ROOT/rel)==old['sources'][rel]
            records=[json.loads(s) for s in oldfile(BANK/'records.jsonl').read_text().splitlines()]
            if args.smoke:records=records[:6]
            write_json(out/'protocol.json',dict(kind='Training-only geometric margin and delayed action-gradient check',
                smoke=args.smoke,episodes=len(records),sources=sources,buffer_m=.02,buffer_provenance='design only',
                ring_numeric_tolerance_m=1e-10,classification_boundary_band_m=1e-10,
                classification_gate='zero disagreements away from boundary band; report band separately',
                gradient_episodes='first eight deterministic episodes; first two in smoke',
                gradient_gate='finite, nonzero if loss positive; directional abs error<=1e-8 OR relative<=1e-3',
                gradient_direction_seed=11606100,finite_difference_epsilon=1e-5,
                precision='CUDA float64',max_seconds=600,allocator_fraction=.15,
                caveat='Loss surrogate only; not full policy feedback, camera, terminal or passage gradients',
                actor_updated=False,policy_promoted=False,acceptance=False,physical_flight=False))
            write_json(out/'status.json',dict(status='checking_geometry',completed=0,total=len(records)))
            all_values=[];all_truth=[];lengths=[];geometries=[];reports=[]
            for index,row in enumerate(records):
                check();folder=BANK/'trajectories'/row['mode']/str(row['seed'])
                logs=[json.loads(s) for s in oldfile(folder/'episodes.jsonl').read_text().splitlines()]
                assert len(logs)==1
                with np.load(oldfile(folder/logs[0]['trace']),allow_pickle=False) as z:
                    positions=z['positions'].copy();headings=z['headings'].copy()
                assert len(positions)==row['steps']
                env=ApproachCurriculumEnv(level=1,randomise=True,hanger=True)
                try:
                    env.reset(seed=row['seed'])
                    assert np.array_equal(env.pos,row['initial']['position'])
                    assert np.array_equal(env.hoop_centre,row['initial']['centre'])
                    g=geometry_from_env(env,device='cuda');geometries.append(g)
                    with torch.no_grad():values=margins(torch.as_tensor(positions,device='cuda'),
                        torch.as_tensor(headings,device='cuda'),**g).cpu().numpy()
                    truth=np.zeros((len(positions),4))
                    for j,(pos,heading) in enumerate(zip(positions,headings)):
                        if j%100==0:check()
                        # Set saved poses only to query existing geometry. No motion.
                        env.pos=pos.copy();env.heading=float(heading);env.vel=np.zeros(3)
                        truth[j,0]=env._box_distance(env._centreline).min()-env.hoop.thickness/2
                        truth[j,1]=env._hit();truth[j,2]=env._hits_hanger()
                        truth[j,3]=env._touch_walls()[0]
                    all_values.append(values);all_truth.append(truth);lengths.append(len(values))
                    reports.append(dict(seed=row['seed'],mode=row['mode'],steps=len(values),
                        loss=float(np.mean(np.maximum(.02-values.min(1),0)**2)),
                        frames_in_design_buffer=int((values.min(1)<.02).sum())))
                finally:env.close()
                write_json(out/'status.json',dict(status='checking_geometry',completed=index+1,total=len(records)))
            values=np.concatenate(all_values);truth=np.concatenate(all_truth)
            numeric=float(np.max(abs(values[:,0]-truth[:,0])))
            predicted=np.column_stack((values[:,0]<0,values[:,1]<=0,values[:,2]<0))
            labels=truth[:,1:].astype(bool);boundary=abs(values)<=1e-10
            mismatch=predicted!=labels
            away=(mismatch & ~boundary).sum(0);near=(mismatch & boundary).sum(0)
            np.savez_compressed(out/'margin_values.npz',margins=values,source_geometry=truth,
                lengths=np.array(lengths),predicted_collision=predicted,boundary=boundary)
            write_json(out/'episode_losses.json',reports)
            check();write_json(out/'status.json',dict(status='checking_gradient'))
            prior=json.loads(bind(MOTION/'protocol.json').read_text())
            prior_sources={k.replace('\\','/'):v for k,v in prior['sources'].items()}
            if len(prior_sources)!=len(prior['sources']):raise ValueError('Ambiguous prior source path spellings')
            assert sha(ROOT/'src/shadow_dynamics.py')==prior_sources['src/shadow_dynamics.py']
            motion_evidence=json.loads(bind(MOTION/'evidence.json').read_text())
            path=bind(MOTION/'diagnostic_inputs.npz');assert inputs[str(path)]==motion_evidence['diagnostic_inputs.npz']
            ids=[i for i,row in enumerate(records) if row['mode']=='deterministic'][:2 if args.smoke else 8]
            with np.load(path,allow_pickle=False) as z:
                selected={k:z[k][ids].copy() for k in ('actions','dt','lengths','initial_position','initial_heading','half_box','cage','speed','yaw_degrees','tau','delay')}
            tensors={k:torch.as_tensor(v,device='cuda') for k,v in selected.items()}
            direction=np.random.default_rng(11606100).uniform(-.5,.5,selected['actions'].shape)
            valid=np.arange(selected['actions'].shape[1])[None,:]<selected['lengths'][:,None]
            direction*=valid[:,:,None] & (abs(selected['actions'])<.9)
            vector=torch.as_tensor(direction,device='cuda')
            def loss(a):
                kw=dict(tensors);kw['actions']=a;state=rollout(**kw,check=check);total=a.new_zeros(())
                for j,i in enumerate(ids):
                    check();n=int(selected['lengths'][j])
                    total=total+clearance_loss(margins(state['positions'][j,:n],state['headings'][j,:n],**geometries[i]))
                return total/len(ids)
            a=tensors['actions'].detach().clone().requires_grad_();cost=loss(a)
            grad=torch.autograd.grad(cost,a)[0];analytic=float((grad*vector).sum().detach())
            with torch.no_grad():fd=float((loss(a+1e-5*vector)-loss(a-1e-5*vector))/(2e-5))
            ae=abs(analytic-fd);re=ae/max(abs(analytic),abs(fd),1e-12)
            finite=bool(torch.isfinite(grad).all());norm=float(torch.linalg.vector_norm(grad).detach())
            gradient_ok=finite and (float(cost.detach())==0 or norm>0) and (ae<=1e-8 or re<=1e-3)
            np.savez_compressed(out/'loss_gradient.npz',record_indices=ids,action_gradient=grad.cpu().numpy(),direction=direction)
            for rel,digest in sources.items():assert sha(ROOT/rel)==digest and sha(out/'source'/rel)==digest
            for path,digest in inputs.items():assert sha(path)==digest
            write_json(out/'inputs.json',inputs)
            passed=numeric<=1e-10 and not away.any() and gradient_ok
            result=dict(status='complete',smoke=args.smoke,episodes=len(records),frames=len(values),ring_numeric_error_m=numeric,
                mismatches_away_from_boundary=away.tolist(),mismatches_inside_boundary_band=near.tolist(),boundary_frames=boundary.sum(0).tolist(),
                loss=float(cost.detach()),action_gradient_norm=norm,gradient_finite=finite,analytic_directional_derivative=analytic,
                finite_difference_derivative=fd,gradient_absolute_error=ae,gradient_relative_error=re,
                checks_passed=bool(passed),supports_bounded_transfer_test=bool(passed and not args.smoke),
                elapsed_seconds=time.monotonic()-started,actor_updated=False,policy_promoted=False,acceptance=False,physical_flight=False)
            write_json(out/'result.json',result)
            write_json(out/'evidence.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'})
            write_json(out/'status.json',dict(status='complete',checks_passed=bool(passed)))
            print(json.dumps(result),flush=True)
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
