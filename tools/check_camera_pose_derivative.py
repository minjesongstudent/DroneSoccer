"""Bounded source-camera derivative check on saved FIT poses; no actor/run steps."""
from pathlib import Path
import argparse
import json
import shutil
import time
import traceback
import numpy as np

from src.approach_curriculum import ApproachCurriculumEnv
from src.camera_pose_derivative import consistency
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer

BANK=Path('D:/DroneSoccerData/runs/policy-sampling-comparison-20261008')
PLAN='experiments/faisal/docs/CAMERA_DERIVATIVE_PROTOCOL_20261008.md'


def decide(rows):
    visible=[r for r in rows if r['visible']];stable=[r for r in visible if r['stable']]
    reasons=[]
    if len(stable)<100:reasons.append('Fewer than100 stable visible poses')
    boundary=sum(not r['stable'] for r in visible)/max(len(visible),1)
    agreement=sum(r['agrees'] for r in stable)/max(len(stable),1)
    if boundary>.10:reasons.append('More than10% visible poses cross a branch')
    if agreement<.95:reasons.append('Fewer than95% stable visible derivatives agree')
    invisible=[r for r in rows if not r['visible'] and r['stable']]
    if not all(r['zero_invisible'] for r in invisible):reasons.append('Stable invisible derivative is nonzero')
    return dict(poses=len(rows),visible=len(visible),stable_visible=len(stable),
        visible_branch_fraction=boundary,stable_visible_agreement=agreement,
        stable_invisible=len(invisible),supports_local_feedback_derivative=not reasons,reasons=reasons,
        policy_promoted=False,actor_updated=False,acceptance=False,physical_flight=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();inputs={};rows=[]
    def check():
        if (out/'STOP').exists() or time.monotonic()-started>=600:raise RuntimeError('STOP or600second total cap')
    def bind(path,expected=None):
        path=Path(path);digest=sha(path)
        if expected is not None and digest!=expected:raise ValueError('Input changed: '+str(path))
        inputs[str(path)]=digest;return path
    try:
        with single_trainer(ROOT.parent/'.clearance-training.lock'):
            old_evidence=json.loads(bind(BANK/'evidence.json').read_text())
            evidence={k.replace('\\','/'):v for k,v in old_evidence.items()}
            if len(evidence)!=len(old_evidence):raise ValueError('Ambiguous evidence paths')
            def old(path):return bind(path,evidence[Path(path).relative_to(BANK).as_posix()])
            records=[r for r in map(json.loads,old(BANK/'records.jsonl').read_text().splitlines()) if r['mode']=='deterministic']
            if len(records)!=64:raise ValueError('Expected the64 unchanged source FIT episodes')
            prior=json.loads(old(BANK/'protocol.json').read_text())
            sources={}
            for path in list((ROOT/'src').glob('*.py'))+[Path(__file__).resolve(),ROOT/PLAN,
                    ROOT/'tools/run_clearance_experiment.py',ROOT/'tests/test_camera_pose_derivative.py']:
                relative=path.relative_to(ROOT);dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,dest);sources[relative.as_posix()]=sha(dest)
            for key in ('src/sim.py','src/contract.py','src/params.py','src/approach_curriculum.py'):
                if sha(ROOT/key)!=prior['sources'][key]:raise ValueError('Saved flight camera/reset changed')
            write_json(out/'protocol.json',dict(kind='Local source-camera pose derivative stability',sources=sources,
                sample='all64 deterministic source FIT flights; post-step indices0,20,40,... plus final unique index',
                epsilon=.001,half_epsilon=.0005,direction_seed=11608000,max_seconds=600,
                gate='stable visible>=100; visible branch fraction<=.10; >=.95 stable visible agree; stable invisible zero',
                agree='both Jacobian and directional L2 error <=.002 OR relative<=.05',
                failure='Close numerical route at these scales; no automatic stencil sweep or training',
                success='Supports local camera-feedback gradient integration only, not actor improvement',
                no_env_steps=True,no_actor_inference=True,no_training=True,no_acceptance=True))
            rng=np.random.default_rng(11608000);arrays=[]
            for index,record in enumerate(records):
                check();folder=BANK/'trajectories/deterministic'/str(record['seed'])
                logs=[json.loads(s) for s in old(folder/'episodes.jsonl').read_text().splitlines()]
                if len(logs)!=1:raise ValueError('Expected one complete trace')
                with np.load(old(folder/logs[0]['trace']),allow_pickle=False) as z:
                    pos=z['positions'];head=z['headings'];observed=z['next_actor_observations'][:,64:72]
                    indices=sorted(set(range(0,len(pos),20))|{len(pos)-1})
                    e=ApproachCurriculumEnv(level=1,randomise=True,hanger=True)
                    try:
                        e.reset(seed=record['seed'])
                        if not np.array_equal(e.hoop_centre,record['initial']['centre']):raise ValueError('Scene changed')
                        for t in indices:
                            check();direction=rng.normal(size=4);direction/=np.linalg.norm(direction)
                            r=consistency(e,np.r_[pos[t],head[t]],direction,check)
                            if not np.array_equal(r['observation'],observed[t]):raise ValueError('Source camera forward mismatch')
                            arrays.append(dict(pose=np.r_[pos[t],head[t]],direction=direction,
                                observation=r['observation'],jacobian=r['jacobian'],half_jacobian=r['half_jacobian'],directional=r['directional']))
                            row={k:v for k,v in r.items() if k not in arrays[-1]}
                            row.update(seed=record['seed'],frame=t,
                                zero_invisible=bool(not r['jacobian'].any() and not r['half_jacobian'].any() and not r['directional'].any()))
                            rows.append(row)
                    finally:e.close()
                write_json(out/'status.json',dict(status='running',completed_episodes=index+1,total_episodes=64,poses=len(rows)))
            result=decide(rows);result.update(status='complete',elapsed_seconds=time.monotonic()-started)
            np.savez_compressed(out/'derivatives.npz',**{k:np.array([r[k] for r in arrays]) for k in arrays[0]})
            write_json(out/'pose_results.json',rows);write_json(out/'inputs.json',inputs)
            for rel,digest in sources.items():
                if sha(ROOT/rel)!=digest:raise ValueError('Source changed during check')
            write_json(out/'result.json',result)
            write_json(out/'evidence.json',{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='status.json'})
            write_json(out/'status.json',dict(status='complete',supports_local_feedback_derivative=result['supports_local_feedback_derivative']))
            print(json.dumps(result))
    except Exception as exc:
        write_json(out/'status.json',dict(status='failed',error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc());raise


if __name__=='__main__':main()
