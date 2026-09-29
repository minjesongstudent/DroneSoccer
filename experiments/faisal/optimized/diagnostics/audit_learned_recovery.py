"""FIT-only two-policy recovery diagnostic, never a single-policy acceptance test."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch

ROOT=Path(__file__).parent
RUN=ROOT/'lap-recurrent-anchor10-20260928'
OUT=ROOT/'learned-recovery-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.metric_policy import MetricRingSAC
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    seeds=[2820026,2820664,2820750,2820805,2820828,2821928,2821958,2822002,2822042]
    student_path=RUN/'model_50.zip'
    teacher_path=Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip')
    hashes={str(p):sha(p) for p in (student_path,teacher_path)}
    modes=['student','teacher','after20','after60','after120']
    protocol=dict(seeds=seeds,modes=modes,hashes=hashes,source_snapshot=str(RUN/'source_snapshot'),
        horizon=600,training_updates=0,acceptance=False,
        purpose='Can the frozen learned teacher recover after an authentic student prefix? Two-policy diagnostic only, not deployable controller or single-policy success.',
        switch_rule='Fixed elapsed steps only; no truth-triggered switch. Same live environment and public observation history throughout. Teacher sees actual student-issued command history.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    student=RecurrentPPO.load(student_path,device='cuda')
    teacher=MetricRingSAC.load(teacher_path,device='cuda')
    rows=[]
    with torch.inference_mode():
        for mode in modes:
            for seed in seeds:
                assert seed%4!=3
                p=RecurrentPredictor(student)
                env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
                o,_=env.reset(seed=seed)
                switch=0 if mode=='teacher' else (int(mode[5:]) if mode.startswith('after') else 601)
                observations=[]; actions=[]; positions=[]; events=[]; previous=(0,'approach')
                while True:
                    step=env.unwrapped.steps
                    a=teacher.predict(o,deterministic=True)[0] if step>=switch else p.predict(o)[0]
                    observations.append(o.copy());actions.append(a.copy());positions.append(env.unwrapped.pos.copy())
                    o,r,t,tr,info=env.step(a)
                    state=(info['passes'],info['lap_phase'])
                    if state!=previous: events.append(dict(step=info['steps'],passes=state[0],phase=state[1]))
                    previous=state
                    if t or tr:break
                observations.append(o.copy());positions.append(env.unwrapped.pos.copy())
                np.savez_compressed(OUT/f'{mode}_{seed}.npz',observations=observations,actions=actions,positions=positions)
                row=dict(mode=mode,seed=seed,steps=info['steps'],passes=info['passes'],outcome=info['outcome'],
                    switched=info['steps']>switch,full=info['steps']==600 and info['outcome']=='timeout' and info['passes']>=2,events=events)
                rows.append(row);env.close()
            selected=[r for r in rows if r['mode']==mode]
            print(mode,'full',sum(r['full'] for r in selected),'firstpass',sum(r['passes']>0 for r in selected),flush=True)
            (OUT/'rows.json').write_text(json.dumps(rows,indent=2))
    assert all(sha(Path(p))==value for p,value in hashes.items())
    summary={mode:dict(full=sum(r['full'] for r in rows if r['mode']==mode),
        first_pass=sum(r['passes']>0 for r in rows if r['mode']==mode),
        outcomes={out:sum(r['outcome']==out for r in rows if r['mode']==mode) for out in sorted(set(r['outcome'] for r in rows))}) for mode in modes}
    (OUT/'result.json').write_text(json.dumps(dict(summary=summary,models_unchanged=True,mission_accepted=False,training_updates=0),indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
