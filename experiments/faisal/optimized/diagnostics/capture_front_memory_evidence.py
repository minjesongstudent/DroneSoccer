"""Capture genuine public front-memory histories; no learning or demonstrations."""
import sys, json, hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from src.anchored_policy import AnchoredSAC
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation
from src.front_memory import FrontMemoryObservation, VERSION
from src.return_practice import _code_hashes

OUT=Path('D:/DroneSoccerData/diagnostics/front-memory-capture-v1.json')
assert not OUT.exists(), 'Preserve earlier result'
checkpoint=ROOT/'runs/return-route-40k-20260927/best_model.zip'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before=digest(checkpoint)
manifest=json.loads((ROOT/'return-libraries/train-v1/manifest.json').read_text())
assert manifest['code_sha256']==_code_hashes() and manifest['source_sha256']==before
assert torch.cuda.is_available()
torch.set_num_threads(1)
model=AnchoredSAC.load(checkpoint,device='cuda')
env=FrontMemoryObservation(MissionObservation(GuidedLessonEnv(stage=4),single_pass=False))
records=[]
with torch.inference_mode():
    for seed in range(2000000,2000016):
        observation,_=env.reset(seed=seed)
        seen=set()
        front_seen_steps=0
        for step in range(600):
            front_seen_steps+=int(observation[70]>.5 and observation[71]>.5)
            # Added features cannot change the100D source policy's actions.
            action,_=model.predict(observation[:100],deterministic=True)
            observation,_,terminated,truncated,info=env.step(action)
            phase=env.unwrapped._lap_phase
            if phase in ('return_side','return_front') and phase not in seen:
                seen.add(phase)
                records.append(dict(seed=seed,step=step+1,phase=phase,
                    genuine_prior_passes=int(env.unwrapped.passes),
                    remaining_steps=600-int(env.unwrapped.steps),
                    public_front_sightings=front_seen_steps,
                    original_memory=observation[84:99].tolist(),
                    appended_memory=observation[100:].tolist()))
            if terminated or truncated: break
        print(json.dumps({'seed':seed,'steps':step+1,'captured':len(records)}),flush=True)
env.close()
assert before==digest(checkpoint) and manifest['code_sha256']==_code_hashes()
summary={}
for phase in ('return_side','return_front'):
    rows=[r for r in records if r['phase']==phase and r['remaining_steps']>=200]
    summary[phase]=dict(authentic_eligible_entries=len(rows),
        with_genuine_front_history=sum(r['appended_memory'][8]>.5 for r in rows),
        mean_front_age_seconds=float(np.mean([r['appended_memory'][9]*60 for r in rows])) if rows else None,
        mean_any_detection_age_seconds=float(np.mean([r['original_memory'][9]*60 for r in rows])) if rows else None)
result=dict(version=VERSION,checkpoint_sha256=before,seed_range=[2000000,2000015],
    source_actor_inputs=100,candidate_observation_inputs=115,training_updates=0,
    actions='unchanged saved learned policy using original100D prefix',
    memory_inputs='camera observations and issued commands with fixed nominal dynamics only',
    independent_acceptance_used=False,records=records,summary=summary,
    limitations='Availability probe only; no new policy learns or succeeds here. Not a usable training bank.')
OUT.write_text(json.dumps(result,indent=2))
print(json.dumps(summary))
