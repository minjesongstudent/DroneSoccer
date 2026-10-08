"""Re-score identical recorded state/actions; never generate additional flights."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import SAC
from tools.run_clearance_experiment import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    assert (args.run/'DONE').exists()
    original=SAC.load(args.run/'model.zip',device='cpu');candidate=SAC.load(args.model,device='cpu')
    for module in ['actor','critic_target']:
        for name,value in getattr(original,module).state_dict().items():
            assert torch.equal(value,getattr(candidate,module).state_dict()[name]),(module,name)
    assert torch.equal(original.log_ent_coef,candidate.log_ent_coef)
    assert original.gamma==candidate.gamma
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.model,args.out/'model.zip');shutil.copy2(__file__,args.out/'rescore_source.py')
    records=copy.deepcopy(json.loads((args.run/'results.json').read_text()))
    rows=[json.loads(line) for line in (args.run/'continuations.jsonl').read_text().splitlines()]
    hashes={str(args.run/name):sha(args.run/name) for name in ['protocol.json','results.json','continuations.jsonl','model.zip']}
    values={}
    for state in records:
        for action in state['actions']:
            path=args.run/f"state_{state['state']:03d}"/f"action_{action['candidate']}_replica_0.npz"
            hashes[str(path)]=sha(path)
            with np.load(path) as trace:
                obs=torch.tensor(trace['observations'][0:1],dtype=torch.float32)
                act=torch.tensor(trace['actions'][0:1],dtype=torch.float32)
            with torch.no_grad():
                old=float(torch.cat(original.critic(obs,act),1).min())
                new=float(torch.cat(candidate.critic(obs,act),1).min())
            np.testing.assert_allclose(old,action['q'],rtol=1e-5,atol=1e-5)
            action['q']=new;values[(state['state'],action['candidate'])]=new
    for row in rows:row['q']=values[(row['state'],row['candidate'])]
    write_json(args.out/'protocol.json',dict(kind='Offline critic re-score; zero additional flights',
        input_hashes=hashes,model_sha256=sha(args.model),actor_and_target_exact=True,
        original_run=str(args.run.resolve()),rollouts=0,actor_updates=0,acceptance=False))
    write_json(args.out/'results.json',records)
    (args.out/'continuations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (args.out/'DONE').write_text('Offline re-score complete. Outcomes reused unchanged; no additional flights.\n')


if __name__=='__main__':main()
