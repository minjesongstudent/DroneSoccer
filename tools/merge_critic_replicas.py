"""Merge the fixed disjoint replica batches; preserve every original run."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from tools.run_clearance_experiment import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--extra',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    rows=[];hashes={};seen=set()
    for root,expected in [(args.base,set(range(6))),(args.extra,set(range(6,24)))]:
        assert (root/'DONE').exists()
        batch=[json.loads(line) for line in (root/'continuations.jsonl').read_text().splitlines()]
        assert len(batch)==12*4*len(expected)
        for state in range(12):
            for candidate in range(4):
                assert {r['replica'] for r in batch if r['state']==state and r['candidate']==candidate}==expected
        for row in batch:
            key=(row['state'],row['candidate'],row['replica'])
            assert key not in seen;seen.add(key)
        rows.extend(batch)
        for name in ['model.zip','protocol.json','continuations.jsonl']:
            hashes[str(root/name)]=sha(root/name)
    assert sha(args.base/'model.zip')==sha(args.extra/'model.zip')
    args.out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.base/'model.zip',args.out/'model.zip')
    records=json.loads((args.base/'results.json').read_text())
    for state in records:
        for action in state['actions']:
            selected=[r for r in rows if r['state']==state['state'] and r['candidate']==action['candidate']]
            assert len(selected)==24
            np.testing.assert_allclose([r['q'] for r in selected],action['q'],rtol=1e-6)
            for target,field in [('finite_mean','finite_soft_return'),('bootstrapped_mean','bootstrapped_soft_return'),('mission_rate','mission')]:
                action[target]=float(np.mean([r[field] for r in selected]))
            rel=Path(f"state_{state['state']:03d}")/f"action_{action['candidate']}_replica_0.npz"
            dest=args.out/rel;dest.parent.mkdir(exist_ok=True)
            shutil.copy2(args.base/rel,dest);hashes[str(args.base/rel)]=sha(dest)
    write_json(args.out/'results.json',records)
    rows.sort(key=lambda r:(r['state'],r['candidate'],r['replica']))
    (args.out/'continuations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write_json(args.out/'protocol.json',dict(kind='Offline merge of fixed24 replicas, zero new flights',
        states=12,candidates=4,replicas=24,unique_continuations=1152,input_hashes=hashes,acceptance=False,
        trace_locations=[str(args.base.resolve()),str(args.extra.resolve())]))
    shutil.copy2(__file__,args.out/'merge_source.py')
    (args.out/'DONE').write_text('All1152 unique continuations combined. No additional simulation.\n')


if __name__=='__main__':main()
