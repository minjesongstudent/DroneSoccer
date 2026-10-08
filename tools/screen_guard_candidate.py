"""Apply the recorded one-shot guard-reward candidate decision."""
import json
from math import comb
from pathlib import Path
import shutil
from tools.run_clearance_experiment import sha,write_json


def main():
    run=Path('D:/DroneSoccerData/runs/guard-reward-fresh-20261007')
    assert (run/'DONE').exists()
    results=json.loads((run/'results.json').read_text())
    reasons=[];hashes={str(run/'results.json'):sha(run/'results.json')}
    for suite,record in results.items():
        for model in ['original','guard_reward']:
            log=run/model/suite/'episodes.jsonl';hashes[str(log)]=sha(log)
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            assert len(rows)==200 and len({r['seed'] for r in rows})==200
            for field in ['clean_first_pass','clean_repeated_episode','crashed']:
                assert sum(r[field] for r in rows)==record['counts'][model][field]
        a,b=record['counts']['original'],record['counts']['guard_reward']
        for field in ['clean_first_pass','clean_repeated_episode']:
            if a[field]-b[field]>2:reasons.append(suite+'/'+field+' retention loss')
        if b['crashed']-a['crashed']>2:reasons.append(suite+'/crash retention loss')
    pair=results['randomised']['paired']['clean_repeated_episode']
    gains,losses=pair['candidate_only'],pair['original_only'];n=gains+losses
    p=sum(comb(n,k) for k in range(gains,n+1))/2**n if n else 1.
    if gains-losses<5:reasons.append('Randomized net repeat gain below five episodes')
    if p>.05:reasons.append('Paired one-sided binomial screen above 0.05')
    output=run/'decision.json'
    assert not output.exists(), 'Preserve existing decision'
    write_json(output,dict(decision='reject' if reasons else 'retain_for_further_validation',
        reasons=reasons,gains=gains,losses=losses,one_sided_p=p,
        original_retained=True,acceptance=False,physical_flight=False,input_hashes=hashes,
        caveat='Prespecified development screen, not acceptance or multiplicity-adjusted proof.'))
    shutil.copy2(__file__,run/'decision_source.py')
    print(output.read_text())


if __name__=='__main__':main()
