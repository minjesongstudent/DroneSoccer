"""Single-pass simulation evaluation only. No drone/network control."""
from pathlib import Path
import hashlib
import json
import html
import webbrowser
from datetime import datetime
import torch
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.curriculum import LessonEnv
from src.return_reset_curriculum import wrap

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1)
    path=ROOT/'policy.zip'
    expected=json.loads((ROOT/'manifest.json').read_text())['model_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, 'Model checksum mismatch'
    # CPU inference is portable on Macs; this program never trains a network.
    model=RecurrentPPO.load(path,device='cpu')
    predictor=RecurrentPredictor(model)
    report={'mode':'single-pass simulation only','training':False,'device':'cpu',
            'model_sha256':expected,'suites':{}}
    for name,stage in [('centred',0),('varied',1)]:
        env=wrap(LessonEnv(stage=stage),single=True,metric_ring=True,geometric_return=True)
        records=[]
        try:
            for seed in range(5000000,5000025):
                obs,_=env.reset(seed=seed)
                predictor.reset_episode()
                for step in range(150):
                    with torch.inference_mode():
                        action,_=predictor.predict(obs)
                    obs,_,terminated,truncated,info=env.step(action)
                    if terminated or truncated:
                        break
                records.append({'seed':seed,'outcome':info['outcome'],'passes':info['passes'],'steps':info['steps']})
                print(name,len(records),'/25:',info['outcome'],flush=True)
        finally:
            env.close()
        report['suites'][name]={'successes':sum(r['outcome']=='passed' for r in records),'episodes':25,'records':records}
    output=ROOT/'results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output.mkdir(parents=True)
    (output/'results.json').write_text(json.dumps(report,indent=2))
    parts=['<!doctype html><meta charset="utf-8"><title>Policy check</title><style>body{font:18px system-ui;max-width:850px;margin:40px auto;padding:20px}td,th{padding:8px;text-align:left}table{border-collapse:collapse}tr{border-bottom:1px solid #ddd}</style>',
           '<h1>Single-pass policy check</h1><p>Simulation only. No real drone connected or controlled. No training performed.</p>',
           '<p>Reference Windows/CUDA results: centred 25/25; varied 21/25. Local results may differ across runtimes. These fixed tests are not independent acceptance or proof of real flight.</p>']
    for name,suite in report['suites'].items():
        parts.append(f'<h2>{name}: {suite["successes"]}/25 successful passes</h2><table><tr><th>Seed</th><th>Outcome</th><th>Steps</th></tr>')
        for r in suite['records']:
            parts.append(f'<tr><td>{r["seed"]}</td><td>{html.escape(r["outcome"])}</td><td>{r["steps"]}</td></tr>')
        parts.append('</table>')
    page=output/'report.html'
    page.write_text(''.join(parts),encoding='utf-8')
    print('Saved:',output)
    webbrowser.open(page.as_uri())

if __name__=='__main__':
    main()
