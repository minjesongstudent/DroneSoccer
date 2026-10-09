"""Offline export audit. No hardware or network access."""
import argparse,hashlib,json,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('exports',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
reports=[];traces=[]
for archive in sorted(args.exports.glob('*.zip')):
 with zipfile.ZipFile(archive) as z:
  names={Path(n).name:n for n in z.namelist()}
  get=lambda n:json.loads(z.read(names[n]))
  rows=lambda n:[json.loads(s) for s in z.read(names[n]).splitlines() if s.strip()]
  verified=all(hashlib.sha256(z.read(names[n])).hexdigest()==h for n,h in get('SHA256.json').items())
  if not verified:raise ValueError('Hash failure: '+str(archive))
  session=get('session.json');events=[r for r in rows('startup_search.jsonl') if r['event']=='startup_send_result'];commands=rows('commands.jsonl')
  report=dict(session=archive.stem.split('-export-')[0],hashes_verified=verified,first_stop_reason=session['flight_outcome']['first_stop_reason'],takeoff=any(r.get('command')=='takeoff' for r in commands),policy_rows=len(rows('policy_preview.jsonl')),search_frames=len(events),visible_frames=sum(r['camera_features'][6]==1 for r in events),height_start=events[0]['height_cm'] if events else None,height_end=events[-1]['height_cm'] if events else None,up_commands=sum(r.get('command')=='rc 0 0 15 0' for r in commands))
  reports.append(report)
  if events:
   start=events[0]['analysis_completed_monotonic']
   traces.append(dict(session=report['session'],frames=[dict(t=round(r['analysis_completed_monotonic']-start,6),h=r['height_cm'],yaw=r['yaw'],visible=bool(r['camera_features'][6])) for r in events]))
(args.output/'summary.json').write_bytes((json.dumps(reports,indent=2)+'\n').encode())
(args.output/'recorded-climb-traces.json').write_bytes((json.dumps(traces,indent=2)+'\n').encode())
for r in reports:print(json.dumps(r))
