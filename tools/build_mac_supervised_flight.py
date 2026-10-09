"""Build a separately named physical-flight package; does not contact hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.build_mac_lab_preview import SOURCE, build_package


def build(out):
    # Preserve the shadow build as a distinct package; no old download is overwritten.
    shadow=out.with_name(out.name+'-shadow-source')
    build_package('seenflag-s1-level1',shadow)
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError(out)
    files={p.relative_to(shadow).as_posix():p.read_bytes() for p in shadow.rglob('*') if p.is_file()}
    for name in ('flight_control.py','supervised_flight.py','test_flight_control.py','test_supervised_flight.py'):
        files[name]=(SOURCE/name).read_bytes()
    files['2-Motors-Off-Record.command']=files.pop('2-Record.command')
    files['2-Supervised-Flight.command']=b'''#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
echo 'REAL FLIGHT TEST: sends takeoff, policy RC, and landing commands.'
echo 'Use the enclosed lab cage, guards fitted, and a supervising operator.'
echo 'The UI requires a held control. Release or Escape requests landing.'
echo 'Do not launch another drone controller/recorder at the same time.'
if [ ! -x .preview-venv/bin/python ]; then
  echo 'Run 1-Setup.command on normal internet first.'
else
  printf 'Type FLIGHT to connect and open the controls (no automatic takeoff): '
  read -r reply
  if [ "$reply" = FLIGHT ]; then
    .preview-venv/bin/python supervised_flight.py --physical-flight
  fi
fi
pause_window
'''
    files['1-Setup.command']=files['1-Setup.command'].replace(b'test_compare_session.py',b'test_compare_session.py test_flight_control.py test_supervised_flight.py').replace(b'2-Record.command',b'2-Supervised-Flight.command')
    manifest=json.loads(files['preview-manifest.json']);manifest.update(flight_control=True,
        mode='operator-started experimental physical flight',validated_for_physical_flight=False)
    files['preview-manifest.json']=json.dumps(manifest,indent=2).encode()
    files['START-HERE.txt']=(SOURCE/'SUPERVISED-FLIGHT.txt').read_bytes()
    files.pop('PACKAGE-SHA256.json')
    out.mkdir(parents=True)
    for name,data in files.items():
        if name.endswith(('.command','.sh')):data=data.replace(b'\r\n',b'\n');files[name]=data
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    files['PACKAGE-SHA256.json']=json.dumps(hashes,indent=2).encode()
    (out/'PACKAGE-SHA256.json').write_bytes(files['PACKAGE-SHA256.json'])
    with ZipFile(out.with_suffix('.zip'),'x',ZIP_DEFLATED) as z:
        for name,data in files.items():
            entry=ZipInfo(out.name+'/'+name);entry.create_system=3
            entry.external_attr=(0o100755 if name.endswith(('.command','.sh')) else 0o100644)<<16
            entry.compress_type=ZIP_DEFLATED;z.writestr(entry,data)
    with ZipFile(out.with_suffix('.zip')) as z:
        assert z.testzip() is None
        for name,sha in hashes.items():assert hashlib.sha256(z.read(out.name+'/'+name)).hexdigest()==sha
    print('Physical test package built:',out.with_suffix('.zip'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    build(p.parse_args().out.resolve())
