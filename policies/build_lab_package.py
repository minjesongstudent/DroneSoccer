"""Build the current complete lab folder with portable paths and executable launchers."""
from pathlib import Path
import argparse,hashlib,json,zipfile

ROOT=Path(__file__).resolve().parents[1]
def build(output):
    output=Path(output)
    paths=[ROOT/'LICENSE',ROOT/'policies/registry.json',ROOT/'policies/save_checkpoint.py']
    for folder in ('models','policies/checkpoints','policies/lab-runner'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
                     not any(x in ('__pycache__','checks','sessions','.preview-venv','.pytest_cache') for x in p.parts))
    registry=json.loads((ROOT/'policies/registry.json').read_text())
    for entry in registry['checkpoints']:
        for item in (entry,*entry['shared_models'].values()):
            p=(ROOT/item['path']).resolve()
            if not p.is_relative_to(ROOT) or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Invalid model component: '+item['path'])
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            rel=p.relative_to(ROOT).as_posix();payload=p.read_bytes()
            if p.suffix in ('.py','.command','.sh'):payload=payload.replace(b'\r\n',b'\n')
            info=zipfile.ZipInfo('PC-Policy-Lab-Progress-Search-v3/'+rel);info.create_system=3
            info.external_attr=(0o100755 if p.suffix in ('.command','.sh') else 0o100644)<<16
            z.writestr(info,payload,compress_type=zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(output) as z:
        if z.testzip() is not None:raise ValueError('Corrupt package')
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.sha256').write_bytes((digest+'  '+output.name+'\n').encode())
    print(json.dumps(dict(path=str(output),bytes=output.stat().st_size,sha256=digest)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    build(p.parse_args().output)
