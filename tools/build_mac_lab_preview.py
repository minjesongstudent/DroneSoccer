"""Build the Mac camera recorder from branch source; excludes all user sessions."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'experiments/faisal/lab-tools/Mac-Lab-Recorder'
OUT=ROOT.parent/'Mac-Lab-Recorder-Live'


def main():
    files={}
    for name in ['recorder.py','test_recorder.py','live_preview.py','test_live_preview.py',
                 'export_latest.py','replay_latest.py','mac-common.sh','MEASUREMENTS.md',
                 '3-Export-Latest.command','preview-requirements.txt']:
        files[name]=(SOURCE/name).read_bytes()
    files['1-Setup.command']=(SOURCE/'4-Setup-Camera-Window.command').read_text().replace(
        '5-Record-With-Camera.command','2-Record.command').encode()
    files['2-Record.command']=(SOURCE/'5-Record-With-Camera.command').read_text().replace(
        '4-Setup-Camera-Window.command','1-Setup.command').encode()
    files['4-Replay-Latest.command']=(SOURCE/'6-Replay-Latest-Camera.command').read_text().replace(
        '4-Setup-Camera-Window.command','1-Setup.command').encode()
    for name in ['__init__.py','contract.py','params.py','detect.py','camera_history.py']:
        files['src/'+name]=(ROOT/'src'/name).read_bytes()
    files['preview-policy.zip']=(ROOT/'trainedModels/lags60-scratch.zip').read_bytes()
    files['preview-manifest.json']=json.dumps(dict(
        model_sha256=hashlib.sha256(files['preview-policy.zip']).hexdigest(),
        model='lags60-scratch',group_master='71c577e',policy_inputs=109,flight_control=False),indent=2).encode()
    files['START-HERE.txt']=(SOURCE/'LIVE-START-HERE.txt').read_bytes()
    files['LICENSE']=(ROOT/'LICENSE').read_bytes()
    hashes={}
    for name,data in files.items():
        if name.endswith(('.command','.sh')):
            data=data.replace(b'\r\n',b'\n');files[name]=data
        path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    files['PACKAGE-SHA256.json']=json.dumps(hashes,indent=2).encode()
    (OUT/'PACKAGE-SHA256.json').write_bytes(files['PACKAGE-SHA256.json'])
    archive=OUT.with_suffix('.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name,data in files.items():
            entry=ZipInfo(OUT.name+'/'+name);entry.create_system=3
            entry.external_attr=(0o100755 if name.endswith(('.command','.sh')) else 0o100644)<<16
            entry.compress_type=ZIP_DEFLATED;z.writestr(entry,data)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name,digest in hashes.items():
            assert hashlib.sha256(z.read(OUT.name+'/'+name)).hexdigest()==digest
    print('Verified',archive,len(files),'files',archive.stat().st_size,'bytes')

if __name__=='__main__':main()
