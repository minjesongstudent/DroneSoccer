"""Open the newest nonempty recording; never connects to drone sockets."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
videos = [p for p in (root/'sessions').glob('*/camera.h264') if p.stat().st_size]
if not videos:
    if sys.platform != 'darwin':
        raise SystemExit('No saved video. Use live_preview.py --video /path/to/camera.h264')
    choice = subprocess.run(['osascript','-e',
        'POSIX path of (choose file with prompt "Choose your saved camera.h264 recording")'],
        capture_output=True,text=True)
    if choice.returncode:
        raise SystemExit('No video selected.')
    latest = Path(choice.stdout.strip())
    if not latest.is_file():
        raise SystemExit('Selected video is not a file.')
else:
    latest = max(videos,key=lambda p:p.stat().st_mtime)
raise SystemExit(subprocess.call([sys.executable,str(root/'live_preview.py'),
                                 '--video',str(latest)],cwd=root))
