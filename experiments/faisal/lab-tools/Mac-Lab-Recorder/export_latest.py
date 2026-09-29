"""Export the newest session including added measurements and photos."""
from pathlib import Path
import subprocess
import sys
from recorder import export

if __name__ == '__main__':
    root = Path(__file__).resolve().parent / 'sessions'
    sessions = sorted(p.parent for p in root.glob('*/session.json'))
    if not sessions:
        raise SystemExit('No recording sessions found. Run 2-Record.command first.')
    print('Exporting newest session:', sessions[-1])
    archive = export(sessions[-1])
    if sys.platform == 'darwin':
        subprocess.run(['open', '-R', str(archive)], check=False)
