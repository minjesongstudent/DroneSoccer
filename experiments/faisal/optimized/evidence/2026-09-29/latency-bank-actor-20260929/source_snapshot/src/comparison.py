"""Read-only A/B/C comparison using identical saved checkpoint protocols."""
import json
from pathlib import Path
from urllib.request import urlopen

WORKSPACE = Path(__file__).resolve().parents[2]
BASELINE_RUN = WORKSPACE / 'DroneSoccer' / 'runs' / 'local-gpu-staged-20260926-01'
CORRECTED_RUN = WORKSPACE / 'DroneSoccer-Corrected' / 'runs' / 'corrected-reward-20260926-01'
BASELINE_URL = 'http://127.0.0.1:8771'
CORRECTED_URL = 'http://127.0.0.1:8772'
GUIDED_URL = 'http://127.0.0.1:8773'
SEED_START = 3_000_000


def _read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def _checkpoints(folder):
    return {int(p.stem): p for p in (folder/'evaluations').glob('*.json') if p.stem.isdigit()}


def _status(url):
    try:
        with urlopen(url + '/state', timeout=1) as response:
            data = json.load(response)
        return {key: data.get(key) for key in ('mode', 'steps', 'target', 'lesson', 'total_passes')}
    except Exception:
        return {'mode': 'Viewer unavailable'}


def _metric(report, suite):
    return {key: report['suites'][suite][key]
            for key in ('success_rate', 'total_passes', 'crash_rate')}


def _protocol_matches(legacy, guided, suite, config):
    """Legacy reports predate metadata; validate their recorded setup.

    A/B have identical layout, dynamics and success definitions for these suites.
    Their reward changes do not affect deterministic policy evaluation. Their
    current lesson is the fixed centered benchmark only at stage zero.
    """
    source = 'lesson' if suite == 'original_centered' else suite
    if source not in legacy.get('suites', {}) or suite not in guided.get('suites', {}):
        return False
    if suite == 'original_centered' and legacy.get('stage') != 0:
        return False
    protocol = guided.get('protocols', {}).get(suite, {})
    expected_id = 'centered-clean-pass-v1' if suite == 'original_centered' else suite + '-v1'
    expected_horizon = 150 if suite == 'original_centered' else 600
    expected_success = 'passed' if suite == 'original_centered' else 'timeout and passes >= 2'
    if (protocol.get('id') != expected_id or not protocol.get('deterministic')
            or protocol.get('max_steps') != expected_horizon
            or protocol.get('success') != expected_success):
        return False
    geometry = config.get('geometry') if config else None
    if geometry is None:
        return False
    geometry = {key: dict(value) for key, value in geometry.items()}
    if suite == 'full_task_delay_200ms':
        geometry['drone']['delay_steps'] = 2
    return protocol.get('geometry') == geometry


def _target_progress(files):
    qualifying = []
    latest = None
    for step in sorted(files):
        data = _read(files[step])
        if not data or 'original_centered' not in data.get('suites', {}):
            if step > 0:
                qualifying.clear()
            continue
        suite = data['suites']['original_centered']
        count = suite.get('successful_episodes', round(suite['success_rate']*data['episodes_per_suite']))
        latest = {'step': step, 'successful_episodes': count,
                  'episodes': data['episodes_per_suite'], **_metric(data, 'original_centered')}
        # Step zero evaluates the pretrained policy but cannot establish
        # repeated progress after online learning has started.
        if step <= 0:
            continue
        valid = (data.get('seed_start') == SEED_START and data.get('episodes_per_suite') == 25
                 and data.get('protocols', {}).get('original_centered', {}).get('id')
                 == 'centered-clean-pass-v1')
        if valid and count >= 20:
            qualifying.append(step)
        else:
            qualifying.clear()
    return latest, {'successful_episodes': 20, 'episodes': 25, 'required_checkpoints': 2,
                    'consecutive_checkpoints': len(qualifying), 'checkpoint_steps': qualifying,
                    'met': len(qualifying) >= 2}


def read_comparison(candidate_run):
    candidate_run = Path(candidate_run)
    files = _checkpoints(candidate_run)
    latest, target = _target_progress(files)
    result = {
        'original_url': BASELINE_URL, 'corrected_url': CORRECTED_URL, 'guided_url': GUIDED_URL,
        'candidate_run': str(candidate_run),
        'original': _status(BASELINE_URL), 'corrected': _status(CORRECTED_URL),
        'matched': None, 'latest_original_centered': latest, 'target': target,
        'step_count_note': 'Online steps exclude demonstrations and pretraining. Equal online steps are not equal total compute.',
    }
    original_files, corrected_files = _checkpoints(BASELINE_RUN), _checkpoints(CORRECTED_RUN)
    common = original_files.keys() & corrected_files.keys() & files.keys()
    if not common:
        return result
    step = max(common)
    original, corrected, guided = (_read(original_files[step]), _read(corrected_files[step]), _read(files[step]))
    if not all((original, corrected, guided)):
        result['warning'] = 'A matching checkpoint report could not be read.'
        return result
    if any(data.get('seed_start') != SEED_START or data.get('episodes_per_suite') != 25
           for data in (original, corrected, guided)):
        result['warning'] = 'Evaluation seeds or sample sizes differ; no comparison is shown.'
        return result
    configs = {'original': _read(BASELINE_RUN/'config.json'), 'corrected': _read(CORRECTED_RUN/'config.json')}
    matched = {'step': step, 'episodes_per_suite': 25, 'suites': {}, 'omitted': []}
    for suite in ('original_centered', 'full_task', 'full_task_delay_200ms'):
        if all(_protocol_matches(data, guided, suite, configs[label])
               for label, data in (('original', original), ('corrected', corrected))):
            source = 'lesson' if suite == 'original_centered' else suite
            matched['suites'][suite] = {
                'original': _metric(original, source), 'corrected': _metric(corrected, source),
                'guided': _metric(guided, suite),
            }
        else:
            matched['omitted'].append(suite)
    result['matched'] = matched
    return result
