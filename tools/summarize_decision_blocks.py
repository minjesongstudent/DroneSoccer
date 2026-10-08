"""Compare conditional learned-prefix outcomes with single-action families."""
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
from tools.audit_family_noise import summarize
from tools.run_clearance_experiment import sha, write_json


def main():
    root = Path('D:/DroneSoccerData/runs')
    block = root / 'decision-block-20261007'
    single = root / 'critic-independent-families-20261007'
    out = root / 'decision-block-analysis-20261007'
    datasets = {}
    hashes = {}
    for name, folder in [('block', block), ('single', single)]:
        assert (folder / 'DONE').exists()
        rows = [json.loads(line) for line in (folder / 'continuations.jsonl').read_text().splitlines()]
        keys = [(r['state'], r['candidate'], r['replica']) for r in rows]
        assert len(keys) == 1536 and len(set(keys)) == 1536
        assert set(keys) == set(itertools.product(range(64), range(4), range(6)))
        rows.sort(key=lambda r: (r['state'], r['candidate'], r['replica']))
        datasets[name] = rows
        for filename in ['protocol.json', 'continuations.jsonl', 'model.zip']:
            hashes[str(folder / filename)] = sha(folder / filename)
    assert hashes[str(block / 'model.zip')] == hashes[str(single / 'model.zip')]
    terminal = np.array([r['prefix_terminal'] for r in datasets['block']]).reshape(64, 4, 6)
    assert (terminal == terminal[:, :, :1]).all()
    surviving = ~terminal.any(axis=(1, 2))
    fields = ['finite_soft_return', 'bootstrapped_soft_return', 'mission']
    result = dict(input_hashes=hashes, terminal_prefixes=int(terminal[:, :, 0].sum()),
                  excluded_states=np.flatnonzero(~surviving).tolist(), comparisons={})
    arrays = {}
    for name, rows in datasets.items():
        arrays[name] = {field: np.array([float(r[field]) for r in rows]).reshape(64, 4, 6) for field in fields}
        for values in arrays[name].values():
            assert np.isfinite(values).all()
        result['comparisons'][name] = {}
        for scope, keep in [('all', np.ones(64, dtype=bool)), ('all_prefixes_survive', surviving)]:
            result['comparisons'][name][scope] = {}
            for split, mask in [('fit', np.arange(64) % 5 != 4), ('heldout', np.arange(64) % 5 == 4)]:
                selected = np.flatnonzero(keep & mask)
                assert len(selected)
                result['comparisons'][name][scope][split] = {
                    field: summarize(values[selected]) for field, values in arrays[name].items()}
    result['caveat'] = ('Block results condition on realized learned prefixes. Single and block future RNG differ. '
                        'Survivor subset is descriptive and selected using block outcomes. '
                        'Contrasts within a family are dependent; no adjusted significance or policy improvement claim.')
    out.mkdir(exist_ok=False)
    write_json(out / 'report.json', result)
    np.savez_compressed(out / 'outcomes.npz', **{name+'_'+field: v for name, fs in arrays.items() for field, v in fs.items()}, surviving=surviving)
    shutil.copy2(__file__, out / 'analysis_source.py')
    shutil.copy2(Path(__file__).with_name('audit_family_noise.py'), out / 'summary_source.py')
    (out / 'DONE').write_text('Offline conditional comparison complete; no policy updates.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
