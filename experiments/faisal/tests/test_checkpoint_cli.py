import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'tools/evaluate_checkpoints.py'
spec = importlib.util.spec_from_file_location('checkpoint_cli', SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_defaults_are_portable_and_use_reserved_selection_range():
    args = cli.parse_args([])
    assert args.models == [Path('models')/f'c{step}k.zip' for step in (520, 670, 1000)]
    assert (args.single_episodes, args.full_episodes, args.seed_start) == (100, 25, 4_000_000)
    assert cli.resolve_path(args.output) == cli.ROOT/'runs/checkpoint-selection'


@pytest.mark.parametrize('arguments', [
    ['--single-episodes', '0'], ['--full-episodes', '-1'], ['--seed-start', '-1'],
])
def test_invalid_counts_or_seed_are_rejected(arguments):
    with pytest.raises(SystemExit):
        cli.parse_args(arguments)


def test_model_labels_do_not_expose_absolute_paths(tmp_path):
    root = tmp_path/'experiment'
    assert cli.portable_path(root/'models/c520k.zip', root) == 'models/c520k.zip'
    assert cli.portable_path(tmp_path/'private-folder/custom.zip', root) == 'custom.zip'


def candidate(centered, varied, full, crash):
    return {'suites': {'centered': {'success_rate': centered},
                       'varied': {'success_rate': varied},
                       'full_task': {'success_rate': full, 'crash_rate': crash}}}


def test_rank_prioritizes_weaker_single_pass_suite_then_full_task():
    robust = candidate(.9, .8, .0, .8)
    uneven = candidate(1., .7, 1., .0)
    better_full = candidate(.9, .8, .2, .8)
    lower_crash = candidate(.9, .8, .2, .4)
    assert sorted([uneven, robust, better_full, lower_crash], key=cli.rank_key, reverse=True) == [
        lower_crash, better_full, robust, uneven]
