"""Select return-practice progress without changing full-mission acceptance.

The fixed 50-start development benchmark can identify an intermediate skill.
It cannot declare the mission solved or request the independent 7M-seed test.
Reports supplied here must never be sampled into training replay.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real


EPSILON = 1e-12


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _rate(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f'{name} must be a finite rate between 0 and 1')
    return float(value)


@dataclass(frozen=True)
class ReturnSelectionConfig:
    """Predeclared trial rules, not universal reliability thresholds.

    Reference retention rates belong to original C-520k. They do not reset to
    the next experiment's weaker inherited policy. Every skill comparison uses
    the fixed step-zero return benchmark from this experiment.
    """

    episodes: int = 50
    reference_centered: float = 24 / 25
    reference_varied: float = 23 / 25
    retention_drop: float = .12
    centered_floor: float = .70
    varied_floor: float = .60
    improvement_episodes: int = 5
    improvement_checks: int = 2
    extension_gate_steps: int = 20_000
    no_new_pass_cap_steps: int = 40_000

    def __post_init__(self):
        for name in ('episodes', 'improvement_episodes', 'improvement_checks'):
            _integer(getattr(self, name), name, 1)
        for name in ('extension_gate_steps', 'no_new_pass_cap_steps'):
            _integer(getattr(self, name), name, 1)
        for name in ('reference_centered', 'reference_varied', 'retention_drop',
                     'centered_floor', 'varied_floor'):
            _rate(getattr(self, name), name)
        if self.improvement_episodes > self.episodes:
            raise ValueError('improvement_episodes cannot exceed benchmark episodes')
        if self.no_new_pass_cap_steps < self.extension_gate_steps:
            raise ValueError('The no-new-pass cap cannot precede the extension gate')


class ReturnSkillSelector:
    """Pure, JSON-serializable selection for one immutable return benchmark.

    ``observe`` consumes a merged report with original centered/varied success
    rates, a ``return_practice`` suite, and ``protocols.return_practice``. The
    latter declares ``id``, ``bank_sha256`` and the exact evaluation ``seeds``.

    Eligible candidates rank by successful new-pass episodes, safe front-return
    episodes, then lower crash rate. Ties keep the earlier model. Step zero is
    always preserved as a fallback, including a zero-success return baseline.

    Extension requires two consecutive eligible checks with at least five more
    new-pass successes OR safe front returns than the fixed baseline. Once
    earned, this permission persists; the separate mission guard still handles
    retention regression. A stop decision latches. Neither permission nor model
    selection starts training, overrides that guard, or certifies completion.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else ReturnSelectionConfig()
        if not isinstance(self.config, ReturnSelectionConfig):
            raise TypeError('config must be ReturnSelectionConfig')
        self._reports = []
        self._decisions = []
        self._best = None
        self._improvement_streak = 0
        self._extension_allowed = False
        self._stop_reason = None

    @property
    def best_return_step(self):
        return self._best['step'] if self._best is not None else None

    @property
    def should_stop(self):
        return self._stop_reason is not None

    @property
    def stop_reason(self):
        return self._stop_reason

    @property
    def extension_allowed(self):
        return self._extension_allowed

    @property
    def history(self):
        return deepcopy(self._decisions)

    @property
    def retention_thresholds(self):
        c = self.config
        return {'original_centered': max(c.centered_floor, c.reference_centered-c.retention_drop),
                'varied_approach': max(c.varied_floor, c.reference_varied-c.retention_drop)}

    def _compact(self, report):
        step = _integer(report.get('step'), 'step')
        suites = report.get('suites', {})
        compact = {'step': step, 'suites': {}}
        for name in ('original_centered', 'varied_approach'):
            compact['suites'][name] = {
                'success_rate': _rate(suites.get(name, {}).get('success_rate'), f'{name}.success_rate')}
        practice = suites.get('return_practice', {})
        episodes = _integer(practice.get('episodes'), 'return_practice.episodes', 1)
        if episodes != self.config.episodes:
            raise ValueError('Return benchmark episode count differs from the declared configuration')
        counts = {name: _integer(practice.get(name), f'return_practice.{name}')
                  for name in ('successful_episodes', 'safe_front_return_episodes')}
        if any(count > episodes for count in counts.values()):
            raise ValueError('Return benchmark counts cannot exceed the episode count')
        crash = _rate(practice.get('crash_rate'), 'return_practice.crash_rate')
        compact['suites']['return_practice'] = {'episodes': episodes, **counts, 'crash_rate': crash}

        protocol = deepcopy(report.get('protocols', {}).get('return_practice', {}))
        if not isinstance(protocol.get('id'), str) or not protocol['id'].strip():
            raise ValueError('Return benchmark requires a protocol id')
        digest = protocol.get('bank_sha256', '')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdefABCDEF' for c in digest)):
            raise ValueError('Return benchmark requires its bank_sha256')
        seeds = protocol.get('seeds')
        if not isinstance(seeds, list) or len(seeds) != episodes:
            raise ValueError('Return benchmark requires one explicit seed per episode')
        seeds = [_integer(seed, 'evaluation seed') for seed in seeds]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Return benchmark evaluation seeds must be unique')
        protocol['seeds'] = seeds
        protocol['bank_sha256'] = digest.lower()
        compact['protocols'] = {'return_practice': protocol}
        return compact

    @staticmethod
    def _practice(report):
        return report['suites']['return_practice']

    @classmethod
    def _better(cls, current, incumbent):
        a, b = cls._practice(current), cls._practice(incumbent)
        for key in ('successful_episodes', 'safe_front_return_episodes'):
            if a[key] != b[key]:
                return a[key] > b[key]
        return a['crash_rate'] < b['crash_rate']-EPSILON

    def observe(self, report):
        current = self._compact(report)
        step = current['step']
        if not self._reports:
            if step != 0:
                raise ValueError('Initialize return selection with a step-zero baseline')
            baseline = current
        else:
            if current == self._reports[-1]:
                return deepcopy(self._decisions[-1])
            if step <= self._reports[-1]['step']:
                raise ValueError('Return evaluation steps must increase; duplicates must be identical')
            baseline = self._reports[0]
            if current['protocols'] != baseline['protocols']:
                raise ValueError('Return protocol, bank, or evaluation seeds changed from baseline')

        eligible = all(current['suites'][name]['success_rate']+EPSILON >= threshold
                       for name, threshold in self.retention_thresholds.items())
        selected = self._best is None or (eligible and self._better(current, self._best))
        if selected:
            self._best = current
        base, now, best = map(self._practice, (baseline, current, self._best))
        new_pass_gain = now['successful_episodes']-base['successful_episodes']
        front_gain = now['safe_front_return_episodes']-base['safe_front_return_episodes']
        improved = eligible and max(new_pass_gain, front_gain) >= self.config.improvement_episodes
        self._improvement_streak = self._improvement_streak+1 if improved else 0
        if self._improvement_streak >= self.config.improvement_checks:
            self._extension_allowed = True
        genuine_new_pass_gain = best['successful_episodes'] > base['successful_episodes']
        if self._stop_reason is None:
            if step >= self.config.no_new_pass_cap_steps and not genuine_new_pass_gain:
                self._stop_reason = 'return_no_new_pass_gain'
            elif step >= self.config.extension_gate_steps and not self.extension_allowed:
                self._stop_reason = 'return_no_sustained_progress'

        decision = {
            'step': step, 'baseline_step': baseline['step'],
            'best_return_step': self.best_return_step, 'eligible': eligible, 'selected': selected,
            'retention_thresholds': self.retention_thresholds,
            'successful_episodes': now['successful_episodes'],
            'safe_front_return_episodes': now['safe_front_return_episodes'],
            'new_pass_gain': new_pass_gain, 'safe_front_return_gain': front_gain,
            'improvement_streak': self._improvement_streak,
            'extension_allowed': self.extension_allowed,
            'genuine_new_pass_gain': genuine_new_pass_gain,
            'should_stop': self.should_stop, 'stop_reason': self.stop_reason,
            'mission_acceptance': False, 'independent_test_needed': False}
        self._reports.append(current)
        self._decisions.append(decision)
        return deepcopy(decision)

    def to_dict(self):
        return {'schema_version': 1, 'config': asdict(self.config), 'history': deepcopy(self._reports)}

    @classmethod
    def from_dict(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported return-selection schema version')
        if not isinstance(state.get('history'), list):
            raise ValueError('Return selection history must be a list')
        selector = cls(ReturnSelectionConfig(**state['config']))
        for report in state['history']:
            selector.observe(report)
        return selector
