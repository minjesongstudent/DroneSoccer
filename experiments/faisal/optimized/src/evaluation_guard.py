"""Configurable checkpoint selection and stopping rules for fixed validation.

These experiment gates are not evidence of real-flight reliability. Passing
them requests an independent test; it never marks a policy as solved. This
module only consumes reports and returns decisions. It does not save models,
change rewards, train a policy, or control the simulator.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real


SUITES = ('original_centered', 'varied_approach', 'full_task',
          'full_task_delay_200ms', 'lesson')
FIXED_SUITES = SUITES[:-1]
EPSILON = 1e-12


@dataclass(frozen=True)
class GuardConfig:
    """Fractions, evaluation counts, and *new training* steps for this trial."""

    max_retention_drop: float = .12
    min_centered_success: float = .70
    min_varied_success: float = .60
    severe_regression_drop: float = .20
    severe_regression_patience: int = 3
    regression_warmup_steps: int = 20_000
    stagnation_steps: int = 100_000
    candidate_full_success: float = .80
    candidate_delayed_success: float = .80
    candidate_centered_success: float = .85
    candidate_varied_success: float = .80
    candidate_patience: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith('_steps') or name.endswith('_patience'):
                minimum = 1 if name.endswith('_patience') else 0
                if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                    raise ValueError(f'{name} must be an integer >= {minimum}')
            else:
                _rate(value, name)


def _rate(value, name):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f'{name} must be a finite rate between 0 and 1')
    return float(value)


def _meets(value, threshold):
    # Thresholds such as .96 - .12 must not fail because of roundoff.
    return value + EPSILON >= threshold


def _better(left, right):
    for candidate, incumbent in zip(left, right):
        if abs(candidate - incumbent) > EPSILON:
            return candidate > incumbent
    return False


class EvaluationGuard:
    """Observe step zero, then increasing checkpoints from one fixed protocol.

    ``observe(report)`` returns a plain dictionary. ``to_dict`` can be saved
    directly as JSON; ``from_dict`` reconstructs every counter and decision by
    replaying compact reports. A duplicate of the last report is idempotent,
    which permits resuming at an already evaluated checkpoint.

    The baseline remains the fallback even if it misses the eligibility gates.
    Eligible candidates rank by full-task success, then delayed success, then
    lower full-task crash rate. Exact ties preserve the earlier checkpoint.
    Stagnation means no *eligible selected* improvement over baseline full-task
    success; an unsafe candidate cannot disable this stopping rule.
    """

    def __init__(self, config=None):
        self.config = config if config is not None else GuardConfig()
        if not isinstance(self.config, GuardConfig):
            raise TypeError('config must be a GuardConfig')
        self._reports = []
        self._decisions = []
        self._best = None
        self._regression_streak = 0
        self._candidate_streak = 0
        self._stop_reason = None
        self._independent_test_needed = False

    @property
    def baseline_step(self):
        return self._reports[0]['step'] if self._reports else None

    @property
    def best_step(self):
        return self._best['step'] if self._best is not None else None

    @property
    def should_stop(self):
        return self._stop_reason is not None

    @property
    def stop_reason(self):
        return self._stop_reason

    @property
    def independent_test_needed(self):
        return self._independent_test_needed

    @property
    def history(self):
        return deepcopy(self._decisions)

    @staticmethod
    def _compact(report):
        step = report.get('step')
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 0:
            raise ValueError('report step must be a nonnegative integer')
        suites = report.get('suites', {})
        compact = {'step': int(step), 'suites': {}}
        for suite in SUITES:
            if suite not in suites:
                raise ValueError(f'Missing evaluation suite: {suite}')
            compact['suites'][suite] = {
                key: _rate(suites[suite].get(key), f'{suite}.{key}')
                for key in ('success_rate', 'crash_rate')}
        for key in ('episodes_per_suite', 'seed_start', 'policy_mode'):
            if key in report:
                compact[key] = deepcopy(report[key])
        if 'protocols' in report:
            if any(name not in report['protocols'] for name in FIXED_SUITES):
                raise ValueError('Report protocols must describe all fixed suites')
            # The lesson may change with curriculum stage; fixed tests may not.
            compact['protocols'] = {name: deepcopy(report['protocols'][name])
                                    for name in FIXED_SUITES}
        return compact

    @staticmethod
    def _success(report, suite):
        return report['suites'][suite]['success_rate']

    @classmethod
    def _rank(cls, report):
        return (cls._success(report, 'full_task'),
                cls._success(report, 'full_task_delay_200ms'),
                -report['suites']['full_task']['crash_rate'])

    def observe(self, report):
        current = self._compact(report)
        step = current['step']
        if not self._reports and step != 0:
            raise ValueError('Initialize the guard with a step-zero baseline')
        if self._reports:
            last = self._reports[-1]
            if current == last:
                return deepcopy(self._decisions[-1])
            if step <= last['step']:
                raise ValueError('Evaluation steps must increase; a duplicate must be identical')
            baseline = self._reports[0]
            for key in ('episodes_per_suite', 'seed_start', 'policy_mode', 'protocols'):
                if (key in baseline) != (key in current) or baseline.get(key) != current.get(key):
                    raise ValueError(f'Fixed validation {key} changed from baseline')
        else:
            baseline = current

        config = self.config
        centered = self._success(current, 'original_centered')
        varied = self._success(current, 'varied_approach')
        centered_baseline = self._success(baseline, 'original_centered')
        varied_baseline = self._success(baseline, 'varied_approach')
        thresholds = {
            'original_centered': max(config.min_centered_success,
                                     centered_baseline - config.max_retention_drop),
            'varied_approach': max(config.min_varied_success,
                                  varied_baseline - config.max_retention_drop)}
        eligible = (_meets(centered, thresholds['original_centered'])
                    and _meets(varied, thresholds['varied_approach']))
        selected = self._best is None or (eligible and _better(self._rank(current), self._rank(self._best)))
        if selected:
            self._best = current

        severe = (centered_baseline - centered > config.severe_regression_drop + EPSILON
                  or varied_baseline - varied > config.severe_regression_drop + EPSILON)
        if step >= config.regression_warmup_steps and severe:
            self._regression_streak += 1
        else:
            self._regression_streak = 0

        candidate = (eligible
                     and _meets(centered, config.candidate_centered_success)
                     and _meets(varied, config.candidate_varied_success)
                     and _meets(self._success(current, 'full_task'), config.candidate_full_success)
                     and _meets(self._success(current, 'full_task_delay_200ms'), config.candidate_delayed_success))
        self._candidate_streak = self._candidate_streak + 1 if candidate else 0
        full_improved = (self._success(self._best, 'full_task')
                         > self._success(baseline, 'full_task') + EPSILON)
        if self._candidate_streak >= config.candidate_patience:
            self._independent_test_needed = True
        if self._stop_reason is None:
            if self._regression_streak >= config.severe_regression_patience:
                self._stop_reason = 'severe_regression'
            elif self._independent_test_needed:
                self._stop_reason = 'independent_test_required'
            elif step >= config.stagnation_steps and not full_improved:
                self._stop_reason = 'stagnation'

        decision = {
            'step': step, 'baseline_step': baseline['step'], 'best_step': self.best_step,
            'eligible': eligible, 'selected': selected,
            'retention_thresholds': thresholds,
            'success_rates': {name: self._success(current, name) for name in SUITES},
            'full_task_crash_rate': current['suites']['full_task']['crash_rate'],
            'severe_regression': severe, 'regression_streak': self._regression_streak,
            'full_task_improved': full_improved,
            'candidate_thresholds_met': candidate, 'candidate_streak': self._candidate_streak,
            'independent_test_needed': self.independent_test_needed,
            'should_stop': self.should_stop, 'stop_reason': self.stop_reason,
            'solved': False}
        self._reports.append(current)
        self._decisions.append(decision)
        return deepcopy(decision)

    def to_dict(self):
        return {'schema_version': 1, 'config': asdict(self.config),
                'history': deepcopy(self._reports)}

    @classmethod
    def from_dict(cls, state):
        if state.get('schema_version') != 1:
            raise ValueError('Unsupported evaluation guard schema version')
        if not isinstance(state.get('history'), list):
            raise ValueError('Evaluation guard history must be a list')
        guard = cls(GuardConfig(**state['config']))
        for report in state['history']:
            guard.observe(report)
        return guard
