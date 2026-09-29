"""Training-only simulator context, kept entirely outside the actor observation.

The critic can condition its value estimates on simulator state and hidden
observation-history state. This does not give a camera-only actor information
with which to distinguish identical observations, or guarantee that it can
complete the task. The existing 100-value observation and snapshot format are
unchanged. Context is carried separately in transition info for replay only.

Values use fixed physical scales, never clipping: clipping raw estimator
displacement would discard state needed when an estimate returns into range.
Most values are order one; negative wall clearances and values outside [-1, 1]
are valid, especially on terminal transitions. RNG internals are not features:
the control-interval distribution describes stochastic transitions instead.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from src import contract
from src.mission_observation import MissionObservation, OBSERVATION_DIM
from src.observation_memory import ObservationMemory
from src.return_mixture import ReturnMixtureEnv
from src.return_practice import ReturnPracticeEnv
from src.sim import HoopEnv, LAPS_MAX_STEPS


VERSION = 'separate-simulator-critic-context-v1'
MAX_DELAY_STEPS = 3
OMITTED_FRAME_LAGS = tuple(lag for lag in range(contract.FRAME_HISTORY-1, -1, -1)
                          if lag not in contract.OBS_LAGS)
PHASES = ('approach', 'return_side', 'return_front', 'align_front')

# Order and dimensions are a checkpoint/replay contract. Public actor inputs
# are concatenated with these only inside the critic, never inside an actor.
_LAYOUT = (
    ('motion', 9),               # position, sin/cos heading, velocity, yaw rate
    ('ring', 13),                # centre, normal, radii, thickness, hanger, detector geometry
    ('wall_clearances', 6),      # centre-to-geofence distances, x-/x+/y-/y+/z-/z+
    ('geometry', 8),             # cage sizes/margins and drone collision-box sizes
    ('camera', 7),               # normalized intrinsics, sin/cos pitch, offset
    ('dynamics', 8),             # tau, delay, speed, yaw limit, dt, random flag, rate range
    ('pending_commands', 12),    # oldest pending command first, zero padded
    ('route', 17),               # phase, latch+gate, pending crossing, feasible gates
    ('clock', 5),                # physical steps, task, practice flag, local elapsed/cap
    ('omitted_frames', len(OMITTED_FRAME_LAGS)*contract.OBS_DIM),
    ('memory_state', 19),        # last frame, seen, age, raw displacement/velocity/yaw/rate
    ('memory_nominal', 5),       # nominal tau, dt, speed, yaw limit, delay
    ('memory_pending_commands', 12),
)
CONTEXT_DIM = sum(size for _, size in _LAYOUT)
CONTEXT_SLICES = {}
_offset = 0
for _name, _size in _LAYOUT:
    CONTEXT_SLICES[_name] = slice(_offset, _offset+_size)
    _offset += _size
del _offset, _name, _size

_FRAME_SCALE = np.array([contract.CLIP]*3+[1.]*5, dtype=np.float64)
_METRES = 3.0
_SPEED = .4
_YAW_RATE = np.deg2rad(60.)


def _contract(env):
    if (not isinstance(env.observation_space, gym.spaces.Box)
            or env.observation_space.shape != (100,)
            or env.observation_space.dtype != np.float32
            or OBSERVATION_DIM != 100
            or env.action_space.shape != (4,)):
        raise ValueError('Critic context requires the unchanged float32 100D actor / 4D action contract.')


def _parts(env):
    """Follow the active branch on every call; practice reset replaces its env."""
    _contract(env)
    mission = memory = practice = None
    visited = set()
    while True:
        if id(env) in visited:
            raise ValueError('Cyclic environment wrapper chain.')
        visited.add(id(env))
        if isinstance(env, ReturnMixtureEnv):
            env = env.active
            continue
        if isinstance(env, ReturnPracticeEnv):
            if practice is not None:
                raise ValueError('Nested return-practice wrappers are unsupported.')
            practice = env
        if isinstance(env, MissionObservation):
            if mission is not None:
                raise ValueError('Nested mission observations are unsupported.')
            mission = env
        if isinstance(env, ObservationMemory):
            if memory is not None:
                raise ValueError('Nested camera-command memories are unsupported.')
            memory = env.memory
        if isinstance(env, gym.Wrapper):
            env = env.env
            continue
        if not isinstance(env, HoopEnv) or mission is None or memory is None:
            raise ValueError('Critic context requires a MissionObservation-wrapped HoopEnv active branch.')
        if not hasattr(env, '_frames'):
            raise RuntimeError('Reset the environment before reading critic context.')
        return env, mission, memory, practice


def _vector(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Invalid {name}: expected finite shape {shape}.')
    return result


def _delay_queue(pending, delay, name):
    if (not np.isfinite(delay) or int(delay) != delay
            or not 0 <= delay <= MAX_DELAY_STEPS):
        raise ValueError(f'{name} delay must be an integer between 0 and {MAX_DELAY_STEPS}.')
    if len(pending) != delay:
        raise ValueError(f'{name} pending queue length disagrees with its delay.')
    padded = np.zeros((MAX_DELAY_STEPS, contract.ACTION_DIM), dtype=np.float64)
    for index, action in enumerate(pending):
        padded[index] = _vector(action, (4,), f'{name} command')
        if np.any(np.abs(padded[index]) > 1.):
            raise ValueError(f'{name} pending commands must already be clipped.')
    return padded.ravel()


def _validate_dynamics(drone, name):
    values = np.array([drone.tau, drone.dt, drone.max_speed, drone.max_yaw_rate_deg])
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f'{name} requires finite positive response and rate parameters.')


def critic_context(env) -> np.ndarray:
    """Read current training state without stepping, resetting, or drawing RNG.

    Supported: upright hoop geometry, 25 camera frames / nine exposed lags,
    three issued actions, and true/nominal delays from zero through three.
    The 16 omitted frames complement the nine in the unchanged actor prefix.
    Full raw estimator state complements its compressed/clipped actor output.
    No policy, checkpoint, snapshot payload, or simulator state is mutated.
    """
    base, mission, memory, practice = _parts(env)
    if (contract.POLICY_INPUT_DIM != 84 or contract.FRAME_HISTORY != 25
            or contract.OBS_STACK != 9 or contract.ACTION_HISTORY != 3
            or contract.OBS_DIM != 8 or len(OMITTED_FRAME_LAGS) != 16):
        raise ValueError('Unsupported camera/action history contract.')
    if len(base._frames) != 25 or len(base._actions) != 3:
        raise ValueError('Incomplete camera/action history.')
    frames = _vector(list(base._frames), (25, 8), 'camera history')
    actions = _vector(list(base._actions), (3, 4), 'action history')
    if np.any(np.abs(actions) > 1.):
        raise ValueError('Issued action history must already be clipped.')

    d, n, cage, hoop, camera = base.dyn, memory.nominal, base.cage, base.hoop, base.camera
    _validate_dynamics(d, 'Simulator')
    _validate_dynamics(n, 'Nominal memory')
    pending = _delay_queue(base._pending, d.delay_steps, 'Simulator')
    memory_pending = _delay_queue(memory._pending, n.delay_steps, 'Nominal memory')
    position = _vector(base.pos, (3,), 'drone position')
    centre = _vector(base.hoop_centre, (3,), 'ring centre')
    normal = _vector(base.hoop_normal, (3,), 'ring normal')
    if abs(normal[2]) > 1e-8 or not np.isclose(np.linalg.norm(normal), 1.):
        raise ValueError('Only upright hoops with a unit horizontal normal are supported.')
    if base._lap_phase not in PHASES:
        raise ValueError('Unknown lap reward phase.')
    gates = np.zeros((2, 3))
    gate_mask = np.zeros(2)
    if not 1 <= len(base._gates) <= 2:
        raise ValueError('Expected one or two feasible return gates.')
    for index, gate in enumerate(base._gates):
        gates[index] = _vector(gate, (3,), 'feasible gate')/_METRES
        gate_mask[index] = 1.
    has_gate = base._return_gate is not None
    if base._lap_phase == 'return_front' and not has_gate:
        raise ValueError('Return-front phase requires its latched gate.')
    gate = (_vector(base._return_gate, (3,), 'latched gate')/_METRES
            if has_gate else np.zeros(3))
    rates = _vector(base.ranges.rate_hz, (2,), 'control-rate range')
    if rates[0] <= 0 or rates[1] < rates[0]:
        raise ValueError('Invalid control-rate range.')
    if not np.isfinite([camera.width, camera.height]).all() or min(camera.width, camera.height) <= 0:
        raise ValueError('Camera dimensions must be finite and positive.')
    if not np.isfinite(base.steps) or int(base.steps) != base.steps or base.steps < 0:
        raise ValueError('Invalid physical episode clock.')
    local_elapsed = base.steps if practice is None else practice.elapsed
    local_cap = (150 if mission.single_pass else LAPS_MAX_STEPS) if practice is None else practice.horizon
    if (not np.isfinite([local_elapsed, local_cap]).all() or local_elapsed < 0
            or local_cap <= 0 or int(local_elapsed) != local_elapsed or int(local_cap) != local_cap):
        raise ValueError('Invalid local continuation clock.')
    pitch = np.deg2rad(camera.pitch_deg)
    half_t = hoop.thickness/2
    radius_h, radius_v = base._line_h+half_t, base._line_v+half_t

    blocks = {
        'motion': np.r_[position/_METRES, np.sin(base.heading), np.cos(base.heading),
                        _vector(base.vel, (3,), 'velocity')/_SPEED, base.yaw_rate/_YAW_RATE],
        'ring': np.r_[centre/_METRES, normal, radius_h, radius_v, hoop.thickness,
                      hoop.hanger_half_width, hoop.n_points/32., base._marker_cos, float(base.hanger)],
        'wall_clearances': np.array([position[0]-cage.margin, cage.length-cage.margin-position[0],
                                    position[1]-cage.margin, cage.width-cage.margin-position[1],
                                    position[2]-cage.min_altitude, cage.height-cage.margin-position[2]])/_METRES,
        'geometry': np.r_[np.array([cage.length, cage.width, cage.height, cage.margin, cage.min_altitude])/_METRES,
                          base.drone.length, base.drone.width, base.drone.height],
        'camera': np.array([camera.fx/camera.width, camera.fy/camera.height,
                            camera.cx/camera.width, camera.cy/camera.height,
                            np.sin(pitch), np.cos(pitch), camera.offset_forward/.1]),
        'dynamics': np.r_[d.tau/.3, d.delay_steps/MAX_DELAY_STEPS, d.max_speed/_SPEED,
                         d.max_yaw_rate_deg/60., d.dt/.1, float(base.randomise), rates/10.],
        'pending_commands': pending,
        'route': np.r_[[float(base._lap_phase == p) for p in PHASES], float(has_gate), gate,
                       float(base._crossed_pending), gate_mask, gates.ravel()],
        'clock': np.array([base.steps/LAPS_MAX_STEPS, float(mission.single_pass), float(practice is not None),
                           local_elapsed/LAPS_MAX_STEPS, local_cap/LAPS_MAX_STEPS]),
        'omitted_frames': (frames[[-1-lag for lag in OMITTED_FRAME_LAGS]]/_FRAME_SCALE).ravel(),
        'memory_state': np.r_[_vector(memory._last, (8,), 'last detection')/_FRAME_SCALE,
                             float(memory._has_seen), memory._age/60.,
                             _vector(memory._displacement, (3,), 'raw estimated displacement')/_METRES,
                             _vector(memory._velocity, (3,), 'estimated velocity')/_SPEED,
                             np.sin(memory._yaw), np.cos(memory._yaw), memory._yaw_rate/_YAW_RATE],
        'memory_nominal': np.array([n.tau/.3, n.dt/.1, n.max_speed/_SPEED,
                                    n.max_yaw_rate_deg/60., n.delay_steps/MAX_DELAY_STEPS]),
        'memory_pending_commands': memory_pending,
    }
    for name, size in _LAYOUT:
        _vector(blocks[name], (size,), name)
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.concatenate([blocks[name] for name, _ in _LAYOUT]).astype(np.float32)
    if result.shape != (CONTEXT_DIM,) or not np.isfinite(result).all():
        raise ValueError('Critic context must be a finite float32 vector.')
    return result


class CriticContextWrapper(gym.Wrapper):
    """Capture transition context before vector wrappers can auto-reset.

    Place inside DummyVecEnv, outside the whole environment/Monitor/mixture.
    ``current_critic_context`` is a defensive copy for replay initialization
    after reset. Reset info also carries ``critic_context`` for vector APIs.
    A terminal transition's info owns independent pre/post arrays: a later
    reset never replaces its post-action context with the next episode's state.
    """
    def __init__(self, env):
        _contract(env)
        super().__init__(env)
        self._current_critic_context = None
        self.critic_context_version = VERSION

    @property
    def current_critic_context(self):
        if self._current_critic_context is None:
            raise RuntimeError('Reset the environment before reading critic context.')
        return self._current_critic_context.copy()

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._current_critic_context = critic_context(self.env)
        return observation, dict(info, critic_context=self.current_critic_context)

    def step(self, action):
        if self._current_critic_context is None:
            raise RuntimeError('Reset the environment before stepping with critic context.')
        before = critic_context(self.env)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._current_critic_context = critic_context(self.env)
        return observation, reward, terminated, truncated, dict(
            info, critic_context=before, next_critic_context=self.current_critic_context)
