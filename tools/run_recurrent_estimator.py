"""Bounded offline pre-action motion estimation; never a flight controller.

python -m tools.run_recurrent_estimator --out D:/DroneSoccerData/runs/<new-name>
Only saved learned-policy trajectories are read. No simulator is stepped and no
flight actor is loaded or changed. A passing decision only supports considering
future integration of estimated state, not policy promotion or physical flight.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.camera_history import CameraHistory
from src.policy_runtime import SPECS
from tools.run_clearance_experiment import ROOT, sha, single_trainer, write_json


PROFILE = SPECS['seenflag-s1-level1']
WARMUP_RUN = Path('D:/DroneSoccerData/runs/clearance-reward-20261007')
GAE_RUN = Path('D:/DroneSoccerData/runs/gae95-curriculum-20261007')
PRIOR_SPLIT = Path('D:/DroneSoccerData/runs/motion-width-control-20261007/split.npz')
SEED = 11580000
VARIANTS = ('feedforward', 'gru64')


@dataclass
class Episode:
    observations: np.ndarray
    targets: np.ndarray
    mask: np.ndarray
    visible: np.ndarray
    seed: int
    index: int
    trace: str


def episode_arrays(trace):
    """Inputs before action t; target is motion already completed before t.

    Saved positions/headings are AFTER action t. Therefore target[t] uses
    positions[t-1] - positions[t-2], oriented at heading[t-1], never position[t],
    heading[t], action[t], or any future truth. The first two observations are
    retained to initialize recurrent state, but have no supervised target.
    """
    observations = np.asarray(trace['actor_observations'], np.float32)
    actions = np.asarray(trace['actions'], np.float32)
    positions = np.asarray(trace['positions'], np.float64)
    headings = np.asarray(trace['headings'], np.float64)
    n = len(observations)
    if (observations.shape != (n, 85) or actions.shape != (n, 4)
            or positions.shape != (n, 3) or headings.shape != (n,)
            or not 3 <= n <= 600):
        raise ValueError('Complete episode with 3..600 rows and the explicit 85-input profile required')
    if not all(np.isfinite(array).all() for array in (observations, actions, positions, headings)):
        raise ValueError('Non-finite saved episode data')
    if np.any(np.abs(actions) > 1):
        raise ValueError('Issued actions must be normalized')

    # Reading saved actions here verifies history; no current/future action is
    # concatenated into estimator inputs. Rebuild all camera lags and the flag.
    history = CameraHistory(PROFILE.lags)
    for t, observation in enumerate(observations):
        prior = np.zeros(4, np.float32) if t == 0 else actions[t-1]
        rebuilt = history.update(observation[64:72], prior)
        if not np.array_equal(rebuilt, observation):
            raise ValueError(f'Public camera/issued-command history mismatch at row {t}')

    displacement = positions[1:-1] - positions[:-2]
    heading = headings[1:-1]
    c, s = np.cos(heading), np.sin(heading)
    x, y, z = displacement.T
    targets = np.zeros((n, 3), np.float32)
    targets[2:] = np.column_stack((c*x+s*y, s*x-c*y, z)) * 1000.0
    mask = np.arange(n) >= 2
    return observations.copy(), targets, mask, observations[:, 70] > .5


def load_episodes(folder, *, warmup_limit=None, max_episodes=None, expected_total=None,
                  deadline=None, stop_path=None):
    """Use complete whole episodes, in the same order as the prior motion audit."""
    folder = Path(folder).resolve()
    log = folder/'episodes.jsonl'
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    if expected_total is not None and len(rows) != expected_total:
        raise ValueError(f'Expected exactly {expected_total} saved cohort episodes')
    episodes, records, hashes, seeds = [], [], {str(log): sha(log)}, set()
    cumulative = 0
    for row in rows:
        ensure_budget(deadline, stop_path)
        cumulative += row['steps']
        if warmup_limit is not None and cumulative > warmup_limit:
            break  # The episode straddling the 200k boundary is not wholly original warmup.
        if row['status'] != 'complete' or row['steps'] < 3:
            if expected_total is not None:
                raise ValueError('Separate GAE cohort must contain complete usable episodes only')
            continue
        seed = row['seed']
        if type(seed) is not int or not 11_000_000 <= seed < 11_250_000 or seed in seeds:
            raise ValueError('Expected unique FIT episode seeds')
        seeds.add(seed)
        path = (folder/row['trace']).resolve()
        if path.parent != folder:
            raise ValueError('Trace must be directly inside its declared trajectory folder')
        with np.load(path, allow_pickle=False) as trace:
            observations, targets, mask, visible = episode_arrays(trace)
            if len(observations) != row['steps']:
                raise ValueError('Episode length disagrees with metadata')
            terminated = np.asarray(trace['terminated'], bool)
            truncated = np.asarray(trace['truncated'], bool)
            ends = terminated | truncated
            if (ends.shape != (len(observations),) or ends[:-1].any() or not ends[-1]
                    or (terminated[-1] and truncated[-1])):
                raise ValueError('Trace does not contain exactly one complete episode')
        index = len(episodes)
        digest = sha(path)
        hashes[str(path)] = digest
        episodes.append(Episode(observations, targets, mask, visible, seed, index, str(path)))
        records.append(dict(index=index, seed=seed, trace=str(path), sha256=digest,
                            rows=len(observations), targets=int(mask.sum()),
                            visible_targets=int((mask & visible).sum()),
                            blind_targets=int((mask & ~visible).sum())))
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    if not episodes:
        raise ValueError('No complete usable episodes')
    return episodes, records, hashes


def fit_normalizers(episodes):
    """FIT inputs only; target moments use only the t>=2 supervised rows."""
    x = np.concatenate([episode.observations for episode in episodes])
    y = np.concatenate([episode.targets[episode.mask] for episode in episodes])
    return dict(x_mean=x.mean(0, dtype=np.float64).astype(np.float32),
                x_scale=np.maximum(x.std(0, dtype=np.float64), .01).astype(np.float32),
                y_mean=y.mean(0, dtype=np.float64).astype(np.float32),
                y_scale=np.maximum(y.std(0, dtype=np.float64), .001).astype(np.float32))


def episode_schedule(fit_indices, updates):
    """One frozen whole-episode draw sequence, shared without model-specific RNG."""
    return np.random.default_rng(SEED).choice(np.asarray(fit_indices), size=(updates, 8), replace=True)


def make_batch(episodes, indices, normalizers, device='cpu'):
    chosen = [episodes[int(index)] for index in indices]
    lengths = np.array([len(episode.observations) for episode in chosen], np.int64)
    longest = int(lengths.max())
    x = np.zeros((len(chosen), longest, 85), np.float32)
    y = np.zeros((len(chosen), longest, 3), np.float32)
    mask = np.zeros((len(chosen), longest), bool)
    for index, episode in enumerate(chosen):
        length = lengths[index]
        x[index, :length] = (episode.observations-normalizers['x_mean'])/normalizers['x_scale']
        y[index, :length] = (episode.targets-normalizers['y_mean'])/normalizers['y_scale']
        mask[index, :length] = episode.mask
    return (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device),
            torch.as_tensor(mask, device=device), torch.as_tensor(lengths))


class FeedForwardEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(85, 128), nn.Tanh(), nn.Linear(128, 128),
                                     nn.Tanh(), nn.Linear(128, 3))

    def forward(self, observations, lengths):
        return self.network(observations)


class RecurrentEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(85, 64, batch_first=True)
        self.output = nn.Linear(64, 3)

    def forward(self, observations, lengths):
        # No state passed: every whole episode begins at zero, including duplicate
        # sampled episodes. Packing prevents padded rows from extending an episode.
        packed = pack_padded_sequence(observations, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True,
                                       total_length=observations.shape[1])
        return self.output(hidden)

    def stream(self, observations, state=None):
        hidden, state = self.gru(observations, state)
        return self.output(hidden), state


def masked_mse(predictions, targets, mask):
    if predictions.shape != targets.shape or mask.shape != targets.shape[:-1] or not mask.any():
        raise ValueError('Matching predictions/targets and a nonempty target mask required')
    return (predictions[mask]-targets[mask]).square().mean()


def prediction_metrics(predictions, targets, mask, visible):
    predictions, targets = np.asarray(predictions), np.asarray(targets)
    mask, visible = np.asarray(mask, bool), np.asarray(visible, bool)
    if (predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 3
            or mask.shape != (len(targets),) or visible.shape != mask.shape
            or not np.isfinite(predictions[mask]).all() or not np.isfinite(targets[mask]).all()):
        raise ValueError('Invalid finite prediction arrays')
    result = {}
    for name, selection in [('all', mask), ('visible', mask & visible), ('blind', mask & ~visible)]:
        residual = predictions[selection]-targets[selection]
        result[name] = dict(samples=int(selection.sum()),
            rmse_mm_per_axis=float(np.sqrt(np.mean(residual**2))) if selection.any() else None,
            mean_vector_error_mm=float(np.linalg.norm(residual, axis=1).mean()) if selection.any() else None)
    return result


class BoundedStop(RuntimeError):
    pass


def ensure_budget(deadline, stop_path):
    if stop_path is not None and Path(stop_path).exists():
        raise BoundedStop('STOP file requested closure')
    if deadline is not None and time.monotonic() >= deadline:
        raise BoundedStop('Fixed total wall-clock budget exhausted')


def evaluate_estimator(model, episodes, indices, normalizers, destination, device,
                       *, deadline=None, stop_path=None):
    """Final predictions only; preserve first two rows but exclude their targets."""
    arrays = {name: [] for name in ('predictions', 'targets', 'mask', 'visible',
                                   'episode_indices', 'seeds', 'steps')}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 8):
            ensure_budget(deadline, stop_path)
            batch_ids = indices[start:start+8]
            x, _, _, lengths = make_batch(episodes, batch_ids, normalizers, device)
            predicted = model(x, lengths).cpu().numpy()
            predicted = predicted*normalizers['y_scale']+normalizers['y_mean']
            for offset, index in enumerate(batch_ids):
                episode = episodes[int(index)]
                n = len(episode.observations)
                arrays['predictions'].append(predicted[offset, :n])
                for name in ('targets', 'mask', 'visible'):
                    arrays[name].append(getattr(episode, name))
                arrays['episode_indices'].append(np.full(n, episode.index, np.int64))
                arrays['seeds'].append(np.full(n, episode.seed, np.int64))
                arrays['steps'].append(np.arange(n))
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
    metrics = prediction_metrics(*(arrays[name] for name in ('predictions', 'targets', 'mask', 'visible')))
    metrics['episodes'] = len(indices)
    np.savez_compressed(destination, **arrays)
    return metrics


def gate_estimator(results, *, complete, source_unchanged, smoke=False):
    reasons = []
    if smoke:
        reasons.append('Software-only smoke; never an estimator qualification')
    if not complete:
        reasons.append('Both variants must complete their fixed update budgets and all final evaluations')
    if not source_unchanged:
        reasons.append('Original flight policy changed')
    comparisons = {}
    if complete:
        for cohort in ('heldout', 'gae_separate'):
            comparisons[cohort] = {}
            for group in ('all', 'blind', 'visible'):
                old = results['feedforward'][cohort][group]
                new = results['gru64'][cohort][group]
                reference, candidate = old['rmse_mm_per_axis'], new['rmse_mm_per_axis']
                valid = (old['samples'] == new['samples'] and old['samples'] > 0
                         and reference is not None and candidate is not None
                         and np.isfinite([reference, candidate]).all() and reference > 0 and candidate >= 0)
                if not valid:
                    reasons.append(f'{cohort}/{group}: invalid comparable errors or no samples')
                    continue
                ratio = candidate/reference
                comparisons[cohort][group] = dict(feedforward_rmse=reference, recurrent_rmse=candidate,
                                                  relative_reduction=1-ratio, samples=old['samples'])
                if group in ('all', 'blind') and ratio > .85:
                    reasons.append(f'{cohort}/{group}: RMSE reduction below 15%')
                if group == 'visible' and ratio > 1.05:
                    reasons.append(f'{cohort}/{group}: visible RMSE degradation exceeds 5%')
    passed = not reasons
    return dict(passed=passed, decision='software_only' if smoke else
                'supports_future_estimated_state_integration_review' if passed else 'reject_close_estimator_variant',
                reasons=reasons, comparisons=comparisons, actor_updated=False,
                actor_integration_approved=False, policy_promoted=False, acceptance=False, physical_flight=False)


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def save_state(path, model, optimizer, updates, normalizers):
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=updates,
                    normalizers=normalizers, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                    python_rng=random.getstate()), path)


def run(args):
    out = args.out.resolve()
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires local CUDA')
    started = time.monotonic()
    deadline = started + (180 if args.smoke else 1800)
    stop_path = out/'STOP'
    source_policy = ROOT/'trainedModels'/f'{PROFILE.name}.zip'
    if sha(source_policy) != PROFILE.sha256:
        raise ValueError('Selected original flight policy changed')
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True, exist_ok=False)
        models, optimizers, updates_done, results = {}, {}, {}, {}
        checkpoints_saved = set()
        normalizers = None
        status, error = 'preparing', None
        try:
            input_hashes = {}
            metadata_paths = [WARMUP_RUN/'protocol.json', WARMUP_RUN/'clearance/warmup_check.json',
                              GAE_RUN/'protocol.json', GAE_RUN/'result.json', PRIOR_SPLIT]
            for path in metadata_paths:
                input_hashes[str(path)] = sha(path)
            warmup_protocol = json.loads(metadata_paths[0].read_text())
            warmup_check = json.loads(metadata_paths[1].read_text())
            gae_protocol = json.loads(metadata_paths[2].read_text())
            gae_result = json.loads(metadata_paths[3].read_text())
            if (warmup_protocol['source']['sha256'] != PROFILE.sha256 or warmup_protocol['warmup'] != 200000
                    or not warmup_check['exact_policy_weights'] or warmup_check['new_updates'] != 0
                    or warmup_check['steps'] != 200000 or not gae_result['source_unchanged']
                    or gae_result['status'] != 'bounded_trial_complete'
                    or gae_protocol['reset_seed_base'] != 11241000
                    or len(gae_result['iterations']) != 5
                    or sum(row['episodes'] for row in gae_result['iterations']) != 240):
                raise ValueError('Saved warmup/GAE provenance does not match the fixed experiment')
            warmup, warm_records, hashes = load_episodes(WARMUP_RUN/'clearance/training_trajectories',
                warmup_limit=200000, max_episodes=10 if args.smoke else None,
                deadline=deadline, stop_path=stop_path)
            input_hashes.update(hashes)
            gae, gae_records, hashes = load_episodes(GAE_RUN/'trajectories', expected_total=240,
                max_episodes=4 if args.smoke else None, deadline=deadline, stop_path=stop_path)
            input_hashes.update(hashes)
            if set(episode.seed for episode in warmup) & set(episode.seed for episode in gae):
                raise ValueError('Warmup and separate GAE cohort overlap')
            if not args.smoke and [episode.seed for episode in gae] != list(range(11241000, 11241240)):
                raise ValueError('Unexpected GAE cohort seed sequence')
            groups = np.repeat(np.arange(len(warmup)), [int(episode.mask.sum()) for episode in warmup])
            with np.load(PRIOR_SPLIT, allow_pickle=False) as prior:
                prior_groups = prior['episode_ids'][:len(groups)] if args.smoke else prior['episode_ids']
                if not np.array_equal(groups, prior_groups):
                    raise ValueError('Whole-episode order/usable rows differ from the prior motion audit')
            fit = np.array([index for index in range(len(warmup)) if index % 5 != 4], np.int64)
            heldout = np.array([index for index in range(len(warmup)) if index % 5 == 4], np.int64)
            if len(fit) < 8 or len(heldout) < 1:
                raise ValueError('Insufficient FIT or held-out complete episodes')
            for record in warm_records:
                record['split'] = 'heldout' if record['index'] % 5 == 4 else 'fit'
            for record in gae_records:
                record['split'] = 'gae_separate_no_fitting_or_normalization'
            normalizers = fit_normalizers([warmup[index] for index in fit])
            np.savez_compressed(out/'normalizers.npz', **normalizers)
            write_json(out/'episodes.json', dict(warmup=warm_records, gae_separate=gae_records))
            np.savez_compressed(out/'split.npz', fit_episode_indices=fit, heldout_episode_indices=heldout,
                                warmup_target_episode_ids=groups)
            total_updates = 2 if args.smoke else 3000
            schedule = episode_schedule(fit, total_updates)
            np.save(out/'episode_schedule.npy', schedule)
            sources = {}
            source_paths = [Path(__file__), ROOT/'tests/test_recurrent_estimator.py',
                ROOT/'experiments/faisal/docs/RECURRENT_ESTIMATOR_TRIAL_20261008.md',
                ROOT/'src/camera_history.py', ROOT/'src/contract.py', ROOT/'src/policy_runtime.py',
                ROOT/'tools/run_clearance_experiment.py', ROOT/'tools/validate_flight_policies.py', source_policy]
            for path in source_paths:
                relative = path.resolve().relative_to(ROOT)
                dest = out/'source'/relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
                sources[str(relative)] = sha(dest)
            protocol = dict(kind='Offline recurrent pre-action displacement estimator', smoke=args.smoke,
                initialization_seed=SEED, updates_per_variant=total_updates, batch_whole_episodes=8,
                episode_sampling='Same precomputed eight-episode schedule for both variants, with replacement',
                adam_lr=.0003, gradient_norm_clip=10., max_seconds=180 if args.smoke else 1800,
                full_episode_unroll=True, state_reset='Zero at each actual episode start; no truncated BPTT',
                inputs='Recorded public85; full camera and issued-action history equality checked per row',
                target='Before action t: body-frame (positions[t-1]-positions[t-2]) at heading[t-1], millimetres',
                axes=['forward', 'right', 'up'], first_two_rows='Fed to model, targets masked',
                target_is_velocity=False, truth_use='Supervised diagnostic targets only; never model inputs',
                normalizers='FIT episodes only; x all actual rows, y valid t>=2 targets only',
                variants=dict(feedforward='85->128 tanh->128 tanh->3', gru64='85->GRU64->linear3'),
                comparison='Equal episode/update budget; parameter count and FLOPs are not matched',
                split='Every fifth whole warmup episode: development-heldout, exact prior audit grouping; previously inspected in other diagnostics',
                separate_cohort='Preserved GAE240 episodes; no fitting, normalization, model selection or checkpoint search',
                selection='Final prescribed models only; no sweep or best checkpoint',
                gates='BOTH heldout and GAE: >=15% RMSE reduction all/blind, <=5% visible degradation',
                source_policy=dict(path=str(source_policy), sha256=PROFILE.sha256), sources=sources,
                input_sha256=input_hashes,
                artifacts={name: sha(out/name) for name in ('normalizers.npz', 'episodes.json', 'split.npz', 'episode_schedule.npy')},
                runtime=dict(python=sys.version, packages={name: importlib.metadata.version(name)
                    for name in ('torch', 'numpy', 'stable-baselines3', 'gymnasium')},
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), device='cuda:0'),
                actor_updated=False, acceptance=False, physical_flight=False)
            write_json(out/'protocol.json', protocol)
            print('PROTOCOL', str(out), flush=True)
            device = torch.device('cuda:0')
            cohorts = {'fit': (warmup, fit), 'heldout': (warmup, heldout),
                       'gae_separate': (gae, np.arange(len(gae)))}
            for variant in VARIANTS:
                seed_everything()
                model = (FeedForwardEstimator() if variant == 'feedforward' else RecurrentEstimator()).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
                if not all(parameter.device.type == 'cuda' for parameter in model.parameters()):
                    raise RuntimeError('Estimator parameters are not all on CUDA')
                models[variant], optimizers[variant], updates_done[variant] = model, optimizer, 0
                model.train()
                for update, batch_ids in enumerate(schedule, 1):
                    ensure_budget(deadline, stop_path)
                    x, y, mask, lengths = make_batch(warmup, batch_ids, normalizers, device)
                    loss = masked_mse(model(x, lengths), y, mask)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite estimator loss')
                    optimizer.zero_grad()
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Non-finite estimator gradients')
                    optimizer.step()
                    updates_done[variant] = update
                    if update % 100 == 0 or update == total_updates:
                        progress = dict(variant=variant, updates=update, prescribed_updates=total_updates,
                                        last_batch_loss=float(loss.detach()), elapsed_seconds=time.monotonic()-started)
                        write_json(out/'status.json', progress)
                        print('PROGRESS', json.dumps(progress), flush=True)
                save_state(out/f'{variant}.pt', model, optimizer, updates_done[variant], normalizers)
                checkpoints_saved.add(variant)
            # Neither final cohort is read for model choice or the other model's fitting.
            for variant in VARIANTS:
                results[variant] = {}
                for cohort, (episodes, indices) in cohorts.items():
                    results[variant][cohort] = evaluate_estimator(models[variant], episodes, indices,
                        normalizers, out/f'{variant}-{cohort}-predictions.npz', device,
                        deadline=deadline, stop_path=stop_path)
                    write_json(out/'results.json', results)
                    print('FINAL', variant, cohort, json.dumps(results[variant][cohort]), flush=True)
            # Input changes during fitting invalidate the comparison rather than silently
            # reinterpreting the final estimator against a different dataset.
            for path, digest in input_hashes.items():
                ensure_budget(deadline, stop_path)
                if sha(path) != digest:
                    raise ValueError(f'Input changed during the run: {path}')
            status = 'complete'
        except BoundedStop as exc:
            status, error = 'bounded_stop', str(exc)
        except Exception as exc:
            status, error = 'failed', f'{type(exc).__name__}: {exc}'
        finally:
            for variant, model in models.items():
                # Completed variants retain their own end-of-training RNG. Do not
                # overwrite FF's RNG with the later GRU/evaluation process state.
                if variant not in checkpoints_saved:
                    save_state(out/f'{variant}.pt', model, optimizers[variant], updates_done[variant], normalizers)
            source_unchanged = sha(source_policy) == PROFILE.sha256
            decision = gate_estimator(results, complete=status == 'complete',
                                      source_unchanged=source_unchanged, smoke=args.smoke)
            if error:
                decision['reasons'].append(error)
            decision.update(status=status, updates_completed=updates_done,
                            elapsed_seconds=time.monotonic()-started, source_policy_unchanged=source_unchanged)
            decision['evidence'] = {str(path.relative_to(out)): sha(path)
                for path in sorted(out.iterdir()) if path.is_file() and path.name not in ('STOP', 'decision.json', 'DONE')}
            write_json(out/'decision.json', decision)
            if status == 'complete':
                (out/'DONE').write_text('Offline estimator comparison complete; no flight policy changed.\n')
            print('DECISION', json.dumps(decision), flush=True)
        if status == 'failed':
            raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
