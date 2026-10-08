# Full-clearance reward comparison - 7 October 2026

Completed: both candidates were stopped for regression and rejected. See
`POLICY_CLEARANCE_RESULT_20261007.md` for the results. No trainer remains active.

The user authorized implementing and running the proposed comparison. It was
launched on the home RTX 4070 Ti at 2026-10-06 16:09 UTC (7 October Sydney).
This is simulation learning work. No hardware, publication or recurring automation
was started. No model has been promoted or accepted for flight.

## Implementation and verification

`src/clearance_training.py` wraps the unchanged group simulator with the original
85-input camera history. Only the treatment's reward timing changes: remove the
40-point centre-crossing bonus, and pay it once when the full body clears the far
face without contact. The control keeps the original reward. Both keep level 1,
the hanger, randomization, collision geometry, penalties and 600-step episodes.

The wrapper takes actor inputs only from public camera observations and actually
issued learned actions. Truth-based passage, position and heading diagnostics are
stored separately. A crossing on the timeout step is detected from the native
pass counter, since the native outcome label becomes `timeout` on that step.

120 relevant repository tests passed, including 20 new reward, profile, reset,
seed-exclusion, stop-boundary and trainer-lock tests. Two original Gymnasium
warnings concern the existing unbounded observation space. CUDA smoke trials
verified 32 actual gradient updates per arm, finite changed actor weights,
reloadable model/optimizer/environment state and the preserved source checksum.
The final smoke also verified identical warmup observation/action buffers between
the two arms. Smoke results are software checks, not learning evidence.

The first smoke exposed an algorithm-seed/layout-seed mix-up before any simulated
movement. It was fixed by explicitly queuing the declared FIT seed in the vector
environment; policy and minibatch RNG retain their separate algorithm seed. The
failed smoke and its error artifacts are preserved.

## Frozen run

Full evidence: `D:/DroneSoccerData/runs/clearance-reward-20261007`.
Console logs are the adjacent `.stdout.log` and `.stderr.log` files.
The protocol and exact source/model hashes were saved before evaluation.

- Sequential arms: original reward control, then full-clearance reward treatment.
  An OS lock excludes concurrent instances of this experiment; the process
  inventory was empty before launch.
- Both start from unchanged `seenflag-s1-level1`, SHA256
  `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.
- CUDA SAC, learning rate 0.00003, batch 256, gamma 0.995, tau 0.005,
  inherited actor/critic/entropy optimizer state. One gradient per learning
  transition. Replay capacity 250,000 retains the entire bounded experiment arm.
- Per arm: 200,000 transitions collected by the learned stochastic policy with
  no optimizer updates, then at most 50,000 learning transitions. No uniform
  random warmup or scripted movement. Warmup weights must match exactly, and
  the paired observation/action buffers must match before treatment optimization.
- At most two hours per arm, including intermediate checks. Final evaluation
  is separately bounded to 100 episodes per suite with 600 steps per episode.
- FIT reset seeds: 11,000,000 plus episode index. Algorithm seed: 11,500,000.
  Existing development seeds and reserved acceptance seeds are excluded.
- Retention checks after 1,000, 10,000, 20,000, 30,000, 40,000 and 50,000 new
  learning transitions. The early check was frozen before the first run to
  catch regression before spending the entire first 10,000 updates.
- Each check uses the first 25 episodes of each existing development suite.
  Stop an arm if any suite loses more than 2 first-pass episodes, gains more
  than 3 crashes, or loses more than 3 clean repeat episodes relative to the
  unchanged source. These are development stopping guards, not relaxed mission
  acceptance criteria. A stopped candidate remains preserved and receives the
  full development evaluation.
- Each checkpoint retains model/optimizers, replay buffer, RNG, environment and
  collector state. Every simulated training episode retains actual observations,
  actions, rewards and separate truth diagnostics, including a partial last
  episode. No automatic resume or candidate promotion is performed.

`status.json` records the current stage. `result.json` and `REPORT.md` are written
when the comparison completes or fails. Each arm has detailed outcome summaries
and saved evaluation trajectories. Creating a file named `STOP` inside the run
directory requests a saved stop and prevents the next arm from starting.

To reproduce, use the saved `source` directory with the recorded dependency
versions and invoke `python -m tools.run_clearance_experiment --out <new-directory>`.
The output directory must not already exist. The source checkpoint is verified
by hash and CPU training fallback is refused.

The original policy, all historical artifacts and group master remain untouched.
Final simulation acceptance still requires sustained fixed validation and the
independent fresh-seed acceptance test. Physical flight remains unvalidated.
