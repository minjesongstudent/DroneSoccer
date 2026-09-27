# Short soft-Retrace value-learning experiment

The stopped 20k return-practice run underestimated crash costs despite small one-step errors. Adding separate simulator context to the critic did not help the controlled comparison: crash-terminal MAE was 22.51 versus 18.59 with public inputs. That option was not promoted.

This separate experiment keeps both actor and critics on the same 100 camera/history/command-memory/task inputs. It changes how recorded transitions train the critics: at most eight consecutive transitions contribute, with trace lambda 0.95. The actor, frozen retention reference, reward v3, fixed entropy coefficient, mixture, and acceptance criteria stay unchanged.

For target value B=min(target twin Q) and soft next value V=E[B(next,a)-alpha log pi(a|next)], the residual is delta=r+gamma*(1-terminal)*V-B(current,stored action). The trace target is B at the anchor plus discounted residuals. Successor weights are lambda*min(1, pi(action)/behavior(action)); the anchor has no importance coefficient. There is no extra entropy subtraction for the queried current action. A one-transition trace, or lambda zero, reduces to the ordinary SAC target.

This is an experimental adaptation of the [Retrace residual operator](https://papers.nips.cc/paper/2016/file/c3992e9a68c5ae12bd18488bc579b30d-Paper.pdf) to the [SAC soft Bellman target](https://arxiv.org/html/1812.05905v2). The exact-state theory does not establish convergence for this partial observation, neural approximation and twin-min target setup.

## Frozen diagnostic prerequisite

`diagnostics/retrace_critic_comparison.py` reuses the saved authentic 100-flight dataset from the asymmetric comparison. It collects no new flight actions and uses no privileged context. The failed 20k actor is frozen solely for diagnosis; a live trial must start from the preserved good source instead. Both fresh public critics have identical initial weights. They use the same saved training anchors and paired first-step bootstrap action samples, with separate randomness for additional trace terms. Each gets 5,000 CUDA updates, with zero actor updates. The verified fixed actor is identical to the behavior actor, so diagnostic importance ratios are exactly one.

Keep training and development flights separate. Compare true terminal crash returns, excluding the queried action's entropy; timeout prefixes are not complete-return truth and retain a separately reported bootstrap tail. The scoped predeclared evidence gate requires at least four distinct held-out terminal source flights, at least 20% crash-return MAE reduction, and no material deterioration in supported phase estimates. Unsupported phases remain explicitly unassessed. Passing would support only a bounded behavioral trial, not successful laps or broad critic accuracy. Preserve the failed and intermediate results.

Completed result: the prerequisite passed. At 5,000 updates the one-step control had crash-terminal MAE 18.589965, versus 8.829801 for soft Retrace, a 52.5% reduction. Evidence comprises 16 side-return crash continuations from eight distinct development-source flights. The 16 front-return timeouts do not establish terminal accuracy. Bias changed from +18.59 to -8.83, and error was lower at 2,500 than 5,000 updates, so longer calibration is not justified. Both saved actors and references matched the source exactly, actor optimizers remained unchanged, and actor update counts were zero. The one-step control's entire critic state matched the previous public baseline byte for byte. Results, protocol, immutable inputs, source snapshot and independent review are at `evidence/diagnostics/retrace-comparison-v1`.

## Production collection and replay, if warranted

The sampler must cache the exact generated action, original pre-tanh Gaussian sample, and joint behavior log density before stepping the environment. Reconstructing inverse tanh from a rounded saturated action is not equivalent. Current-policy densities use those saved pre-tanh samples for clipped importance ratios. Refilling replay still uses the learned actor; there are no teacher actions.

Task-balanced sampling chooses anchor transitions 50/50 single-pass/lap, then follows their actual recorded successors. Insertion, episode and collection-stream IDs prevent joining overwritten slots or different episodes. Every timeout ends the trace, while its true final observation remains eligible for value bootstrap. Disk resume, external reset and interrupted collection create new streams without falsely declaring a terminal failure. Old replay without exact behavior metadata is incompatible.

Supported collection is intentionally explicit: one environment, no external action noise, no gSDE, no observation normalization, normalized four-dimensional actions, and zero random-action warmup. The policy uses CUDA; physical simulation remains local.

## Bounded behavioral protocol

Only after reviewing the diagnostic and passing lifecycle tests: fresh transfer from `runs/return-route-40k-20260927/best_model.zip`, same authentic banks and five-way mixture, optional `--retrace-critic`, new run name and output on D:. Maximum 40k, fixed validation every 5k, unchanged 20k/40k return-development stops. Keep only one training worker active. The full-mission candidate and independent 7M acceptance gates remain unchanged and separate from intermediate return skill.

Simulation is still incomplete until those gates pass. No physical flight is validated by critic calibration or simulation tests.

## Reproduction

The integrated code passed 403 tests in 34.59 seconds, with only the two pre-existing infinite observation-bound warnings. Run tests with `..\DroneSoccer\.venv\Scripts\python.exe -m pytest -q --basetemp D:\DroneSoccerData\test-temp\UNIQUE_NAME`.

The reviewed production launch, from this directory, is:

```powershell
..\DroneSoccer\.venv\Scripts\python.exe -m src.live --steps 40000 --eval-every 5000 --name return-retrace-20260927 --port 8779 --runs-dir runs --source-checkpoint runs/return-route-40k-20260927/best_model.zip --source-config runs/return-route-40k-20260927/config.json --return-library return-libraries/train-v1 --return-validation return-libraries/development-v1 --retrace-critic
```

Do not launch a duplicate. Inspect the handoff and live status first. The frozen diagnostic's failing actor or fitted critic is not transferred to production.

The command was launched successfully on the RTX 4070 Ti, direct execution session 35270. Port 8779 is the live viewer; outputs and an exact verification record are in `runs/return-retrace-20260927`. Baseline results reproduced the source: 24/25 centered, 23/25 varied, zero repeated-pass mission successes; development continuations had 0/50 new passes, 0/50 safe returns and 1/50 crashes. The archived initial actor/reference match the source, and the source snapshot records the tested code. Later outcomes must be read from that run's saved evaluations.
