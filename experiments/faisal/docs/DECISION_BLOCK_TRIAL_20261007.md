# Learned decision-block diagnostic

The independent single-action audit found unstable action rankings despite repeated continuations. This bounded diagnostic tests whether outcomes following longer learned prefixes are more distinguishable. It does not update or promote an actor.

## Reproduction

From the repository, using the project Python environment:

```text
python -m tools.audit_decision_blocks --out D:/DroneSoccerData/runs/decision-block-20261007
```

The source is `critic-independent-families-20261007`: the same 64 saved state families, four candidate first actions and frozen 121-input adapter. The model SHA256 is `f504022e10de757e5d50f0e6737981ffde9bb89b177690e243f6e365748f6ed4`.

Each candidate starts with its saved learned first action at the identical observation, then samples up to eleven additional actions from the policy at the observations actually received. The resulting state is cloned for six learned continuations. Prefix and continuation RNG seeds, state snapshots, observations, actions, rewards, log probabilities and positions are preserved. The run has a 30-minute cap and at most 1,536 continuation rows. This is a CPU diagnostic, with no GPU trainer active.

## Checks and status

The smoke run completed one state, four prefixes and two replicas. All eight recorded block/tail lengths matched their outcome records; actions were finite, and every continuation's first observation exactly matched its prefix's next observation. The full run subsequently reached 37 states and 888 actual tails without stderr output. Completion and outcome analysis remain pending; inspect `DONE`, `status.json` and logs before further work.

## Prespecified interpretation

- Keep whole state families together: indices modulo five equal to four are held out, preserving the source split.
- Compare finite soft returns, timeout-bootstrapped soft returns and mission outcomes separately.
- Report all families and, separately, families where all four prefixes survive. A terminal prefix is one realized event; its repeated output rows are not independent flights and can artificially inflate apparent ranking stability.
- Replicate dispersion estimates uncertainty conditional on a realized prefix. It does not estimate the expected value of a fixed open-loop action sequence. Every movement was selected by the learned policy at actual observations.
- Six action contrasts from one family are dependent. Descriptive agreement or a two-standard-error contrast is not an adjusted significance test.
- Better conditional separation alone does not justify selecting lucky prefixes, changing the deployed actor, or extending this run until a desired result appears.

The original 85-input policy remains selected. Simulation acceptance is not achieved, reserved acceptance seeds remain unused, and physical flight remains unvalidated. No hardware, publication or group-master changes were made.
