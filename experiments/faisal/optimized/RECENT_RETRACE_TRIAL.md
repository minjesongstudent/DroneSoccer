# Recent and all-lap replay trial

The prior public soft-Retrace run stopped at 20k with no return progress. A read-only audit showed substantial divergence between current and stored action distributions: lap traces averaged 7.88 valid transitions but only 1.76 total discounted residual weight; back-visible lap traces retained 1.33. Frozen-actor calibration had importance ratios of one, unlike actual learning.

This treatment changes only the anchor sampling distribution. Half of each batch remains uniformly sampled from all retained single-pass experience. The lap half splits equally: half from all retained lap transitions and half from the latest 2048 lap insertions. Older data remains eligible; no observations, rewards, phases, outcomes or successful examples influence recency. Each chosen anchor follows its real saved successors, retaining exact importance correction, terminal/truncation handling and stream boundaries. Actor inputs remain the same 100 public camera/history/command features. All flight actions remain learned.

## Prerequisite, declared before the probe

On the stopped20k model/replay, with seed928173 and4096 anchors, require at least25% improvement in mean lap residual-weight sum, no deterioration in hidden/back-visible lap groups, and at least128 sampled anchors in each of those groups. This measures temporal target contribution, not Q accuracy or success. The protocol and immutable inputs/results are in `evidence/diagnostics/return-retrace-audit/RECENCY_PROTOCOL.md` and `recent-replay-probe-20000.json`.

Result: lap weight sum increased from1.76117 to2.55979 (+45.34%). Hidden-ring laps increased1.60980 to2.56860 (1278anchors); back-visible laps increased1.32631 to1.90649 (635anchors). No model parameters or replay files changed. The scoped prerequisite passed. It does not prove that stale replay caused all failures. Oversampling recent experience could increase correlation or emphasize poor recent behaviour, which is why uniform older lap coverage and unchanged flight gates remain essential.

## Bounded production protocol

Fresh transfer from the preserved good `runs/return-route-40k-20260927/best_model.zip`, SHA256 `884834222681c545623110f8b00fe6b7def32322ec069906d728a2373dd39337`. Preserve its current actor and original frozen retention reference; initialize fresh critics, optimizers, replay and counters. No weights from failed20k trials are transferred. Keep5000 refill transitions,5000 critic calibration updates, actor/critic rates, fixed entropy, rewardv3, authentic banks, five-way reset mixture, retention criteria and independent acceptance protocol unchanged.

New option `--recent-retrace`; version `recent-and-all-lap-soft-retrace-v1`. Resume requires the saved sampling version and `RecentLapReplay` contract; it cannot silently switch uniform/recent sampling. Only one training worker, on RTX4070Ti CUDA, outputs on D:. Maximum40000 new steps with fixed evaluation every5000 and the same automatic20k sustained-progress and40k new-pass gates. Independent7M acceptance is not used for this prerequisite or tuning.

```powershell
..\DroneSoccer\.venv\Scripts\python.exe -m src.live --steps 40000 --eval-every 5000 --name recent-retrace-20260927 --port 8780 --runs-dir runs --source-checkpoint runs/return-route-40k-20260927/best_model.zip --source-config runs/return-route-40k-20260927/config.json --return-library return-libraries/train-v1 --return-validation return-libraries/development-v1 --recent-retrace
```

Inspect current handoff/runtime before launching to avoid duplicate workers. This document specifies the reviewed treatment, not proof that it has started or succeeded. Physical flight remains unvalidated.

Validation before launch: all409 tests passed in34.70seconds, with only two pre-existing Gymnasium bounds warnings. Focused13 tests passed in19.91seconds. These include chronological sampling across circular overwrites, old-lap coverage, exact task balance, true terminal/timeout sequence boundaries, incompatible replay rejection, and actual CUDA live training/save/resume with the new sampler. No success criteria were modified for production.
