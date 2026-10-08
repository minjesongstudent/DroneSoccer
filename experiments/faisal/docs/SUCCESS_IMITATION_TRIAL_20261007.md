# End collision-replay trials; bounded successful-trajectory imitation

The critic-only risk replay ablation failed retention at 1,000 updates. Final aligned/nominal/randomised first-passage counts were 86/79/68, crash counts 14/26/57, return counts 86/79/66 and clean-repeat counts 86/74/43 (100 episodes each). Original remains retained. Restoring uniform actor sampling did not remove the regression, and collision replay is not extended.

A read-only audit of 66 heldout actual terminal transitions found critic RMSE against realized terminal reward: source 16.80, uniform adapted 17.27, collision-adapted 9.10. Bias fell from +13.14 to -0.34 with collision adaptation. This excludes bootstrap and entropy at true terminal, but remains realized-outcome calibration under potentially aliased camera histories, not exact expected-value ground truth. Better terminal prediction did not yield better flight. Evidence is `D:/DroneSoccerData/runs/critic-terminal-audit-20261007`, including per-transition values, indices, hashes and audit source.

The original warmup contains 558 complete episodes (199,929 transitions), 403 with a full passage, 334 with repeated passages and 316 with a crash. There is enough authentic successful learned behavior to test an alternative update without critic action gradients.

New experiment: source-constrained maximum-likelihood imitation of the original policy's successful FIT episodes. Select only complete 600-step episodes with at least two full-body clearances, zero wall contacts, and timeout survival. Every fifth episode remains held out. Verify replay boundaries against logged episodes. This selects 117,000 actual transitions from 195 successful FIT episodes. No validation trajectories, scripted actions, lab hand-carry data or fabricated observations enter training.

The actor starts from the unchanged original 85-input model; the critic is unused and unchanged. Loss is negative log likelihood of actual learned actions plus source Gaussian KL and boundary penalty weight 0.1. Numerical likelihood clamps recorded actions within 1e-6 of the tanh bounds; evaluation movements are unmodified learned outputs. Success filtering intentionally biases the offline training distribution and may overfit easy layouts; unchanged development guards can reject it.

Budget: 2,000 actor updates maximum; checks at 100/500/1,000/2,000, 25 episodes per suite, final 100 per suite; 30-minute training cap plus bounded evaluation. Actor learning rate 1e-6; hard source KL mean <=0.01, maximum <=0.1; exact actor and optimizer rollback and stop after 100 consecutive rejected proposals. Guard observations cover all FIT behaviors, including failures. Source model, dataset hashes, selected indices, implementation, model/optimizer and RNG checkpoints preserved. No automatic promotion.

30 focused checks passed; the CUDA smoke completed updates, model reload and learned evaluation. Selection tests exclude heldout, partial, short, crashing and wall-contact episodes. No physical validation or acceptance claimed; reserved 7M seeds remain unused.

```text
python -m tools.run_success_imitation --out D:/DroneSoccerData/runs/success-imitation-20261007
```

The output directory must be new. Run evidence lives on D:; group master and remote branches remain unchanged.
