# Monotonic reference-distance repair: bounded trial

The reset curriculum retained24/25centered and23/25varied passes and improved shorter lessons, but stopped20k with0/50new/safe returns. Original frozen reference remains valuable and must not be replaced silently.

Frozen20k diagnostic128 independent optimizer proposals used the saved actor, Adam and critic, restoring them before each proposal.99/128 batches already exceeded reference KL0.1. The original absolute rule accepted30;97 additional proposals lowered an excessive batch KL by at least1% but were rejected because they remained above0.1. Mean pre/post KL0.794/0.525. This establishes rejected corrective proposals, not improved flight. Critic was held fixed; production updates it first, so exact rates need not repeat. Script/result evidence/diagnostics/anchor-proposal-audit.py/.json.

New policy version monotonic-reference-kl-repair-v1 permits a proposal only if all parameters/KL are finite and either post-KL<=0.1 OR pre-KL>0.1 and post-KL<=0.99*pre-KL. Original reference, penalty coefficient10, KLmask, actor/critic learning rates, exact rollback of model and Adam, replay and entropy remain unchanged. Already admissible batches cannot cross the existing limit. Above-limit batches must make measurable progress toward the same reference. Repair count is saved and displayed. This is a batch constraint, not a guarantee on all possible states; actual passing validation remains essential.

Fresh run planned anchor-repair-20260927, port8783, D:storage. Transfer the preservedgood100D10k actor/originalreference with zero appended columns, fresh critics/replay. Same reset curriculum and fixed development suites as reset-lessons-20260927; only proposal acceptance changes. Maximum40k,5k checks, unchanged20k/40k authentic-return gates; reset scores cannot bypass failure stops. All full600-step/two-pass acceptance criteria and sealed7M independent test remain unchanged. No scripted flight controller, demonstrations or private actor inputs. A production launch requires full tests and CUDA lifecycle checks first; implementation success is not flight success.

Launch verification:455 tests passed in57.02s,two existing non-failing Gymnasium warnings. Focused23 tests include CUDA live save/resume and wrong-policy/changed-contract rejection.

Outcome:failed20k,0/50return success,27/50return crashes;not selected. See evidence/diagnostics/anchor-repair-result.md. Scheduler v2 now stops mastered reset-lesson regressions; do not resume this old experiment under altered rules.
