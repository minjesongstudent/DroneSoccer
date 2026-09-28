# Learned-policy recurrent initialization

The centered RL-only trial stopped at65,536 steps, zero training passes (442 episodes:391 timeouts,51 hits), zero centered validation passes. No unchanged extension.

Collected80 fit and20 holdout authentic full flights from preserved StandardSourceSAC, alternating centered/varied starts. Teacher passed72/80 fit and18/20 holdout. Failed flights retained. Inputs are119 public camera/command-history values; targets are learned teacher commands. No privileged simulator features or scripted controller. This is supervised imitation initialization, not pure reinforcement learning.

Verified batched whole-sequence output matches actual recurrent stepwise inference for2x12 steps at1e-6 tolerance before fitting. Zero state corresponds to true episode starts; no invented midflight histories. CUDA actor-only Adam, masked padding, independent complete episodes. Critic remains untrained. Initial preflight training-mode issue corrected before first successful optimizer update and recorded.

At100 supervised updates: centered25/25,varied4/25. At300: centered25/25,varied20/25. Full-front/general/delayed missions all0/25. Stopped at preset transfer gate; no simulation acceptance claim. Fit action MSE.07257 vs heldout.08402; action fit alone is not task performance.

Next: inspect full-task trajectories and design bounded RL fine-tuning with single-pass rehearsal and retention checks; account for fresh critic and reward normalization. Do not start training simply by treating the supervised update count as environment steps. No7M independent acceptance seeds used. All historical models preserved.
