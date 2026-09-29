# Bounded reinforcement-learning trial from checkpoint50

Checkpoint50's fixed validation had3/25 front-lesson full missions,25/25 centred
and22/25 varied single passes, but0/25 general and delayed full missions. A
follow-up reproduced0/9 FIT full missions (7first passes) and the selected
validation seed5000014's three passes at27/318/590 with survival to600.
These are diagnostic results, not independent acceptance or a deployable claim.

Because continued imitation lowered MSE while losing complete flights, this
trial tests the existing RL reward on the policy's own trajectories instead.
No validation trajectory is inserted into training. The same natural-start
training mixture samples seeds below1M:25% centred single,25% varied single,
50% front laps. No scripted movement or hidden actor observations.

Run: lap-onpolicy-recurrent50-20260928/run.py. CUDA RTX4070Ti, maximum16384
transitions, evaluations0/4096/8192/16384. First4096 freeze actor and calibrate
the existing critic against a fresh reward-only normalizer; observations are
not normalized. Actor hash verifies calibration does not change it. Afterwards
PPO lr3e-5, clip.1, targetKL.003. Stop if centred<22/25 or varied<20/25.
Original119D observation and600-step full mission criteria unchanged.

Source snapshot, model hash, starting model/evaluation, subsequent checkpoints,
normalizers and optimizer-containing model saves are preserved. Only this one
trainer is active. Viewer and Mac exports remain the preserved baseline500.
No physical flight, reserved examples,7M acceptance test or publishing.

RESULT: stopped8192 atfirst actor-update check. Critic-only4096 preservedactorhash and exact25/22/3/0/0 suitecounts. At8192: centred25/25,varied19/25,lesson0/25,general/delayed0/25. Varied retention threshold20failed, so16384notrun. No active trainer. No model promotion or unchanged extension. Analyze actor drift/calibration before another RL run.
