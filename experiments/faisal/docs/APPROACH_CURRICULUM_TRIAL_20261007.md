# Bounded approach curriculum

Earlier failures show lateral misalignment before the near-ring band. The guard penalty failed its fresh development screen and is closed. This trial changes the training start distribution, not that penalty's coefficient.

Start from the original 85-input model on CUDA. Every other fresh FIT seed uses the unchanged randomized full-task reset. The remaining seeds start 45–65 cm in front of the same randomized ring with lateral offset up to 18 cm, vertical offset up to 6 cm and heading within 15 degrees of the ring center. Reset positions must remain within the existing cage and altitude limits. Ring size, hanger and randomized physics are preserved. These are declared training resets, not teleportations during flight.

All observations come from the camera simulator at the actual reset; action history starts at zero, with the normal repeated initial camera frame. Every later action is sampled from the learned actor. All episodes retain native crash endings and the 600-step limit, including returns and repeat passages. No scripted movements, new actor truth inputs, changed collision checks or weakened success criteria are used.

Reproduction:

```text
python -m tools.run_onpolicy_trial --out D:/DroneSoccerData/runs/approach-curriculum-20261007 --approach-curriculum --actor-lr 1e-5 --episodes-per-iteration 48 --actor-epochs 1
```

Bound: five iterations of 48 complete episodes, one actor epoch per iteration, one-hour collection/update cap, unchanged retention checks after each iteration. Uses finite discounted full-clearance returns with a separate value baseline, not the failed guard penalty or SAC action-value gradients. The source trust guard covers 8,192 FIT observations. Fresh reset base is 11,240,000; no development layouts become training data.

Six focused tests passed, including exact native odd-seed reset parity and valid fresh public history across 50 curriculum starts. CUDA smoke completed one update with source unchanged; smoke outcomes are not efficacy evidence. Models, sources, trajectories, returns, optimizer state and RNG state are preserved.

Final development evaluation uses the unchanged three suites, 100 episodes each. Stop and reject for retention regression or no meaningful randomized repeat/crash improvement. Do not extend the curriculum trial unchanged or judge it by its easier training starts. Only a promising candidate proceeds to paired fresh development, sustained validation and the separate reserved acceptance test. Physical flight remains unvalidated.
