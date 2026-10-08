# Action-likelihood numerical check — 8 October 2026

This bounded check did not meet its predeclared threshold for a corrective actor rerun. No policy was trained, changed or promoted. Do not reopen the failed ring-position trial on this basis.

The on-policy helper reconstructs Gaussian samples from recorded, rounded commands and clamps commands to +/- (1-1e-6). This loses information near saturation. The last ring-position trial contained only27 clamped action components among256,084 components, including17 exactly at a boundary. Their significance cannot be inferred from counts alone.

Using the unchanged, hash-bound source actor on8,192 already saved FIT guard observations, the check drew32 samples per observation with fixed RNG11602000 on CUDA. No simulator was stepped, and these offline samples were never issued as movements. It compared the exact sampled Gaussian mean-score with the score reconstructed from the rounded command, normalized by the policy's standard deviation.

- Relative RMS score error: **0.981288%**, below the predeclared greater-than1% threshold.
- Maximum absolute axis-mean normalized score bias: **0.000084997**, below the predeclared0.001 threshold.
- Offline sampled components affected by clamping:120 of1,048,576.

These measurements do not establish the cause of flight failures or prove the numerical approximation harmless in every state. They do not support changing the controller and restarting a failed experiment. The threshold remains unchanged despite the RMS result being close to it.

The initial calculation used `torch.atanh`. A saved-array verification then used the exact installed SB3 inverse transform; the largest reconstructed-latent difference was4.77e-7 and the decision was unchanged. It performed no new model inference, sampling or flight. Both results are preserved rather than overwritten.

Evidence: `D:/DroneSoccerData/runs/action-likelihood-audit-20261008`, including `protocol.json`, all sampled arrays, source/model/input hashes, `audit_source.py`, `result.json` and `verification.json`. CUDA inference took1.718 seconds under the project lock with0.15 allocator fraction and strict float32. The verification records Python/package versions. Original result SHA256: `5570d9630cdf933ca95249d514e87501384ceac685c993412789984f73500be6`.

This numerical investigation is closed. Do not repeat it with extra draws, a lower threshold or outcome-selected observations. No new trainer is active. The original policy remains selected; simulation acceptance is incomplete and physical flight remains unvalidated.
