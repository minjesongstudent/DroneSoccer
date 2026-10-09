# Correcting the model mix-up

The old lab ZIP and all nine uploaded October 9 recordings identify the baseline SHA256:

`8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`

The bytes match `trainedModels/seenflag-s1-level1.zip` from Git commit `9d89c26`, authored by **Tom Storey**, October 6, 2026. This was the team baseline, not the PC-trained composite. The earlier assistant statement that it was the checkpoint trained on Faisal's PC was incorrect.

The PC UI at port 8794 reads run `parallel-credit-20261008`. Its policy uses the baseline plus:

| Component | SHA256 |
|---|---|
| Frozen PC GRU | `b48d6b26136a5e2e6cbfcd615a1a819eb801b495fba4cfa38130c0e2da54f554` |
| Full horizon final correction | `eba7030c293d1b56edf3026aba4b449d426e307ad946bde83a0604a418547006` |
| Local credit final correction | `2223a2cc117565f9f33dd3a3009cc818ac5ae826dc7b14f03266c3921beebb8e` |
| Mission stopped round 02 correction | `db4f747ac244f6871c6823512b59c46d25369a3e3296c3023bfc9ce3ffee224a` |

Each correction has 1,032 trained parameters (4 x 258); the base actor and estimator remain frozen. Full horizon and local credit correction norms are about 0.003504 and 0.003563. The small magnitude does not make them identical to the baseline. Their actual learned actions differ.

Both the baseline screen and full-horizon final screen happened to yield 48 clean repeats and 52 crashes per 100 randomized trials. Equal aggregate counts do not establish identical weights. The new lab runner checks all three hashes and records the complete identity in its UI, metadata, per-command logs and comparison summary. It does not fall back to the baseline if a component is missing or wrong.

This branch publishes the requested full-horizon final checkpoint for experimental testing only. It remains disqualified by the unchanged cohort criteria. No model was retrained or promoted as part of repository organization.
