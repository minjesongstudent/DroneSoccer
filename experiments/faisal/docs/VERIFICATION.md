# Publication verification

Verified on 2026-09-27 using Python 3.12.13, PyTorch 2.11.0+cu128 and the pinned
packages in `requirements-tested.txt`, on an NVIDIA GeForce RTX 4070 Ti.

| Check | Result |
| --- | --- |
| Experiment test suite (`python -m pytest -q`) | 144 passed in 13.12 seconds |
| Existing Gymnasium observation-bound warnings | 2 non-failing warnings |
| Packaged C520k, C670k and C1M evaluation CLI smoke check | All loaded and evaluated on CUDA |
| Evaluation CLI invoked from repository root | Correct isolated experiment modules imported |
| Model metadata export | Every non-metadata archive entry preserved byte-for-byte |
| CUDA save/reload and replay continuation lifecycle | Covered by passing tests |
| Selected policy weights versus exported artifact | Unchanged; original/export hashes in `models/manifest.json` |

The CLI smoke check used one episode per suite per model to verify execution,
not to estimate success rates. The larger recorded selection experiment used
675 episodes; see [RESULTS.md](RESULTS.md).

The experiment omits the unused historical A/B/C comparison viewer. It includes
six portable checkpoint-CLI tests instead. The separate group implementation at
repository root is unchanged by this publication and was not retested here.

Software checks do not establish real-world calibration, flight safety or
successful transfer to physical hardware. No real flight was performed during
publication, and the existing local trainer was not restarted or modified.
