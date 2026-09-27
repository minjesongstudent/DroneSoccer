# Optimized ring-flight experiments

Current snapshot: 27 September 2026. One learned SAC policy controls forward/right/up/yaw commands inside a 2.05 m cage. Reliable repeated laps are **not yet solved**. Physical flight is unvalidated. All runs are stopped; the latest metric-ring trial stopped at15k after regression.

See [development process](DEVELOPMENT_PROCESS.md), [measured results](RESULTS.md), and [saved evidence](evidence/summary.json). Earlier A/B/C and unified work remains in the parent directory. This package has its own `src`; run commands from `experiments/faisal/optimized`.

## Setup and tests

Use Python3.12 and a CUDA-capable PyTorch installation. The tested workstation uses RTX4070Ti and PyTorch2.11.0+cu128. Install requirements from requirements-tested.txt, then:

```powershell
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pytest -q
```

Development verification:479 tests passed, with two existing non-failing Gymnasium infinite-bound warnings. Some integration tests require the original local return banks and run layout; they skip when those artifacts are absent. See PUBLICATION_VERIFICATION.md for the exported package test result.

## Reproduce a fresh metric trial

The included source model retains single-pass skill. The metric15k checkpoint is an explicitly failed diagnostic artifact, not a recommended controller. Archive hashes and unchanged-weight verification are in models/manifest.json. Full replay buffers and large trajectory logs remain in local archives; this package includes checkpoint evaluations, configs and compressed source snapshots for seven runs. Config paths labelled local-artifact are provenance placeholders, not runnable paths. Historical trial notes may refer to diagnostic files retained only in the local archive.

Generate new authentic return banks from complete source-policy flights; do not invent observation histories:

```powershell
python -m src.front_return_library --checkpoint models/route-source-10k.zip --output return-libraries/train --split train --episodes 64
python -m src.front_return_library --checkpoint models/route-source-10k.zip --output return-libraries/development --split development --episodes 128
python -m src.live --steps 40000 --eval-every 5000 --name metric-reproduction --port 8784 --metric-ring --source-checkpoint models/route-source-10k.zip --source-config models/route-source-config.json --return-library return-libraries/train --return-validation return-libraries/development
```

This reproduces the bounded experimental protocol, not a guarantee of success. The simulation and evaluations retain the original stop conditions. Do not extend the failed15k checkpoint. No physical drone connection is required or initiated by these commands.
