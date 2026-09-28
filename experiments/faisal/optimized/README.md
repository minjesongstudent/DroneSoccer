# Optimized ring-flight experiments

Current snapshot:28 September2026. One learned recurrent policy is being developed for repeated ring passages in a2.05m cage. Latest front-start testing recorded3/25 returns to the front but0/25 full missions. Physical flight remains unvalidated.

Start with [Latest progress](LATEST_PROGRESS.md), [development history](DEVELOPMENT_PROCESS.md) and [results](RESULTS.md). Recurrent model hashes are in models/recurrent-manifest.json. All bounded trials in this snapshot have stopped. Historical SAC instructions below are retained for reference.

Install requirements-recurrent.txt in addition to the existing dependencies. Evaluate the exported recurrent model with evaluate_recurrent.py as described in Latest progress.

## Setup and tests

Use Python3.12 and a CUDA-capable PyTorch installation. The tested workstation uses RTX4070Ti and PyTorch2.11.0+cu128. Install requirements from requirements-tested.txt, then:

```powershell
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pytest -q
```

Current exported-package verification:499 tests passed,25 skipped, with two existing non-failing Gymnasium infinite-bound warnings. Some integration tests require the original local return banks and run layout; they skip when those artifacts are absent. See PUBLICATION_VERIFICATION.md for the exported package test result.

## Reproduce a fresh metric trial

The included source model retains single-pass skill. The metric15k checkpoint is an explicitly failed diagnostic artifact, not a recommended controller. Archive hashes and unchanged-weight verification are in models/manifest.json. Full replay buffers and large trajectory logs remain in local archives; this package includes checkpoint evaluations, configs and compressed source snapshots for seven runs. Config paths labelled local-artifact are provenance placeholders, not runnable paths. Historical trial notes may refer to diagnostic files retained only in the local archive.

Generate new authentic return banks from complete source-policy flights; do not invent observation histories:

```powershell
python -m src.front_return_library --checkpoint models/route-source-10k.zip --output return-libraries/train --split train --episodes 64
python -m src.front_return_library --checkpoint models/route-source-10k.zip --output return-libraries/development --split development --episodes 128
python -m src.live --steps 40000 --eval-every 5000 --name metric-reproduction --port 8784 --metric-ring --source-checkpoint models/route-source-10k.zip --source-config models/route-source-config.json --return-library return-libraries/train --return-validation return-libraries/development
```

This reproduces the bounded experimental protocol, not a guarantee of success. The simulation and evaluations retain the original stop conditions. Do not extend the failed15k checkpoint. No physical drone connection is required or initiated by these commands.
