Latest update: [recurrent experiments and measured results,28 September](optimized/LATEST_PROGRESS.md). Reliable repeated laps remain unsolved.

# Local GPU ring-flight experiment

One SAC policy trains in a 2.05 m enclosure with a hanging ring. A browser viewer
shows the live trajectory, learning progress and separate policy evaluations.
This folder is self-contained; run all commands from **`experiments/faisal`**.
The repository-root `src` is a different implementation.

Start with the [development process](docs/DEVELOPMENT_PROCESS.md) and
[results](docs/RESULTS.md). C520k retained useful single-pass skill, but reliable
laps, delayed control and physical flight have not been demonstrated. The
unified 110k snapshot is an interim record, not a completed or best-performing run.

## Setup on Windows with an NVIDIA GPU

Tested with Python 3.12.13, PyTorch 2.11.0+cu128 and an RTX 4070 Ti. Install Python
3.12 and an NVIDIA driver compatible with the CUDA wheel. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA required'; print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m pytest -q
```

The CUDA build is available from the [official PyTorch wheel index](https://download.pytorch.org/whl/cu128/torch/).
The live trainer refuses to fall back to CPU. Simulator stepping and the web
server use the CPU; neural-network inference and updates use CUDA. The optional
physical-flight dependency `djitellopy` is listed in `requirements.txt`; it is
not required for the simulator.

## Start one live training run

```powershell
.\start-live-gpu.ps1
```

Open <http://127.0.0.1:8773/>. This starts from the bundled C520k policy with an
empty replay buffer. The first 1,000 new transitions fill replay using the
student's policy; then ordinary SAC updates begin. New counters start at zero.
The source policy's earlier demonstration-based warm-start remains part of its
history; this run creates no new demonstrations or imitation updates.

The initial practice mixture is 60% current start layouts and 40% earlier starts.
All use the current task's rewards, dynamics and episode horizon. Learning stops
at 1,000,000 new steps, saves and switches to replay. Closing the page does not
stop the Python process. Stop & save pauses learning; Resume and Add 100,000 steps
control further training. Mouse dragging rotates the scene and scrolling zooms.

Run the launcher again to resume an existing `unified-local` run. For a named run:

```powershell
.\start-live-gpu.ps1 -Name experiment-01 -Steps 1000000 -Port 8773
# Equivalent first-run Python command:
.\.venv\Scripts\python.exe -m src.live --name experiment-01 --steps 1000000 --port 8773
# Resume that run using its local model, replay, optimizer and curriculum files:
.\.venv\Scripts\python.exe -m src.live --name experiment-01 --steps 1000000 --port 8773 --resume
```

Run only one live trainer on the chosen port. A port already in use produces an
error before a second training thread starts. Exact interrupted trajectories
are not restored: disk resume starts a fresh episode.

## Reproduce checkpoint selection without training

```powershell
.\.venv\Scripts\python.exe tools/evaluate_checkpoints.py
```

This uses CUDA, deterministic actions and the three bundled C checkpoints. Each
gets 100 centered, 100 varied-approach and 25 full-task episodes on seeds beginning
at 4,000,000. Results go to `runs/checkpoint-selection/`; model files are unchanged.
See `--help` for episode counts, seed range, model selection and output paths.
To inspect the interim unified policy independently:

```powershell
.\.venv\Scripts\python.exe tools/evaluate_checkpoints.py --models models/unified-110000.zip --output runs/unified-snapshot-evaluation
```

The live run has five validation suites every 10,000 steps, on seeds starting at
5,000,000: focus task, centered pass, varied pass, full task and full task with
200 ms delay. Full-task success requires 60 seconds of survival and at least two
passes. Single-pass success requires full-body clearance. Repeated validation
seeds are not untouched final test data.

## Included evidence

- `models/c520k.zip`, `models/c670k.zip`, `models/c1000k.zip`: C candidate checkpoints.
- `models/unified-110000.zip`: immutable interim checkpoint for inference or a new continuation.
- `models/manifest.json`: original and published hashes, counts and seeds.
- `models/c-source-config.json`: geometry and source experiment configuration.
- `docs/results/`: historical evaluations, A/B/C final results, selection records and unified snapshots.

Only the model archive's training-log directory metadata was removed for
portability; all weight entries are unchanged. Full replay buffers, live output,
local environments and personal machine paths are excluded. Consequently the
published checkpoints support inference and new warm-start runs, not exact
continuation of the original archived processes. A new run's own resume files
are saved under its ignored `runs/` directory.

## State, action and reward

The policy input has 84 values: nine frames of eight ring features plus three
previous four-channel actions. Actions command right/left, forward/backward,
up/down and clockwise/anticlockwise yaw, normalized to [-1, 1].

During laps the reward is signed progress toward the current target (+1 per
10 cm closer), +25 for a scored forward crossing, -40 for a crash/wrong-way
crossing/geofence exit, and -0.02 per step without a visible ring. A positive
episode reward can coexist with a failed flight. Lap passes score at center
crossing; separate clean-pass tests require the full body to clear.

Geometry and camera assumptions are documented in the development process and
tagged in `src/params.py`. The inherited [MIT license](../../LICENSE) applies.

## Latest optimization work

The [optimized experiment](optimized/README.md) preserves the subsequent reward corrections, return-practice trials, memory/critic diagnostics and the stopped metric-ring15k trial. See its [development process](optimized/DEVELOPMENT_PROCESS.md) and [results](optimized/RESULTS.md) for the current status. Repeated laps remain unsolved.
