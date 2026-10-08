# DroneSoccer

Drone soccer control, simulation and reinforcement-learning experiments.

**This branch shares the selected 85-input policy with a 52/100 randomized simulation crash result.**
Start with the [selected policy and lab handoff](experiments/faisal/docs/SELECTED_POLICY_LAB_HANDOFF_20261008.md).
The exact model, checksum, evaluation results and a ready-built Mac camera-preview ZIP are included.
Physical flight remains unvalidated; the preview sends no movement commands.

## Current simulator

Use the updated group simulator at the repository root. Run `python -m src.viewer`
and open http://127.0.0.1:8793/ for a learned-policy replay. See
[simulator comparison and limitations](SIMULATOR_COMPARISON.md) for the selection,
input compatibility, tests and differences in passage scoring. The latest bounded
training trial is complete; the original 85-input policy remains selected.

For Mac lab camera recording, overlays and saved-video replay, see the
[camera pipeline report](experiments/faisal/lab-tools/Mac-Lab-Recorder/CAMERA_PIPELINE_20261006.md).
The preview shows learned-policy suggestions without sending movement commands.
Build a separate portable package with `python tools/build_mac_lab_preview.py --out ../Mac-Policy-Preview-New`.
The default is the selected 85-input policy; use a new output directory each time.

The [6 October policy and command-path validation](experiments/faisal/docs/POLICY_FLIGHT_READINESS_20261006.md)
compares two checkpoints over 600 simulated episodes. The stronger 85-input policy
has a separate preview build; neither checkpoint is cleared for physical flight.
Stop/land and watchdog behavior are tested with a local UDP stand-in only.

The [7 October readiness handoff](experiments/faisal/docs/READINESS_HANDOFF_20261007.md)
records the completed training result, detector evidence, explicit camera
calibration, and learned-policy command-session tests. Physical flight remains
unvalidated; the new package provides camera preview and offline command checks.

- [Latest optimized experiments](experiments/faisal/optimized/README.md): current code, seven trial histories, failed checkpoints and next steps.
- [Faisal's local GPU experiment](experiments/faisal/README.md): standalone simulator, one live training policy, saved models and reproducible evaluation.
- [Development process](experiments/faisal/docs/DEVELOPMENT_PROCESS.md): the steps from the initial baseline through the A/B/C experiments and unified training.
- [Results and limitations](experiments/faisal/docs/RESULTS.md): measured outcomes and the current interim snapshot.
- `DroneSoccerWPF/` and `python-front-end/`: the existing desktop/control project.
- `src/`, `hpc/` and `plan.md`: the existing group simulation and training tools.

The preserved experiment has its own `src` package. Follow its README and run its commands
from **`experiments/faisal`**. The repository-root modules belong to the separate
group implementation.

The project retains the existing [MIT license](LICENSE).
