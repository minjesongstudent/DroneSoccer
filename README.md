# DroneSoccer

Drone soccer control, simulation and reinforcement-learning experiments.

- [Faisal's local GPU experiment](experiments/faisal/README.md): standalone simulator, one live training policy, saved models and reproducible evaluation.
- [Development process](experiments/faisal/docs/DEVELOPMENT_PROCESS.md): the steps from the initial baseline through the A/B/C experiments and unified training.
- [Results and limitations](experiments/faisal/docs/RESULTS.md): measured outcomes and the current interim snapshot.
- `DroneSoccerWPF/` and `python-front-end/`: the existing desktop/control project.
- `src/`, `hpc/` and `plan.md`: the existing group simulation and training tools.

The experiment has its own `src` package. Follow its README and run its commands
from **`experiments/faisal`**. The repository-root modules belong to the separate
group implementation.

The project retains the existing [MIT license](LICENSE).
