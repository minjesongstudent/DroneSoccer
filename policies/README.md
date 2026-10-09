# Policies

Every saved checkpoint from the current PC comparison is retained, including small changes, failed candidates and initial references. **Saved does not mean qualified.** Do not overwrite or delete an older checkpoint when a new one improves a metric.

- [Checkpoint index](CHECKPOINTS.md): all 15 artifacts and their guard results.
- [Machine-readable registry](registry.json): hashes, architecture, inputs, shared models, per-checkpoint results and status.
- [PC-Policy-Lab-Search.zip](PC-Policy-Lab-Search.zip): self-contained experimental physical flight and recording package with bounded startup search.
- [Startup search details and tests](../experiments/startup-search-v1/README.md): separate scripted acquisition before learned control.
- [Lab runner source](lab-runner/START-HERE.txt): exact package source and instructions.
- [Offline simulator UI](replay/Open-DroneSoccer.html): download and open locally. The same 576 recorded training flights shown on the PC UI.

The replay contains trajectories collected during training, across different rounds. A five-pass replay is not evidence that every final checkpoint succeeds. Final model evaluation is separate.

## Preserve each future checkpoint

Use `save_checkpoint.py` after the trainer atomically finishes saving each checkpoint. It validates the profile, records immutable weights and JSON evidence under `policies/checkpoints/<run>/<arm>/`, and refuses conflicting replacements. Save checkpoints regardless of metric direction; attach later measured results as separate evidence files. Never select or promote a checkpoint merely because it is the best failed candidate. Update the index/registry and publish only in an authorized branch update.

Example from the repository root:

```sh
python policies/save_checkpoint.py --checkpoint /path/to/policy_06.npz --evidence /path/to/guard_06/summary.json --run next-bounded-run --arm full_horizon --step 6
```

This command archives a completed artifact; it does not train, promote, launch a drone, or silently change the lab selection. The unattended-work handoff requires running it for every future checkpoint.
