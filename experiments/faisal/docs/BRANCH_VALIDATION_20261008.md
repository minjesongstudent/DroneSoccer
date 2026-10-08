# Shared branch validation, 8 October 2026

Branch: `52%-crash-policy`, based on Faisal at `fb3e3b5`.
Selected checkpoint: `trainedModels/seenflag-s1-level1.zip`.
SHA256: `8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef`.
No replacement policy was selected. Master was not modified or merged.

- Repository tests:929passed,1deselected,2warnings in53.41seconds.
- Deselected existing test: `test_hook_blocks_a_symlink_out_of_the_sandbox`.
  The initial complete test invocation failed at symlink creation with Windows
  error1314 (the account lacks symlink privileges), before the hook assertion.
  That assertion remains unverified on this host; its test was not changed.
- Packaged Mac preview:17tests and35subtests passed in5.08seconds, offline on
  Windows. This does not establish execution of the shell launchers on macOS.
- Viewer-specific checks:2passed, including a complete simulated episode through
  the selected85-input wrapper and rejection of an unknown observation profile.
- Package ZIP CRC/content hashes verified;34files,3,234,889bytes. Exact original
  policy included. Team-default camera calibration is explicitly labeled.
- Two direct source-tree preview tests in the initial broad invocation lacked
  generated package manifests. They were subsequently tested in the generated
  package, their intended runtime layout, with the passing result above.

Commands from the repository root (using the local project Python):

```sh
python -m pytest tests -q --disable-warnings -k 'not test_hook_blocks_a_symlink_out_of_the_sandbox'
python tools/build_mac_lab_preview.py --profile seenflag-s1-level1 --out D:/DroneSoccerData/Mac-52-Percent-Policy-Preview-20261008
```

From that generated package directory:

```sh
python -m pytest test_recorder.py test_live_preview.py test_calibration.py -q --disable-warnings
```

Large historical training archives remain local; selected compact evidence is
included under `experiments/faisal/evidence/share-20261008/`. Some research tests
require those archives on the original workstation. No hardware was operated.
Physical flight remains unvalidated and simulation acceptance has not passed.
