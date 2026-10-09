# Experiment evidence

The current comparison is [parallel-credit-20261008](parallel-credit-20261008/README.md): three registered RTX 4070 Ti CUDA workers, separate RNG/model state and outputs, shared reserved fitting seeds for the matched comparison, fixed budgets and stop rules. The cohort selected **no candidate**.

- [Model attribution and lab-package correction](provenance/IDENTITY.md).
- [Training/evaluation evidence](parallel-credit-20261008/README.md).
- [Physical recordings review](physical-flight-20261009/README.md): failed baseline flight evidence, not validation of the PC composite.
- [Publication verification](VERIFICATION.md).
- [Prior tree inventory](provenance/prior-tree.json): removed historical paths and recovery commit.

For an older experiment, browse commit `405b9390b97e25fd29c428d9010e63ea3fa0d255` on this repository. To recover a particular old file without resetting the branch, use `git show 405b939:path/to/file` (binary files should be extracted with `git archive`, not shell text redirection). The complete pre-cleanup tree was also verified in a local ZIP backup. Historical models and runs on the PC were preserved.
