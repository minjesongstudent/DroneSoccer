# Historical baseline flight failures

Nine submitted recordings were reviewed locally. Six contained takeoff/policy activity; three had no policy flight. The model in these recordings was Tom's unchanged baseline, not the PC composite.

All six policy sessions began with no detected ring. Only 48 of 410 recorded policy steps detected one; four initial landing requests coincided with 160 cm height against the 150 cm test ceiling. Another flight triggered the low-height stop. Offline replay reproduced the baseline outputs, but that only proved consistency with the bundled model, not its attribution to PC training.

The preserved [historical report](REPORT-historical.md) and [model replay](model-replay.json) predate the provenance correction. Any statement that this was "not the wrong model" means the logs matched the bundled baseline; it does not mean the requested PC-trained policy was delivered. [Corrected identity](../provenance/IDENTITY.md).

The first replacement runner stopped if the ring was undetected at the first policy step and retains the first stop reason after an emergency action. It does not change the camera detector, RC gains, or learned weights to claim a flight improvement. Team-default camera calibration, actuator response and reliable recovery remain unvalidated. Raw lab videos and images are not published here.

The current [startup-search release](../startup-search-v1/README.md) adds explicitly requested bounded upward/turning acquisition before policy takeover. This is scripted startup, not a new trained checkpoint.
