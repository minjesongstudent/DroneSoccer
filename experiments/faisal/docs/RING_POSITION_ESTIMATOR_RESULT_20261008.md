# Ring-position estimator and initialization result — 8 October 2026

The distinct ring-relative position estimator passed its predeclared offline comparison. This improves information available to a future controller; it does not establish better flight performance. The original 85-input policy remains selected and physical flight is unvalidated.

Both estimators completed 3,000 CUDA updates in 73.5 seconds total, using only camera and actual issued-command history. Simulator truth supplied labels only: the current ring-centre offset in the drone's body frame, measured in millimetres. This is a different quantity and model identity from the previous displacement estimator.

| Recorded cohort | Feedforward all RMSE | Recurrent all RMSE | Feedforward invisible RMSE | Recurrent invisible RMSE |
|---|---:|---:|---:|---:|
| Historical heldout | 79.0004 mm | 43.6827 mm | 121.1057 mm | 60.7128 mm |
| Separate GAE recordings | 75.7594 mm | 44.6635 mm | 114.6269 mm | 61.1818 mm |

Overall error fell 44.7% and 41.0%; invisible-ring error fell 49.9% and 46.6%. Visible error changed by +0.14% and -0.68%, within the unchanged maximum 5% degradation. Both cohorts met the predeclared 15% overall and invisible improvement gates. The separate cohort was not used for fitting or normalization. These historical diagnostic cohorts are not independent acceptance tests.

The saved-data closure verified 894 unique inputs, 531,088 label rows across both models, source hashes, schedules, masks and axis/timing calculations. Run and closure: `D:/DroneSoccerData/runs/ring-position-estimator-20261008`. Final GRU SHA256: `b387f73399052d73e4aa99cc9b6fe2bb8f13f85f8314654c7cc1e3402c8a31ac`.

## Streaming integration

The new `src/recurrent_ring_position.py` interface has an explicit ring-position identity and unit contract. It advances once per public observation, resets at real episode boundaries and masks the first two estimates. The original motion interface remains unchanged.

The first 600-frame streaming check failed its unchanged 0.001 mm numerical bound: fused CUDA GRU streaming differed by 0.002014 mm. That failure and source are preserved under `ring-position-streaming-20261008`. Evaluating the same trained GRU cell equations explicitly, with the same weights and strict float32 settings, reduced the maximum discrepancy to 0.0002442 mm. CPU/CUDA streaming, packed predictions and exact reset replay passed under `ring-position-streaming-20261008-v2`. No tolerance or weights were changed.

## Policy initialization

The separately named `estimated-ring-position-residual-v1` policy keeps the original actor and estimator frozen and starts a 1,032-parameter correction at zero. It rejects the old displacement estimate despite equal vector dimensions.

The component test run passed 136 tests and the initialization runner passed six further tests. The real 75-case initialization check completed in 33.171 seconds. Its saved-artifact review checked 206 evidence hashes and 34 current sources, then independently compared all 31,634 frames: original actions, public observations, positions, headings, geometry and episode outcomes were exact. Evidence: `D:/DroneSoccerData/runs/ring-position-initialization-20261008`; decision SHA256 `c7329979ea84f3fea4f571896160baa31c9dff3a70f1990b56ba1b1416434d3e`.

This qualifies the integration for a separately declared, bounded actor-training test. It is not a policy promotion or flight-release decision.
