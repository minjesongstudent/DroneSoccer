# Guarded drone geometry provenance

The current group collision box is a versioned simulation assumption. Available material does not establish either the group's rectangle or the older conservative square as the exact guarded footprint. No dimensions or simulator collision rules were changed in this review.

| Quantity | Existing evidence | Interpretation |
|---|---|---|
| Width 0.17 m | `src/params.py:68`, introduced in team commit `d2908c58bdaa3da6553addfed7866238a7df26ff` (24 September): `MEASURED 2026-09-24, side to side` | Team-recorded width. Original ruler endpoints/guard configuration were not available in searched assets. |
| Height 0.035 m | Same team commit and current `src/params.py:69` | Tagged measured in source; preserved measurement notes describe an approximate photographic reading. |
| Length 0.105 m | Same commit, current `src/params.py:70`: `from the 20 cm diagonal: sqrt(0.20^2 - 0.17^2)` | Derived from an assumed rectangular diagonal, rather than a direct front-to-back measurement. The `MEASURED` tag does not change that derivation. |
| Legacy 0.20 by 0.20 m | Preserved Faisal commit `cafb376dbf2470d1508caf97329a7de98b43fa02`, `experiments/faisal/src/params.py`; current optimized `src/params.py` | A provisional design allowance from photographed spans, not two verified orthogonal measurements. |

Local `git blame` traces the three group dimensions to the original 24 September commit. The 1 October measured-setup changes updated dynamics, camera and hoop parameters, without replacing the diagonal-derived length. Cached `origin/master` at `71c577ec5b4dd7d424c2addbbc3e068eb5987bf6` contains the same derivation; this is a pinned local-history observation, not a claim of a new remote fetch.

The preserved `../DroneSoccer/MEASUREMENTS.md` explains that the diagonal calculation assumes endpoints at rectangular corners, which the curved-guard photographs did not establish. It also explicitly describes the 20 cm square as a provisional approximation, not a verified bound across propeller positions or flight attitudes. `SIMULATOR_COMPARISON.md` already records that the diagonal-derived length needs a direct measurement.

The supplied `../lab-data/20261006-175222-714047/measurements.md` has blank guard-configuration and drone-dimension fields. No original photo establishing the disputed diagonal endpoints was located among searched tracked/history image assets or available attachment images. This absence does not prove that the user lacks an original elsewhere.

## Required evidence for physical clearance

One direct maximum front-to-back measurement with the actual flight guards fitted, showing the ruler endpoints and identifying the guard configuration, resolves the disputed length. An existing clear original photograph demonstrating those same endpoints can serve instead. This does not require repeating the cage, ring or dynamics measurements.

Until then, retain the current group geometry under its documented assumption and the older square only as a separately labeled conservative sensitivity case. Neither is sufficient by itself to claim verified physical clearance. This blocks a physical-envelope claim; it does not justify pausing all simulation work or explain away the observed simulation crashes.

The [qualification mapping](QUALIFICATION_MAPPING_20261007.md) preserves both environments without silently substituting geometry or equating their scores. No hardware, branch publication or group master modification occurred.
