# Public ring estimate audit

The 32 saved nominal training continuations were replayed exactly. Simulator
geometry was only a reference for measuring the public estimate; no actor input
or model was changed. All 5,782 frames had a valid estimate. Median 3D error was
2.22 cm, 95th percentile 3.26 cm, maximum 5.43 cm. During 3,995 invisible-ring
frames, median error was 2.00 cm and 95th percentile 3.24 cm. This does not rule
out clearance-sensitive errors, but gross loss of ring position is not evident
in these return failures. The actor also lacks unseen cage-wall positions.

## Natural-start fixed validation

The preserved policy was additionally evaluated on the existing fixed 25-seed
general and delayed suites. No optimization or independent acceptance test ran.

| Measure | Nominal | 200 ms delay |
|---|---:|---:|
| Initially valid estimate | 10/25 | 10/25 |
| First pass, initially valid | 6/10 | 6/10 |
| First pass, initially invalid | 0/15 | 0/15 |
| Median valid-frame error | 2.86 cm | 5.59 cm |
| 95th percentile error | 6.29 cm | 12.23 cm |
| Maximum error | 15.76 cm | 23.57 cm |
| Invalid-input frames | 291/1717 | 327/1822 |

These are descriptive associations, not proof that estimate invalidity causes
failure. Different starting layouts can simultaneously affect visibility and
difficulty. Physical camera/telemetry calibration remains unvalidated.

Detailed per-step errors, visibility and actual learned actions are saved under
the original stepwise trial's `ring_estimate_audit` and `natural_estimate_audit`.
Scripts: `audit_ring_estimate_paths.py`, `audit_natural_ring_estimate.py`.

## Next work

Audit the first acquisition failures before further return-only optimization:
distinguish absent visual detection from missing front-marker memory, then
select a genuine natural-reset training distribution that includes these
starts alongside single-pass retention and return histories. Do not fill
invalid estimates from simulator truth or change observations silently. Address
delay calibration separately rather than claiming nominal estimate accuracy
transfers to physical flight.
