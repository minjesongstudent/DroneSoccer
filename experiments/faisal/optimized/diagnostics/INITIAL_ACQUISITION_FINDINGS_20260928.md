# Initial acquisition audit

In the 25 existing fixed natural-start layouts, ten initial ring estimates
were valid. Seven starts lacked ring visibility; eight saw the ring but not
its front marker. No case had a visible marker but failed the estimator's
numeric validity conditions. The missing estimates follow the existing
front-sighting contract; this audit does not establish an estimator bug.

Four of the eight visible/no-marker starts acquired a valid estimate after
one step and still failed to pass. Three of the seven invisible starts acquired
an estimate later (steps 17, 27 and 36) and also failed. Eight of the fifteen
initially invalid cases never acquired one before termination. All initial
positions were geometrically on the front side; several were viewed at highly
oblique angles. Geometry is diagnostic only, not a substitute for marker input.

Thus initial invalidity is a useful coverage label, not a sufficient causal
explanation. Do not mark unseen markers as visible or fill missing actor inputs
with simulator position. A possible any-sighting estimator would change the
observation contract and require separate validation and training.

## Natural-reset collector support

Added explicit `collect_natural_segment`, which performs a genuine environment
reset and initializes zero recurrent memory. An empty history is accepted only
through an explicit natural-start flag; snapshot collection still requires a
nonempty authentic prefix. Sequence evaluation marks only the first natural
observation as an episode start. Continuations append actual observed inputs.

CUDA tests passed for pre-update likelihood/value agreement, gradient flow,
repeated seeded reset isolation, continuation history and rejection of empty
mid-flight history. Existing snapshot integration tests also pass. No model
file changed. Scripts: `test_natural_recurrent_rollout.py` and
`test_snapshot_rollout.py`.

Next integrate genuine natural starts into a declared training mixture with
return histories and single-pass retention. The KL guard's distribution replay
currently expects a nonempty prefix and also needs explicit natural-start
support and tests before using that mixture. Keep fixed validation and reserved
acceptance separate from training resets; do not train on the audited 5M seeds.
