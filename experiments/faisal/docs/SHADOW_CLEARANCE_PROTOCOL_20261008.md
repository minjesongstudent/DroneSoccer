# Differentiable clearance-loss prerequisite - frozen 8 October 2026

The shadow motion prerequisite passed all192 saved learned flights/64,010 frames.
This next component provides a training-only collision-clearance surrogate whose
values can be checked against the unchanged simulator. It is not another
score-function reward-shaping run or a claim that a fixed-history gradient is a
correct closed-loop policy gradient.

Signed ring margin is the source256-point centreline's minimum distance to the
yawed drone box minus noodle half-thickness. Hanger margin is the maximum gap
over the source four horizontal separating axes and bottom-height separation;
it is a separating-axis margin, not Euclidean distance. Cage margin is the
minimum distance to the same oriented-box wall/floor/ceiling limits. Positive
means separated. Source ring uses strict contact, hanger inclusive contact.

Loss is mean squared positive deficit below a DESIGN0.02m nearest-object margin,
in metres squared, over genuine trajectory rows only. The2cm value is neither a
physical measurement nor an inflation of collision geometry. Source scoring,
passage/return/crash metrics and final outcome gates remain unchanged. Truth is
loss/diagnostic input only; deployment still uses public85 camera/actual commands.

Validate all192 saved FIT trajectories against source _box_distance, _hit,
_hits_hanger and _touch_walls at every supplied pose. There are no env.step calls,
actor inference, fitting or new seeds. Fixed gates: ring margin error<=1e-10m;
zero classification disagreements outside a1e-10m numerical boundary band.
Count and report every boundary case/disagreement separately. This band only
qualifies arithmetic comparisons, never alters simulator contacts or acceptance.
Post-wall-clamp poses may sit on a boundary even when that transition contacted
a wall; source geometry is queried at the same saved pose, not confused with its
historical wall event. The prior motion check already verified actual wall counts.

Test gradients through the unchanged shadow motion using the first eight
deterministic saved episodes, never chosen by outcomes. One fixed direction RNG
11606100, epsilon1e-5, zero outside genuine rows or within.1 of command saturation.
Require finite gradients, nonzero norm for a positive loss and directional error
absolute<=1e-8 OR relative<=1e-3. Smoke uses first six saved flights/two deterministic
gradient histories and cannot qualify. CPU/CUDA tests cover geometric classification,
smooth-region gradients, boundary semantics, finite deep-contact derivatives,
disabled hanger and malformed inputs. No tolerance relaxation after results.

Limits: nearest-sample/axis switches and hard contacts are nonsmooth. Deep ring
penetration and post-clamped wall gradients can be zero; progress, wrong-way
passages, camera feedback and variable episode ends are not solved by this loss.
Passing only permits a separately predeclared bounded learned-policy transfer test,
which must preserve first-pass/return/repeat/crash criteria and prevent learning to
hover. It never permits extended fitting or promotion from surrogate loss alone.

One project-lock CUDA process, float64 loss/physics, allocator.15, one CPU thread,
600seconds total, STOP checks per trajectory and integration tick. Preserve every
source/input hash, geometry comparison, episode loss, derivative and result. No
changes to old runs, group master, hardware, or external publication.
