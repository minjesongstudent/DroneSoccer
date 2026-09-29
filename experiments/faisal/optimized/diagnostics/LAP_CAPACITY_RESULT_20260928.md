# Lap-only diagnostic: rejected at the first retention check

The saved 500-update policy was tested with a bounded supervised diagnostic:
nine complete FIT teacher histories, no basic rehearsal, actor-only Adam 1e-4,
maximum 100 updates, checks at 25/50/100. All gradients ran on RTX 4070 Ti CUDA.
Source snapshot, runner, source checkpoint hash, dataset hashes and model25
are preserved in `lap-capacity-20260928`. No held-out acceptance seeds were used.

At 25 updates: centered single-pass fell from 25/25 to 15/25, varied single-pass
from 21/25 to 5/25. All three full mission suites remained 0/25. The predeclared
retention stop triggered and no further updates ran. This checkpoint is not
promoted; the existing viewer and historical models remain unchanged.

The preceding gradient audit (`transfer_gradient_audit.json`) found basic/lap
gradient cosine -0.351 at checkpoint300 and -0.292 at500. Nevertheless, equal
mix gradients had positive dot product with both loss gradients, indicating
first-order descent for both under a sufficiently small ordinary gradient step.
Adam and clipping can behave differently; this is not a guarantee of retention.
Parameter hashes before/after the gradient audit were identical.

The trial changes both rehearsal and learning rate, so it does not isolate a
single cause or prove that the network lacks capacity. It does establish that
this proposed shortcut damages the required basic skill before delivering full
missions. Do not resume or extend this run. Next investigate how to preserve
the basic policy while fitting the return behaviour (for example, constrained
updates with measured retention), rather than simply dropping rehearsal or
increasing the rate. Any new variant still needs closed-loop measured gains;
imitation loss and selected teacher successes are not acceptance evidence.

No training remains active after the retention stop. Physical flight unvalidated.
