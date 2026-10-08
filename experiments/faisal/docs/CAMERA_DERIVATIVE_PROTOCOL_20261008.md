# Source camera feedback derivative check, 8 October 2026

The closed analytic mean-head trial used exact motion but held all public
observations fixed while differentiating. Randomized repeat retention fell13to9/25.
No cause was isolated. Before proposing a distinct gradient that includes camera
feedback, test whether the unchanged OpenCV camera model supplies numerically
stable local pose derivatives. This does not repeat the closed actor trial or
modify its loss, budget, trust constraints or qualification criteria.

Use the64 unchanged deterministic-policy FIT flights already saved in
`D:/DroneSoccerData/runs/policy-sampling-comparison-20261008`. Select post-step
poses0,20,40,... and the final unique pose from each flight, without choosing by
outcome or visibility. Recreate each scene by its recorded reset and require
bitwise identical public eight-feature camera forward output at every selected
pose. No actor loads, optimizer, env.step, new flight, development/acceptance data
or observation adapter. Nearby poses are numerical queries only, never flown.

Central-difference xyz/heading stencil: .001m/.001rad and half-size.0005m/rad.
Each pose also receives one unit direction from RNG11608000 for a separate
half-size directional finite difference. Six continuous output features only;
visibility and marker remain discrete. Detect changes in visibility, marker,
clipping or the projected arc's membership. Invisible interiors have zero
features and do not require identical membership among insufficient arc points.

Freeze gate before reading results: at least100 stable visible poses, no more
than10% of visible poses crossing a branch, and at least95% of stable visible
poses agreeing. Agreement requires BOTH coarse-versus-half Jacobian Frobenius
error and independent directional L2 error <=.002 OR relative error<=.05.
Every stable invisible derivative must be exactly zero. Report excluded branch
cases; do not hide them or smooth their observations. Do not count invisible
zero derivatives toward the95% visible criterion.

One project lock, CPU source-camera numerical queries,600second total cap and
STOP. No neural network training is performed; a later justified actor trial
would use CUDA. Preserve source snapshot, input hashes, all selected poses,
Jacobians, directions, result and evidence. Do not change simulator source.

PASS supports implementing and testing a local camera-feedback gradient only;
it does not validate full closed-loop derivatives, terminal changes, passage
learning, qualification or physical flight. FAIL closes this numerical route at
these scales; no automatic stencil/tolerance sweep or actor training. Discrete
camera switches and variable termination remain limitations even on PASS.
