# Ring reacquisition is not sufficient for the return

Using only existing nine FIT teacher histories and saved student FIT trajectories:

- Teacher: all9 pass and reacquire after66–88 steps (6.6–8.8s) from first full pass.
- Student300:8/9 first passes; all8 reacquire, after81–95 steps.
- Student500:7/9 first passes; all7 reacquire, after79–91 steps.
- Neither student completes a full mission on these FIT layouts.

Thus a statement that this student always fails because it cannot see the ring
again would be incorrect. These selected FIT cases show failures after renewed
visibility. This is not evidence that all general-layout failures have this cause.

At student500 reacquisition, seven cases are .496–.654m behind the ring plane,
and .257–.364m below the ring centre. They collide or leave the cage43–178 steps
later. The trace indicates a poor return path after reacquisition, not an
immediate timeout while blind. Position values are diagnostic simulator truth;
none was supplied to the actor.

The comparisons have unequal end points (teacher first-to-second pass, student
first-pass-to-failure); blind-duration summaries are descriptive, not matched
performance metrics. The scalar visible flag does not prove a useful facing
direction, complete ring outline or reliable front-marker evidence.

Artifacts: audit_blind_returns.py and blind_returns_audit.json; original student
FIT traces remain unchanged. No optimization, new trajectory collection, reserved
examples or independent acceptance seeds were used.

Next diagnostic should align teacher/student geometry and public front-marker
and memory states at reacquisition, then assess whether the policy is confusing
front and back or cannot execute the outside return. Do not patch the policy
with truth-fed ring direction, scripted waypoints or a permanently-visible ring.
No new training started on this check; latest bounded trial remains stopped.
