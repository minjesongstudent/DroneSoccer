# Information to check at the lab

The current experiment is simulation-only. Before connecting the learned policy to the drone, establish which observations can actually be reproduced.

1. **Drone and interface:** exact model, firmware, control SDK/library, and which readings it exposes. Record whether height, horizontal/vertical velocity and yaw are available, including units, coordinate frame and update rate. Do not assume the simulator's true position is available on the drone.
2. **Camera observations:** resolution, frame rate, delay and calibration. Check whether the ring detector can produce the ellipse features in `src/contract.py` and whether a front-side marker is physically present and detectable. The simulated marker feature cannot be relied on if the real detector cannot supply it.
3. **Flight geometry:** ring position and centre height inside the 2.05 m cage, clear opening dimensions, hanging support, and the drone's complete envelope with guards fitted. Check both return routes for actual clearance.

The camera/command memory now used in training estimates movement from commands; it is not a position sensor. Measured telemetry may improve this estimate, but adding it requires a matching observation adapter and another validation run. An inaccessible or unmeasured sensor must not be replaced by privileged simulator state in the deployed actor.

After these checks, calibrate the simulator against recorded observations and control response. Strong simulation results alone do not establish that the drone can repeat laps physically.
