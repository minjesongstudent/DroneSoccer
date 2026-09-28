# Close-start observability diagnosis

The lower-noise trial completed 65,536 steps with zero passes among480 completed training episodes (150 hits,330 timeouts), and0/25 close validation passes at every checkpoint.

Reset visibility audit:100 layouts per stage, seeds9500000..9500099. The unchanged camera detector reported visible rings in0/100 close starts,33/100 near starts,100/100 centered starts,100/100 varied starts. These are diagnostic resets, not flights or success claims. No camera field of view or detector threshold was modified.

Matched learned-policy flights,25 layouts seeds9400000..9400024: preserved119D SAC policy achieved0/25 close passes (25 timeouts) but25/25 centered passes. Final recurrent candidate achieved0/25 on both:close25hits;centered7hits18timeouts. Every action came from the relevant policy. Source loader is StandardSourceSAC, not plain SAC. Scripts and trajectories are retained.

Conclusion: shorter distance does not make this visual task easier. Close starts omit useful initial ring visibility and recurrent prehistory. This finding does not establish that visibility alone explains all learning failures. Next controlled trial changes only the initial training stage to centered, retaining lower-noise hyperparameters, actual observations, geometry and success rules. No final acceptance has been achieved.
