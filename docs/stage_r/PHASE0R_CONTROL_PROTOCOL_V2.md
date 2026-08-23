# Stage-R Phase-0R control protocol v2

Control protocol ID: `r142-stage-r-controls-v2`.

This revision follows the persisted v1 `PIPELINE_INVALID` result. It changes
only the constructed positive-control controller. The natural-task protocol,
null `OpenPlane2D-v1`, 1000 candidate-identity permutations, maximum-over-time
multiple-testing treatment, and 95th-percentile threshold rule are unchanged.

The v2 positive control remains `GeometricCommit2D-v1` with the same horizon,
integrator, barrier, apertures, one-way geometry, goal locations, initial-state
bias and action noise. When x first reaches -0.30, it draws exactly one binary
lane choice from the preregistered bias plus Gaussian noise and holds that lane
choice through the fork. Before x=-0.30 its approach action remains shared.
This makes the known control contain the action-prefix split that the registered
hierarchical metric is intended to detect; success still follows only from
forward simulation through the physical geometry.

The v2 control is not evidence for the natural-policy hypothesis. If it remains
below any frozen null threshold, the result is again `PIPELINE_INVALID` and no
natural-task outcome may be analyzed.
