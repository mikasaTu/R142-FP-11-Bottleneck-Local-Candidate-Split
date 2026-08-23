# Stage-R Phase-0R control protocol v3

Control protocol ID: `r142-stage-r-controls-v3`.

The persisted v2 control failed because its constant lateral command overshot
both goal lanes. V3 changes exactly one positive-control equation after the
binary lane choice: `a_y = fork_gain * 2.0 * (0.48 * lane_choice - y)`.
Everything else in v2 remains frozen, including the shared prefix, one-time lane
choice, physical fork and one-way geometry. Null data, 1000 permutations,
maximum-over-time statistic and 95th-percentile thresholds are unchanged.

If v3 does not pass all positive-control checks, the pipeline remains invalid
and natural outcomes stay blinded.
