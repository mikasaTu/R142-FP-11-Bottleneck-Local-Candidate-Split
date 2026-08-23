# Stage-R Phase-0R metric protocol v7

Metric/control protocol ID: `r142-stage-r-controls-v7`.

V7 retains v6 trajectories and changes one implementation detail: for every
candidate cluster labelling, cluster size is computed as
`sum(labels == value)` for each value in `np.unique(labels)`. No assumption of
contiguous labels is permitted. The complete null and 1000 registered
permutations are rerun; no earlier threshold is reused.
