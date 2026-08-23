# Stage-R Phase-0R metric protocol v8

Metric/control protocol ID: `r142-stage-r-controls-v8`.

For each average-linkage cut, clusters with fewer than two members are assigned
to the nearest centroid among clusters with at least two members, using the same
standardized feature space. No rollout is removed. Silhouette and mode count are
then computed on the complete reassigned labels. Cuts yielding fewer than two
clusters remain invalid. All controls, null permutations and thresholds rerun
under this identical rule before natural outcomes can be analyzed.
