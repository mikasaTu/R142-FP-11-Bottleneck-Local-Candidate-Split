# Stage-R v7 mode-clustering failure

- `observed fact`: The positive control had 210 successful rollouts with
  physical committed modes 102 lower and 108 upper.
- `observed fact`: Average linkage isolated one numerical outlier first; cuts
  at k=2,3,4 had cluster sizes 209/1, 101/108/1, and 101/47/61/1. The frozen
  minimum-size rule rejected every cut.
- `interpretation`: A singleton can dominate hierarchical tree cuts and hide
  two large modes even when no example was filtered.
- `controlled intervention`: v8 may merge any cluster smaller than two into
  the nearest non-small cluster centroid before silhouette evaluation. Every
  example remains assigned and reported. Null thresholds are recomputed.
- `untested hypothesis`: Robust singleton reassignment will expose the two
  large control modes without making the null exceed its recalibrated threshold.

No natural learned-policy outcome was generated or inspected.
