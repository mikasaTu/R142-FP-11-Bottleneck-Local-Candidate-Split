# Stage-R v3 control scope audit

- `observed fact`: v3 exceeded all three null-derived thresholds for pose
  divergence, action split and successful-mode silhouette.
- `observed fact`: v3 succeeded on 512/512 rollouts, so rho was undefined and
  low-p fraction was zero.
- `interpretation`: v3 validates the divergence and mode code paths but does not
  validate Phase-0R's failure-overdispersion precondition.
- `controlled intervention`: v4 may make exactly one geometric aperture
  traversable per initial state, alternating the allowed side. A wrong persistent
  lane choice is then stopped by the real barrier. The control pass condition
  must additionally require rho>=3, low-p fraction>=0.25 and pooled success in
  [0.25,0.75].
- `untested hypothesis`: The alternating aperture will exercise all Phase-0R
  retention metrics while preserving successful modes on both sides.

Natural outcomes remain blinded. V3 thresholds are retained as an incomplete
control audit and are not used for candidate analysis.
