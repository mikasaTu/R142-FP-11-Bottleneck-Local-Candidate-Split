# Stage-R v1 positive-control failure

- `observed fact`: The v1 null-control 95th-percentile maximum action-prefix
  silhouette was 0.7127923128725111.
- `observed fact`: The v1 geometric positive control reached two successful
  modes, rho=21.2550, and passed pose-divergence and mode tests, but its maximum
  action-prefix silhouette was 0.7000218943651908.
- `observed fact`: The frozen v1 decision is `PIPELINE_INVALID`; no natural
  learned-policy outcome was generated or inspected.
- `interpretation`: Maximizing silhouette over 80 prefixes gives the noisy
  open-plane null a high chance peak, while the positive controller's continuous
  lateral action distribution does not form a discrete action-prefix split even
  though collision geometry produces two terminal modes.
- `controlled intervention`: A v2 control may replace only the positive
  controller with a preregistered persistent binary lane choice at the physical
  fork. The null simulator, 95th-percentile rule, 1000 shuffles and all natural
  task rules remain unchanged.
- `untested hypothesis`: A persistent lane choice will make the known positive
  action genealogy exceed the unchanged null-derived threshold.

This record does not license changing a natural-task threshold, interpreting a
candidate outcome, or proceeding past the positive-control gate.
