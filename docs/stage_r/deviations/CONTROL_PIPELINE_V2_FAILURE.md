# Stage-R v2 positive-control failure

- `observed fact`: v2 passed the frozen divergence threshold and action-prefix
  threshold (maximum silhouette 1.0), but produced 0/512 eventual successes.
- `observed fact`: With no successful trajectories, the successful-mode gate
  was undefined and the frozen decision remained `PIPELINE_INVALID`.
- `interpretation`: Holding a constant signed lateral action created a discrete
  genealogy but drove the point mass past the two goal lanes; the control lacked
  closed-loop lateral stabilization.
- `controlled intervention`: v3 may replace the post-choice constant lateral
  command by proportional feedback toward the already frozen lane center
  y=+/-0.48. It may not change the null, geometry, thresholds or natural task.
- `untested hypothesis`: Closed-loop lane feedback will retain the registered
  action split while allowing real forward-simulated success in both modes.

No natural learned-policy outcome was generated or inspected.
