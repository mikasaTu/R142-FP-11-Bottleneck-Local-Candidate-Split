# Stage-R v6 metric-pipeline failure

- `observed fact`: v6 passed low-p prevalence, rho, pooled success, divergence
  and action-prefix checks, but the mode routine returned one mode.
- `observed fact`: Source inspection found that `fcluster` may return
  non-contiguous positive labels while the implementation used
  `min(np.bincount(labels)[1:])`; a missing integer label inserted a zero count
  and rejected otherwise valid clusters.
- `interpretation`: This was a cluster-label accounting bug, not evidence that
  successful control trajectories were single-mode.
- `controlled intervention`: v7 changes only cluster-size counting to enumerate
  actual unique labels. Controls and candidate data rules are unchanged; all
  null thresholds must be recomputed before any natural analysis.
- `untested hypothesis`: Correct unique-label counting will recover the known
  two-sided successful control modes.

No natural learned-policy outcome was generated or inspected.
