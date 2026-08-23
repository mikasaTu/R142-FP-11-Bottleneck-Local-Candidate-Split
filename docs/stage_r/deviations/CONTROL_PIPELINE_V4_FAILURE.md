# Stage-R v4 positive-control failure

- `observed fact`: v4 passed the divergence, action-split, successful-mode,
  rho, and pooled-success checks.
- `observed fact`: Only 1/16 initial states had p_e<=1/32, below the frozen
  4/16 requirement; the decision is `PIPELINE_INVALID`.
- `interpretation`: Alternating aperture side was poorly aligned with the
  monotone initial-state lane bias and did not create enough near-certain
  failure families.
- `controlled intervention`: v5 may group the four most negative-bias states
  against the upper aperture and the four most positive-bias states against the
  lower aperture; the middle eight states use their bias-aligned aperture.
- `untested hypothesis`: This preregistered mapping will create at least four
  all/near-all-fail families while retaining successful families and both modes.

No natural learned-policy outcome was generated or inspected.
