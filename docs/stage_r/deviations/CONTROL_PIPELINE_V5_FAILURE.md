# Stage-R v5 positive-control failure

- `observed fact`: v5 passed all checks except low-p prevalence; 2/16 states
  met p_e<=1/32 instead of the required 4/16.
- `interpretation`: The frozen lane bias coefficient 0.055 remained small
  relative to Gaussian scale 0.28, so several bias-opposed states still chose
  the open aperture too often.
- `controlled intervention`: v6 may change only that coefficient from 0.055 to
  0.15, leaving the aperture map and all thresholds fixed.
- `untested hypothesis`: The analytically stronger outer-state bias will create
  at least four near-all-fail families without removing successful families.

No natural learned-policy outcome was generated or inspected.
