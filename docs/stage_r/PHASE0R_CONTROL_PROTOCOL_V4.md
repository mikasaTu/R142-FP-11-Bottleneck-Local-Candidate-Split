# Stage-R Phase-0R control protocol v4

Control protocol ID: `r142-stage-r-controls-v4`.

V4 retains the v3 positive controller and all null/threshold rules. For initial
state e, only the upper aperture is traversable when e is even and only the
lower aperture is traversable when e is odd. The other aperture is occupied by
the same rigid barrier. Lane choice remains outcome-blind and is drawn once from
the frozen initial-state bias plus Gaussian noise. A wrong choice remains
blocked and reaches timeout; no success/outcome is assigned by control flow.

The positive control passes only if it exceeds all three frozen null thresholds,
rho>=3, low-p fraction>=0.25, pooled eventual success in [0.25,0.75], and has at
least two stable successful modes. Otherwise the decision is `PIPELINE_INVALID`.
