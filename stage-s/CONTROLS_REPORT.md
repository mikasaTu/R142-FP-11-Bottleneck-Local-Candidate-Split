# Stage-S controls report

## Scope

These controls qualify only the S2 family-collapse measurement path. They are
not Stage-S substrate evidence, do not test S3-S5, and do not support the
R142-FP-11 mechanism hypothesis by themselves.

## Frozen detector

- family size: 32 candidates;
- near-all-fail: at most 1 success in 32;
- S2 thresholds: near-all-fail fraction at least 0.10, `rho >= 3`, and observed
  near-all-fail families more than 20 times the pooled-binomial expectation;
- pipeline commit recorded by the artifacts:
  `3021de0947b2978c5f509110c9759898a9d53168`.

## Quantitative readback

| Control | Families | Candidates | Pooled success | Near-all-fail | Fraction | rho | Observed / binomial expected | S2 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Constructed positive | 400 | 12,800 | 0.155000 | 338 | 0.845000 | 32.0802 | 26.9461 | pass |
| Stage-R authoritative NPZ null | 640 | 20,480 | 0.967822 | 0 | 0.000000 | 10.4812 | 0.0000 | fail as expected |

The positive control contained 338 deliberately constructed all-fail families,
and the pipeline detected all 338. The null control contained no near-all-fail
family even though its overdispersion statistic alone was above three; the
conjunctive S2 rule therefore correctly returned `NO_FAMILY_COLLAPSE`.

## Decision and integrity

Overall control verdict: `CONTROLS_PASS`. The authoritative machine-readable
artifacts are under `stage-s/results/controls/`; their completion marker is
`COMPLETED_CONTROLS.json`, and every published JSON is covered by the adjacent
`SHA256SUMS`.
