# Mechanism reverse explanation

This report follows the code-first requirement: the explanation is derived from
the implemented generator and controlled ablations, not from a paper narrative.
No new idea is proposed.

## Reconstructed algorithm

The ordinary generator uses a shared family selector plus small candidate-level
noise. Before `t*`, selector perturbations contract back toward the shared
family value. At `t*`, the selector chooses upper, lower, or central collision
basin; after that step the route is committed. The detector observes a sharp
same-time prefix disagreement onset at `t*` and allocates the remaining 24 of
32 candidates there.

## Confirmed cause of the gain: split location

This is confirmed by controlled intervention.

- Proposed at detected correct location: success@N 1.000.
- Correct location + Gaussian random operator: 1.000.
- Correct location + full resampling: 1.000.
- Wrong location `t*-2`: 0.155.
- Wrong location `t*+2`: 0.155.

Uniform split hits the true step in 138/400 episodes. Conditional success is
0.9855 when it hits and 0.1527 when it misses. Random split hits 91/400;
conditional success is 1.000 when it hits and 0.1586 when it misses. Thus the
dominant causal variable is whether diversity is delivered at the true decision
step.

Early diversity is suppressed by the generator's contraction toward the shared
family selector. Late diversity cannot change a committed route. Local
diversity crosses the gate threshold before collision and converts a correlated
central family into upper/lower candidates.

## Confirmed cause of the more-samples failure: family correlation

B0 with 32 candidates and B0 with 64 candidates both have success@N 0.155.
Their mode discovery rates are both 0.0775. The extra samples draw more members
from the same family-level selector; they do not create a new decision branch.
This confirms that the observed gain is not an N advantage.

## Structured operator is not the source of the gain

The proposed antithetic structured operator is not isolated as superior.
Correct-location Gaussian and full-resampling alternatives match its 1.000
success@N and improve mode discovery (0.98875 and 1.000 versus 0.9225). Candidate
success fraction is also higher for Gaussian/full resampling (0.4538/0.4928)
than structured local split (0.4295).

Code-level cause: the structured pattern includes `-0.5` and `+0.5` offsets.
For some shared selectors these do not cross the absolute mode threshold 0.62,
while full resampling and Gaussian draws more often produce larger magnitudes.
Therefore the supported mechanism is bottleneck-local allocation, not the
specific four-offset operator.

## Why uniform and random are lower

All fixed-budget methods use 32 terminal candidates and the same total
perturbation energy. Uniform spreads energy across steps 2, 5, and 8, reducing
amplitude at any one step. Random spreads it across three randomly selected
steps. Their lower aggregate success is fully consistent with a mixture of
near-perfect hit episodes and B0-like miss episodes.

## Why the detector works here

The benchmark intentionally exposes candidate-specific uncertainty at the
decision step while keeping the earlier stem nearly identical. The robust
`ΔD` threshold detects that onset exactly in all 400 evaluation episodes. This
is confirmed for this generator, but it is also the main external-validity
limitation: a learned policy may not expose such a clean disagreement spike.
The result must not be extrapolated to VLA without Phase-1 small-policy tests.

## Failure mechanisms

- Central family collapse: all candidates collide at the obstacle.
- Early split: the selector contraction erases the perturbation before `t*`.
- Late split: collision basin is already committed.
- Uniform/random miss: no effective perturbation reaches `t*`.
- Structured half-offsets: some remain below the gate threshold, lowering mode
  discovery relative to full resampling.

These mechanisms are supported by controlled code paths and paired metrics;
claims beyond this synthetic environment remain hypotheses.
