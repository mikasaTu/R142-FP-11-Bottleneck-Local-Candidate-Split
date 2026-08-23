# Stage-R Phase-0R control protocol v5

Control protocol ID: `r142-stage-r-controls-v5`.

V5 changes only the v4 allowed-aperture map. States 0--3 open the upper
aperture; states 4--7 open the lower; states 8--11 open the upper; states 12--15
open the lower. Because the frozen lane-choice bias increases monotonically
with state index, the outer eight states are bias-opposed and the middle eight
are bias-aligned. All dynamics, noise, geometry, thresholds, pass criteria and
natural-task rules remain unchanged.
