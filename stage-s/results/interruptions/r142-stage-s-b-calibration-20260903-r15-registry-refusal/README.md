# B controller r15 sealed registry refusal

At 2026-09-03 09:31 Asia/Shanghai, a local blackout-guard self-test exposed
an octal parsing error for the string `0931`. The script reached the local
`pai-job` validation layer, which refused the config because its public
`runtime.pod_env` did not match the resource-registry contract.

No PAI job was created: an exact DLC query for display name
`r142-stage-s-b-calibration-20260903-r15` returned no jobs, and the concurrent
Stage-S `Running` query was empty. The r15 controller id is permanently sealed
and is not scientific evidence. The guard now converts `%H%M` with `10#`, has
an observed blackout exit code 75, and the valid post-window submission uses
controller r16 while retaining the immutable r14 scientific directory.
