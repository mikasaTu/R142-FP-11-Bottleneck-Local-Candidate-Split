# CPU smoke report

- Date: 2026-08-23 (Asia/Shanghai)
- Host role: local development machine
- Execution: CPU only
- Episodes: 8 paired seeds
- Policies: all 10 required baselines/proposed/ablations
- Tests: `5 passed`
- Genealogy: complete compressed JSONL, SHA256
  `1e5339c3cb25e661a8c51cf4e2a72059fcf39f2ec3826e1050ab28d89f928b2f`
- Collapse-valid fraction: `1.0`
- Proposed localization median error: `0.0`
- Proposed localization within one step: `1.0`
- Proposed success@N: `1.0`
- Uniform success@N: `0.625`
- Random success@N: `0.375`

The smoke gate is deliberately `accepted=false`. Its single block cannot meet
the preregistered requirement of at least 7 winning blocks out of 10. These
numbers are functional smoke evidence only and are not the formal scientific
result.

Generated artifacts are under `aggregate/`, `figures/`, and `shard0/`.
