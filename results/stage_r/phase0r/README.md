# Phase-0R published result bundle

This directory contains the complete published Stage-R Phase-0R bundle. All
40 trajectory NPZ files and all 40 paired metadata JSON files are present under
`raw/`; they are no longer CPFS-only. `RAW_SHA256SUMS` verifies exactly those
80 files. Their frozen authority sources and the complete canonical manifest
are recorded in `AUTHORITY_MANIFEST.json`, `COMPLETED_PHASE0R_RAW.json`, and
`REMOTE_SHA256SUMS`.

Canonical CPFS root:

```text
/mnt/cpfs/zbl-cpfs-new/USERS/leon/logs/r142_fp11_stage_r/phase0r_merged/r142-stage-r-phase0r-authoritative-20260827
```

Final decision:

```text
NO_STAGE_R_PRECONDITION_ON_PINNED_PI05_LIBERO
CHECKPOINT_1_STOP
phase1_authorized=false
```

The files in this Git bundle are byte-for-byte copies of the canonical
completion artifacts. `REMOTE_MERGE_SHA256SUMS` is the outcome-blind merge
manifest; `REMOTE_SHA256SUMS` is the CPFS 87-file final manifest after frozen
analysis and finalization, so its `analysis/` paths are not a repository-local
layout. `BUNDLE_SHA256SUMS` verifies the original compact result bundle, while
`RAW_SHA256SUMS` verifies all 80 published raw task files locally.
