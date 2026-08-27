# Phase-0R published result bundle

This directory mirrors the immutable, small completion artifacts for the
Stage-R Phase-0R run. The 40 trajectory NPZ files remain on CPFS because they
are large; their exact per-file hashes and frozen authority sources are listed
in `AUTHORITY_MANIFEST.json`, `COMPLETED_PHASE0R_RAW.json`, and
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
manifest; `REMOTE_SHA256SUMS` is the 87-file final manifest after frozen
analysis and finalization.

