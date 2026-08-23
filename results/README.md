# Archived experiment artifacts

This directory contains the complete repository-sized evidence bundle for
R142-FP-11 Stage-2A:

- `stage2a_formal`: successful main PAI result, 50-rollout baseline, all 24
  selected and unselected snapshot-frame records, 49,344 raw descendant
  genealogy records, 24 branchability curves, 24 fixed-NFE comparisons,
  negative controls, figures, independent diagnostics and runtime evidence;
- `stage2a_continuation`: all 36 representative eventual-continuation cases and
  the persisted successful completion record;
- `dev14_smoke`: local exact-environment continuation smoke evidence;
- `pai_evidence`: preserved and verified cleanup evidence for the failed
  pre-work predecessor only.

Large JSONL files are losslessly gzip-compressed for GitHub. Rendering and
continuation scripts transparently read `.jsonl.gz`. `RESULTS_MANIFEST.json`
contains both compressed-file hashes and decompressed-content hashes.

The original PAI `SHA256SUMS` files are preserved as execution-time evidence.
Their raw JSONL paths refer to the uncompressed CPFS artifacts; the
`decompressed_sha256` fields in the repository manifest bridge the reversible
archive representation to those original hashes.
