# PAI registry export notes

The canonical registry and CPFS evidence remain on dev14. The GitHub export
contains requested/resolved contracts, redacted submit/readback records,
placement evidence, payloads, deletion intent/response/evidence, and workload
logs.

Four raw deletion-helper preflight/execute GetJob/ListJobs snapshots were
intentionally excluded from GitHub because the raw PAI API response can contain
secret environment values. They remain preserved only in the canonical
Leon-owned registry run. Their non-secret deletion result is retained in
`pai-task-delete-evidence.json` and `pai-task-delete-response.json`:
`deleted=true`, `verified_absent=true`.
