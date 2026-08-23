# Stage-R PAI r2 submission refusal

- `observed fact`: Validation passed the 8-GPU Efficiency contract, but submit
  refused before job creation because `create_artifact_dir=true` is restricted
  to resume workloads while this gate uses `output_mode=new`.
- `observed fact`: No JobId and no gate outcome were created.
- `controlled intervention`: Set `create_artifact_dir=false`, matching the
  controller-verified Stage-2A new-output contract; the controller still
  bootstraps the exact declared write path before UID/GID drop.
- `interpretation`: This is a controller schema correction only.
- `untested hypothesis`: The corrected r3 submission will create the job.
