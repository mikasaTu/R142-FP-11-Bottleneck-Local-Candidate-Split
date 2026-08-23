# Stage-R PAI 2-GPU validation refusal

- `observed fact`: Canonical controller validation refused
  `exp-efficiency-a800-2gpu` before job creation because that alias is restricted
  to pinned DINO-WM, R21-P019 and R19-P10-v3 profiles.
- `observed fact`: No PAI JobId was created and no gate outcome ran.
- `controlled intervention`: The resource contract is changed to the verified
  generic `exp-efficiency-fastwam-8gpu` carrier (8xA800, 92 CPU, 1600GiB), still
  in the user-requested Efficiency pool.
- `interpretation`: This is controller admission policy, not an engineering or
  scientific gate result.
- `untested hypothesis`: The generic 8-GPU carrier will validate and admit the
  unchanged gate payload.
