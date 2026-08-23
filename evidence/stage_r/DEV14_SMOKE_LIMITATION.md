# Stage-R dev14 GPU-smoke limitation

- `observed fact`: dev14 exposed four A800 GPUs through `nvidia-smi`, but
  robosuite's EGL enumeration returned zero devices during environment creation.
- `observed fact`: Three attempts covered local and physical GPU numbering and
  explicit `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `EGL_DEVICE_ID`, and
  `MUJOCO_EGL_DEVICE_ID`; each failed before reset, policy load, or rollout.
- `observed fact`: Only the read-only E1 checkpoint manifest was persisted;
  E2--E6 did not run and no smoke-pass claim is made.
- `interpretation`: The dev14 SSH host namespace lacks the NVIDIA EGL device
  exposure available inside the PAI worker container; CUDA visibility alone is
  insufficient for this simulator.
- `controlled intervention`: Formal E1--E6 is moved unchanged to the registered
  2xA800 Efficiency PAI container.
- `untested hypothesis`: The PAI container's standard NVIDIA/EGL mounts will
  allow the frozen simulator gate to execute.

This limitation does not license treating E1 as gate completion or starting
Phase-0R science.
