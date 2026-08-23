# Stage-2A source-interface audit

## Frozen lineage

- repository: `https://github.com/huggingface/lerobot.git`
- commit: `3c0a209f9fac4d2a57617e686a7f2a2309144ba2`
- checkpoint: `lerobot/diffusion_pusht`
- checkpoint revision: `84a7c23178445c6bbf7e1a884ff497017910f653`
- weights SHA-256: `995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`

The checkpoint model card explicitly names the frozen LeRobot commit. The
published `config.json` also contains two later CLI-only keys, `device` and
`use_amp`, which are not dataclass fields at that commit. The loader excludes
only those two non-model keys and `type`, explicitly constructs the pinned
`DiffusionConfig`, and loads the original safetensors with `strict=True`. The
checkpoint files are never modified.

## Real inference path

File:

`lerobot/common/policies/diffusion/modeling_diffusion.py`

Call path:

1. `DiffusionPolicy.select_action`
2. `DiffusionModel.generate_actions`
3. `DiffusionModel.conditional_sample`

The real noisy tensor is named `sample`, with PushT shape `(B,16,2)`. The loop
sets 100 scheduler timesteps and, for every actual timestep `t`, computes:

```python
model_output = self.unet(sample, t_batch, global_cond=global_cond)
sample = self.noise_scheduler.step(
    model_output, t, sample, generator=generator
).prev_sample
```

The checkpoint uses epsilon prediction. Diffusers also exposes a predicted
clean sample in its scheduler return value, but the LeRobot policy reads only
`prev_sample`; this experiment does not invent a clean-action interface.

The normalized horizon is sliced from index 1 through 8 inclusive (Python
slice `1:9`), yielding eight actions, and is then passed through the checkpoint
`Unnormalize` module. Public `select_action` does not expose a generator, so
the original path uses the PyTorch global RNG. Counterfactual resume uses the
same model/scheduler objects with explicit generator states to make suffix RNG
continuations auditable.

## Instrumentation boundary

`PassiveTrace` wraps only the existing UNet forward and scheduler step methods,
then calls the original `conditional_sample`. It captures `z_s`, timestep,
predicted epsilon, `z_{s-1}` and the explicit generator state without replacing
any tensor. `resume_suffix` starts at an actual scheduler index and repeatedly
calls the same frozen UNet and DDPM scheduler.

## Simulator boundary

Standard `gym-pusht==0.1.5` uses Pymunk, not MuJoCo. Its public
`reset_to_state` restores agent/block position and block angle but omits body
velocities, angular velocity, RNG, wrapper time and Pymunk transient state. It
is therefore not used as a complete snapshot. A snapshot is restored by
creating the standard environment, resetting with the exact episode seed and
replaying the full recorded action prefix. A–E test E compares native body
states, observation hashes and the same descendant chunk after independent
replays.

## Verified smoke result

The exact pinned runtime (Python 3.10.20, torch 2.6.0+cu124, torchvision
0.21.0+cu124, diffusers 0.32.2, gymnasium 0.29.1, gym-pusht 0.1.5 and Pymunk
6.11.0) passed A–E on one dev14 A800. This is engineering feasibility evidence,
not a Stage-2A scientific result.
