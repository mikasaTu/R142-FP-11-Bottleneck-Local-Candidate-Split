"""Observe and resume the real LeRobot DiffusionModel reverse loop.

The passive tracer calls the pinned model's original ``conditional_sample`` and
only wraps the existing UNet/scheduler call sites. It therefore provides a
strong no-op equivalence gate. ``resume_suffix`` intentionally calls the same
UNet and scheduler objects used by the policy; it is not a surrogate sampler.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch


def tensor_sha256(value: torch.Tensor) -> str:
    x = value.detach().contiguous().cpu()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def generator_state_sha256(state: torch.Tensor) -> str:
    return hashlib.sha256(state.detach().cpu().numpy().tobytes()).hexdigest()


@dataclass
class DenoisingStep:
    index: int
    timestep: int
    z_before: torch.Tensor
    model_output: torch.Tensor | None = None
    z_after: torch.Tensor | None = None
    rng_before_step: torch.Tensor | None = None

    def manifest(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestep": self.timestep,
            "z_before_shape": list(self.z_before.shape),
            "z_before_sha256": tensor_sha256(self.z_before),
            "model_output_sha256": None
            if self.model_output is None
            else tensor_sha256(self.model_output),
            "z_after_sha256": None if self.z_after is None else tensor_sha256(self.z_after),
            "rng_before_step_sha256": None
            if self.rng_before_step is None
            else generator_state_sha256(self.rng_before_step),
        }


@dataclass
class PassiveTrace:
    """No-op observer for the official ``conditional_sample`` implementation."""

    diffusion_model: Any
    generator: torch.Generator | None = None
    capture_tensors: bool = True
    steps: list[DenoisingStep] = field(default_factory=list)
    final_sample: torch.Tensor | None = None
    _original_unet_forward: Any = field(default=None, init=False, repr=False)
    _original_scheduler_step: Any = field(default=None, init=False, repr=False)

    @contextlib.contextmanager
    def installed(self):
        model = self.diffusion_model
        self.steps.clear()
        self.final_sample = None
        self._original_unet_forward = model.unet.forward
        self._original_scheduler_step = model.noise_scheduler.step

        def observed_unet(sample, timestep, *args, **kwargs):
            index = len(self.steps)
            scalar_t = int(timestep.flatten()[0].detach().cpu().item())
            rng_state = None
            if self.generator is not None:
                rng_state = self.generator.get_state().clone()
            self.steps.append(
                DenoisingStep(
                    index=index,
                    timestep=scalar_t,
                    z_before=sample.detach().clone() if self.capture_tensors else sample.detach().cpu(),
                    rng_before_step=rng_state,
                )
            )
            output = self._original_unet_forward(sample, timestep, *args, **kwargs)
            self.steps[-1].model_output = (
                output.detach().clone() if self.capture_tensors else output.detach().cpu()
            )
            return output

        def observed_scheduler_step(model_output, timestep, sample, *args, **kwargs):
            result = self._original_scheduler_step(model_output, timestep, sample, *args, **kwargs)
            self.steps[-1].z_after = (
                result.prev_sample.detach().clone()
                if self.capture_tensors
                else result.prev_sample.detach().cpu()
            )
            return result

        model.unet.forward = observed_unet
        model.noise_scheduler.step = observed_scheduler_step
        try:
            yield self
        finally:
            model.unet.forward = self._original_unet_forward
            model.noise_scheduler.step = self._original_scheduler_step

    def run(self, batch_size: int, global_cond: torch.Tensor | None = None) -> torch.Tensor:
        with self.installed():
            out = self.diffusion_model.conditional_sample(
                batch_size=batch_size, global_cond=global_cond, generator=self.generator
            )
        self.final_sample = out.detach().clone()
        return out


@torch.no_grad()
def resume_suffix(
    diffusion_model: Any,
    z_s: torch.Tensor,
    checkpoint_index: int,
    global_cond: torch.Tensor | None,
    generator: torch.Generator | list[torch.Generator] | None,
    *,
    capture: bool = False,
) -> tuple[torch.Tensor, list[DenoisingStep]]:
    """Resume the pinned scheduler at a real checkpoint, inclusive.

    ``checkpoint_index`` indexes the scheduler's actual timesteps and ``z_s`` is
    the tensor immediately before the UNet evaluation at that index.
    """

    scheduler = diffusion_model.noise_scheduler
    scheduler.set_timesteps(diffusion_model.num_inference_steps)
    timesteps = scheduler.timesteps
    if checkpoint_index < 0 or checkpoint_index >= len(timesteps):
        raise IndexError(f"checkpoint_index={checkpoint_index} outside [0,{len(timesteps)})")
    sample = z_s.clone()
    records: list[DenoisingStep] = []
    for local_index, timestep in enumerate(timesteps[checkpoint_index:], start=checkpoint_index):
        if isinstance(generator, list):
            rng_state = None
        elif generator is None:
            rng_state = None
        else:
            rng_state = generator.get_state().clone()
        model_output = diffusion_model.unet(
            sample,
            torch.full(sample.shape[:1], timestep, dtype=torch.long, device=sample.device),
            global_cond=global_cond,
        )
        z_before = sample.detach().clone() if capture else sample.detach().cpu()
        sample = scheduler.step(
            model_output, timestep, sample, generator=generator
        ).prev_sample
        if capture:
            records.append(
                DenoisingStep(
                    index=local_index,
                    timestep=int(timestep.detach().cpu().item()),
                    z_before=z_before,
                    model_output=model_output.detach().clone(),
                    z_after=sample.detach().clone(),
                    rng_before_step=rng_state,
                )
            )
    return sample, records


def repeat_condition(global_cond: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    if global_cond is None:
        return None
    return global_cond.repeat_interleave(repeats, dim=0)


def repeat_roots(root_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Order as root0/suffix0..M-1, root1/suffix0..M-1, ..."""

    return root_states.repeat_interleave(repeats, dim=0)
