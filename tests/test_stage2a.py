import numpy as np
import torch

from r142_stage2a.analysis import (
    actual_nfe,
    branchability_vector,
    classify_curve,
    fixed_nfe_suffix_count,
)
from r142_stage2a.snapshot import canonical_sha256
from r142_stage2a.tracing import PassiveTrace, resume_suffix


class _Output:
    def __init__(self, value):
        self.prev_sample = value


class _Scheduler:
    def __init__(self):
        self.timesteps = torch.arange(5, dtype=torch.long)

    def set_timesteps(self, n):
        self.timesteps = torch.arange(n - 1, -1, -1, dtype=torch.long)

    def step(self, model_output, timestep, sample, generator=None):
        noise = torch.randn(sample.shape, generator=generator, device=sample.device) * 0.01
        return _Output(sample - 0.1 * model_output + noise)


class _UNet:
    def forward(self, sample, timestep, global_cond=None):
        return sample * 0.05 + timestep[:, None, None].float() * 0.001

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _Model:
    def __init__(self):
        self.unet = _UNet()
        self.noise_scheduler = _Scheduler()
        self.num_inference_steps = 5

    def conditional_sample(self, batch_size, global_cond=None, generator=None):
        sample = torch.randn((batch_size, 3, 2), generator=generator)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            prediction = self.unet(
                sample, torch.full(sample.shape[:1], timestep, dtype=torch.long), global_cond=global_cond
            )
            sample = self.noise_scheduler.step(
                prediction, timestep, sample, generator=generator
            ).prev_sample
        return sample


def _generator(seed):
    return torch.Generator(device="cpu").manual_seed(seed)


def test_passive_trace_is_noop_and_resume_is_exact():
    model = _Model()
    original = model.conditional_sample(2, generator=_generator(7))
    generator = _generator(7)
    trace = PassiveTrace(model, generator=generator)
    observed = trace.run(2)
    assert torch.equal(original, observed)
    checkpoint = 2
    continuation = _generator(0)
    continuation.set_state(trace.steps[checkpoint].rng_before_step)
    resumed, _ = resume_suffix(
        model, trace.steps[checkpoint].z_before, checkpoint, None, continuation, capture=True
    )
    assert torch.equal(observed, resumed)


def test_changed_suffix_rng_changes_output_without_changing_checkpoint():
    model = _Model()
    trace = PassiveTrace(model, generator=_generator(11))
    trace.run(1)
    checkpoint = 1
    z_s = trace.steps[checkpoint].z_before
    a, _ = resume_suffix(model, z_s, checkpoint, None, _generator(101))
    b, _ = resume_suffix(model, z_s, checkpoint, None, _generator(102))
    assert torch.equal(z_s, trace.steps[checkpoint].z_before)
    assert not torch.equal(a, b)


def test_compute_accounting_and_branchability():
    assert actual_nfe(8, 100, 20, 8) == 5920
    count, slack = fixed_nfe_suffix_count(5920, 8, 100, 60)
    assert count == 16
    assert slack == 0
    values = branchability_vector(
        [0.1, 0.9, 0.5], np.asarray([[[0, 0]], [[1, 0]], [[0, 1]]]), [0.2]
    )
    assert values["progress_spread"] > 0
    assert values["best_descendant_gain"] == 0.7


def test_curve_classification_and_stable_hash():
    curve = [{"progress_spread": 0.30}, {"progress_spread": 0.18}, {"progress_spread": 0.05}]
    assert "multiple-cliff" in classify_curve(curve, min_drop=0.10)
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
