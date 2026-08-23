from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np

from .benchmark import EpisodeSpec, ForkPush2D
from .detector import detect_earliest_bottleneck
from .genealogy import Candidate, CandidateSet


POLICIES = (
    "B0_best_of_n",
    "B1_uniform_split",
    "B2_random_split",
    "proposed_bottleneck_local",
    "A_no_detection",
    "A_wrong_early",
    "A_wrong_late",
    "A_correct_random_operator",
    "A_full_resampling",
    "A_more_samples",
)


def _seed(episode_seed: int, policy_index: int, candidate_index: int, stream: int = 0) -> int:
    sequence = np.random.SeedSequence([episode_seed, policy_index, candidate_index, stream, 14211])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _structured_offset(index: int, strength: float) -> float:
    pattern = (-1.0, -0.5, 0.5, 1.0)
    return strength * pattern[index % len(pattern)]


def _candidate(
    env: ForkPush2D,
    spec: EpisodeSpec,
    policy: str,
    policy_index: int,
    index: int,
    injection_schedule: dict[int, float] | None = None,
    parent: Candidate | None = None,
    split_step: int | None = None,
    independent_selector: bool = False,
) -> Candidate:
    candidate_seed = _seed(spec.seed, policy_index, index, 1)
    copy_until = split_step if parent is not None and split_step is not None else 0
    result = env.rollout(
        spec,
        candidate_seed,
        selector_injections=injection_schedule,
        copied_states=None if parent is None else parent.states,
        copied_actions=None if parent is None else parent.actions,
        copied_latents=None if parent is None else parent.latents,
        copy_until=copy_until,
        independent_selector=independent_selector,
    )
    return Candidate(
        episode_id=spec.episode_id,
        policy=policy,
        candidate_id=f"e{spec.episode_id:05d}-{policy}-c{index:03d}",
        parent_id=None if parent is None else parent.candidate_id,
        generation_step=0 if split_step is None else split_step,
        split_action_step=split_step,
        rng_seed=candidate_seed,
        states=result["states"],
        actions=result["actions"],
        latents=result["latents"],
        terminal_score=result["terminal_score"],
        final_success=result["success"],
        final_mode=result["mode"],
        failure_reason=result["failure_reason"],
        collided_step=result["collided_step"],
    )


def _ordinary_set(
    env: ForkPush2D,
    spec: EpisodeSpec,
    policy: str,
    policy_index: int,
    budget: int,
    split_steps: Iterable[int] = (),
    random_operator: bool = False,
) -> CandidateSet:
    steps = list(split_steps)
    amplitude = env.config.split_strength / math.sqrt(max(len(steps), 1))
    candidates: list[Candidate] = []
    for i in range(budget):
        schedule: dict[int, float] = {}
        for step_index, step in enumerate(steps):
            if random_operator:
                rng = np.random.default_rng(_seed(spec.seed, policy_index, i, step_index + 10))
                schedule[step] = float(rng.normal(0.0, amplitude))
            else:
                schedule[step] = _structured_offset(i + step_index, amplitude)
        candidates.append(
            _candidate(env, spec, policy, policy_index, i, injection_schedule=schedule)
        )
    return CandidateSet(
        policy=policy,
        episode_id=spec.episode_id,
        candidates=candidates,
        predicted_bottleneck_step=None,
        detector_diagnostics={},
        split_steps=steps,
    )


def _local_split_set(
    env: ForkPush2D,
    spec: EpisodeSpec,
    policy: str,
    policy_index: int,
    predicted_step: int | None = None,
    use_detector: bool = True,
    random_operator: bool = False,
    independent_selector: bool = False,
) -> CandidateSet:
    cfg = env.config
    scouts = [
        _candidate(env, spec, policy, policy_index, i)
        for i in range(cfg.scout_count)
    ]
    if use_detector:
        predicted_step, diagnostics = detect_earliest_bottleneck(scouts, cfg.horizon)
    else:
        diagnostics = {
            "detector_inputs": "disabled",
            "terminal_label_access": False,
        }
    candidates = list(scouts)
    if predicted_step is not None and 0 <= predicted_step < cfg.horizon:
        for i in range(cfg.scout_count, cfg.candidate_budget):
            parent = scouts[(i - cfg.scout_count) % len(scouts)]
            if random_operator:
                rng = np.random.default_rng(_seed(spec.seed, policy_index, i, 99))
                offset = float(rng.normal(0.0, cfg.split_strength))
            else:
                offset = _structured_offset(i - cfg.scout_count, cfg.split_strength)
            candidates.append(
                _candidate(
                    env,
                    spec,
                    policy,
                    policy_index,
                    i,
                    injection_schedule={predicted_step: offset},
                    parent=parent,
                    split_step=predicted_step,
                    independent_selector=independent_selector,
                )
            )
    else:
        for i in range(cfg.scout_count, cfg.candidate_budget):
            candidates.append(_candidate(env, spec, policy, policy_index, i))
    return CandidateSet(
        policy=policy,
        episode_id=spec.episode_id,
        candidates=candidates,
        predicted_bottleneck_step=predicted_step,
        detector_diagnostics=diagnostics,
        split_steps=[] if predicted_step is None else [predicted_step],
    )


def generate_policy_set(env: ForkPush2D, spec: EpisodeSpec, policy: str) -> CandidateSet:
    if policy not in POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    pi = POLICIES.index(policy)
    cfg = env.config
    if policy == "B0_best_of_n":
        return _ordinary_set(env, spec, policy, pi, cfg.candidate_budget)
    if policy == "B1_uniform_split":
        return _ordinary_set(env, spec, policy, pi, cfg.candidate_budget, cfg.uniform_split_steps)
    if policy == "B2_random_split":
        rng = np.random.default_rng(_seed(spec.seed, pi, 0, 77))
        steps = sorted(
            int(x)
            for x in rng.choice(cfg.horizon, size=cfg.diversity_credits, replace=False)
        )
        return _ordinary_set(env, spec, policy, pi, cfg.candidate_budget, steps)
    if policy == "proposed_bottleneck_local":
        return _local_split_set(env, spec, policy, pi)
    if policy == "A_no_detection":
        return _local_split_set(env, spec, policy, pi, predicted_step=cfg.horizon // 2, use_detector=False)
    if policy == "A_wrong_early":
        return _local_split_set(
            env, spec, policy, pi, predicted_step=max(0, spec.true_bottleneck_step - 2), use_detector=False
        )
    if policy == "A_wrong_late":
        return _local_split_set(
            env,
            spec,
            policy,
            pi,
            predicted_step=min(cfg.horizon - 1, spec.true_bottleneck_step + 2),
            use_detector=False,
        )
    if policy == "A_correct_random_operator":
        return _local_split_set(
            env,
            spec,
            policy,
            pi,
            predicted_step=spec.true_bottleneck_step,
            use_detector=False,
            random_operator=True,
        )
    if policy == "A_full_resampling":
        return _local_split_set(
            env,
            spec,
            policy,
            pi,
            predicted_step=spec.true_bottleneck_step,
            use_detector=False,
            independent_selector=True,
        )
    if policy == "A_more_samples":
        return _ordinary_set(env, spec, policy, pi, cfg.more_samples_budget)
    raise AssertionError("unreachable")


def policy_metadata() -> dict[str, dict[str, Any]]:
    return {
        "B0_best_of_n": {"baseline": "standard best-of-N", "fixed_budget": True},
        "B1_uniform_split": {"baseline": "uniform split", "fixed_budget": True},
        "B2_random_split": {"baseline": "random split", "fixed_budget": True},
        "proposed_bottleneck_local": {"baseline": None, "fixed_budget": True},
        "A_no_detection": {"ablation": "no bottleneck detection", "fixed_budget": True},
        "A_wrong_early": {"ablation": "wrong split location t*-2", "fixed_budget": True},
        "A_wrong_late": {"ablation": "wrong split location t*+2", "fixed_budget": True},
        "A_correct_random_operator": {"ablation": "correct location + random operator", "fixed_budget": True},
        "A_full_resampling": {"ablation": "full resampling at true bottleneck", "fixed_budget": True},
        "A_more_samples": {"ablation": "2x best-of-N samples", "fixed_budget": False},
    }
