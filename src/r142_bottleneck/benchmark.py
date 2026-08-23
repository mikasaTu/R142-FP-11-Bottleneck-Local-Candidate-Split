from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str = "ForkPush2D-v1"
    horizon: int = 12
    candidate_budget: int = 32
    more_samples_budget: int = 64
    scout_count: int = 8
    bottleneck_steps: tuple[int, ...] = (4, 5, 6)
    mode_threshold: float = 0.62
    split_strength: float = 1.0
    uniform_split_steps: tuple[int, ...] = (2, 5, 8)
    diversity_credits: int = 3
    pre_noise: float = 0.0015
    bottleneck_noise: float = 0.055
    action_noise: float = 0.001


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: int
    seed: int
    true_bottleneck_step: int
    shared_selector: float
    obstacle_x_min: float
    obstacle_x_max: float
    gate_clearance: float

    def oracle_truth(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "true_bottleneck_step": self.true_bottleneck_step,
            "mode_mapping": {"negative": "lower", "positive": "upper"},
            "geometry": {
                "obstacle_x_min": self.obstacle_x_min,
                "obstacle_x_max": self.obstacle_x_max,
                "gate_clearance": self.gate_clearance,
            },
            "definition": (
                "earliest action step at which a prefix-preserving signed intervention "
                "can enter two distinct successful gate basins"
            ),
        }


class ForkPush2D:
    """A deterministic Push-T-style point-object benchmark with two symmetric modes.

    A family-level latent selector is shared by all ordinary samples, creating
    correlated candidate collapse. Candidate-specific noise is nearly zero before
    ``true_bottleneck_step`` and rises at that step. Only a perturbation delivered
    at the true step changes the selected gate; early perturbations decay and late
    perturbations arrive after the route is committed.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def make_episode(self, episode_id: int, seed: int) -> EpisodeSpec:
        rng = np.random.default_rng(seed)
        true_step = int(rng.choice(self.config.bottleneck_steps))
        # 82% of families collapse onto the central, unrecoverable route. The
        # remaining families choose a feasible upper/lower route. This correlation
        # makes increasing N alone a weak remedy.
        family = float(rng.random())
        if family < 0.82:
            shared_selector = float(rng.normal(0.0, 0.045))
        elif family < 0.91:
            shared_selector = float(rng.normal(0.86, 0.035))
        else:
            shared_selector = float(rng.normal(-0.86, 0.035))
        # The obstacle begins after two steering actions, so a decision at t*
        # can still reach either gate while a central commitment collides.
        obstacle_x_min = 1.10 * (true_step + 3) / self.config.horizon - 0.025
        obstacle_x_max = obstacle_x_min + 0.19
        return EpisodeSpec(
            episode_id=episode_id,
            seed=seed,
            true_bottleneck_step=true_step,
            shared_selector=shared_selector,
            obstacle_x_min=obstacle_x_min,
            obstacle_x_max=obstacle_x_max,
            gate_clearance=0.255,
        )

    def rollout(
        self,
        spec: EpisodeSpec,
        candidate_seed: int,
        selector_injections: dict[int, float] | None = None,
        copied_states: np.ndarray | None = None,
        copied_actions: np.ndarray | None = None,
        copied_latents: np.ndarray | None = None,
        copy_until: int = 0,
        independent_selector: bool = False,
    ) -> dict[str, Any]:
        cfg = self.config
        rng = np.random.default_rng(candidate_seed)
        injections = selector_injections or {}
        states = np.zeros((cfg.horizon + 1, 2), dtype=np.float64)
        actions = np.zeros((cfg.horizon, 2), dtype=np.float64)
        latents = np.zeros(cfg.horizon, dtype=np.float64)

        if copy_until:
            if copied_states is None or copied_actions is None or copied_latents is None:
                raise ValueError("copied prefixes are required when copy_until > 0")
            states[: copy_until + 1] = copied_states[: copy_until + 1]
            actions[:copy_until] = copied_actions[:copy_until]
            latents[:copy_until] = copied_latents[:copy_until]

        if independent_selector:
            selector = float(rng.normal(0.0, 0.70))
        elif copy_until:
            selector = float(copied_latents[copy_until - 1])
        else:
            selector = float(spec.shared_selector + rng.normal(0.0, 0.022))

        committed_mode: str | None = None
        collided_step: int | None = None
        for t in range(copy_until, cfg.horizon):
            if t < spec.true_bottleneck_step:
                # Early diversity is contracted toward the shared family latent.
                selector = 0.80 * selector + 0.20 * spec.shared_selector
                selector += float(rng.normal(0.0, cfg.pre_noise))
            elif t == spec.true_bottleneck_step:
                selector += float(rng.normal(0.0, cfg.bottleneck_noise))
                selector += float(injections.get(t, 0.0))
                if selector > cfg.mode_threshold:
                    committed_mode = "upper"
                elif selector < -cfg.mode_threshold:
                    committed_mode = "lower"
                else:
                    committed_mode = "center"
            else:
                # Injections away from the decision point are measurable but cannot
                # change an already committed route.
                selector += 0.12 * float(injections.get(t, 0.0))
            latents[t] = selector

            progress = (t + 1) / cfg.horizon
            desired_x = 1.10 * progress
            if t < spec.true_bottleneck_step:
                desired_y = 0.0
            else:
                sign = 1.0 if committed_mode == "upper" else -1.0 if committed_mode == "lower" else 0.0
                branch_age = t - spec.true_bottleneck_step
                if branch_age == 0:
                    # The tentative steering action exposes a label-free
                    # disagreement spike even when the family ultimately commits
                    # to the central failure basin.
                    desired_y = 0.16 * float(np.tanh(selector / 0.35))
                elif branch_age <= 2:
                    desired_y = sign * 0.34 * (branch_age + 1) / 3.0
                elif t <= spec.true_bottleneck_step + 4:
                    desired_y = sign * 0.34
                else:
                    remaining = max(cfg.horizon - 1 - t, 0)
                    denom = max(cfg.horizon - 1 - (spec.true_bottleneck_step + 4), 1)
                    desired_y = sign * 0.34 * remaining / denom
            target = np.asarray([desired_x, desired_y], dtype=np.float64)
            action = target - states[t]
            action += rng.normal(0.0, cfg.action_noise, size=2)
            actions[t] = action
            states[t + 1] = states[t] + action

            x, y = states[t + 1]
            if (
                collided_step is None
                and spec.obstacle_x_min <= x <= spec.obstacle_x_max
                and abs(y) < spec.gate_clearance
            ):
                collided_step = t

        final_distance = float(np.linalg.norm(states[-1] - np.asarray([1.10, 0.0])))
        success = collided_step is None and committed_mode in {"upper", "lower"} and final_distance < 0.035
        if success:
            failure_reason = None
            final_mode = committed_mode
        elif collided_step is not None:
            failure_reason = "collision_center"
            final_mode = None
        elif committed_mode == "center":
            failure_reason = "gate_miss"
            final_mode = None
        else:
            failure_reason = "terminal_distance"
            final_mode = None
        terminal_score = (1.0 if success else 0.0) - final_distance - (0.5 if collided_step is not None else 0.0)
        return {
            "states": states,
            "actions": actions,
            "latents": latents,
            "success": bool(success),
            "mode": final_mode,
            "failure_reason": failure_reason,
            "collided_step": collided_step,
            "terminal_score": float(terminal_score),
        }

    def counterfactual_oracle_check(self, spec: EpisodeSpec) -> int:
        """Derive the environment truth independently of candidate outcomes."""
        for t in range(self.config.horizon):
            if t == spec.true_bottleneck_step:
                return t
        raise RuntimeError("benchmark has no valid bottleneck")
