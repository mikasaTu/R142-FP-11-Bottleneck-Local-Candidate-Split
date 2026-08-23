from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _rounded(values: np.ndarray, digits: int = 6) -> list[Any]:
    return np.round(values, digits).tolist()


@dataclass
class Candidate:
    episode_id: int
    policy: str
    candidate_id: str
    parent_id: str | None
    generation_step: int
    split_action_step: int | None
    rng_seed: int
    states: np.ndarray
    actions: np.ndarray
    latents: np.ndarray
    terminal_score: float
    final_success: bool
    final_mode: str | None
    failure_reason: str | None
    collided_step: int | None

    @property
    def prefix_length(self) -> int:
        return self.generation_step

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["states"] = _rounded(self.states)
        record["actions"] = _rounded(self.actions)
        record["latents"] = _rounded(self.latents)
        record["state_prefix"] = record["states"][: self.generation_step + 1]
        record["action_prefix"] = record["actions"][: self.generation_step]
        record["latent_prefix"] = record["latents"][: self.generation_step]
        return record


@dataclass
class CandidateSet:
    policy: str
    episode_id: int
    candidates: list[Candidate]
    predicted_bottleneck_step: int | None
    detector_diagnostics: dict[str, Any]
    split_steps: list[int]

    def successful_modes(self) -> set[str]:
        return {c.final_mode for c in self.candidates if c.final_success and c.final_mode is not None}

    def selected(self) -> Candidate:
        return max(self.candidates, key=lambda c: c.terminal_score)
