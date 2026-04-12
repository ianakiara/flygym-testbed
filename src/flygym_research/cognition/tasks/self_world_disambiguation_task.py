"""Self/world disambiguation task — the agent must distinguish self-induced
sensory changes from externally caused ones.

Periodically the world injects external perturbations (position jitter,
observation noise).  The agent must maintain a stable trajectory despite
these perturbations.  Metrics compare behaviour during self-driven and
externally perturbed windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EnvConfig
from ..interfaces import (
    AscendingSummary,
    DescendingCommand,
    RawBodyFeedback,
    WorldInterface,
    WorldState,
)


@dataclass(slots=True)
class SelfWorldDisambiguationTask(WorldInterface):
    """Navigate a target while the world occasionally injects perturbations.

    Parameters
    ----------
    perturbation_period : int
        Steps between external perturbation events.
    perturbation_magnitude : float
        Scale of observation-space noise added during perturbations.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    perturbation_period: int = 5
    perturbation_magnitude: float = 0.3
    _rng: np.random.Generator = field(init=False)
    _target_xy: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._target_xy = np.zeros(2, dtype=np.float64)

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        self._target_xy = raw_feedback.body_positions[0, :2] + self._rng.uniform(
            self.config.target_min_distance, self.config.target_max_distance, size=2
        )
        self._step_count = 0
        target_vector = self._target_xy - raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="self_world_disambiguation_task",
            step_count=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": target_vector,
            },
            info={"external_world_event": False, "perturbation_active": False},
        )

    def step(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del command, summary
        self._step_count += 1
        thorax_xy = raw_feedback.body_positions[0, :2]
        target_vector = self._target_xy - thorax_xy
        distance = float(np.linalg.norm(target_vector))

        # External perturbation injection.
        perturbation_active = self._step_count % self.perturbation_period == 0
        if perturbation_active:
            noise = self._rng.normal(0.0, self.perturbation_magnitude, size=2)
            target_vector = target_vector + noise  # Perturbed observation.

        reward = -distance + 0.1 * float(not perturbation_active)
        success = distance <= self.config.success_radius_mm
        return WorldState(
            mode="self_world_disambiguation_task",
            step_count=self._step_count,
            reward=reward,
            terminated=success,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": target_vector,
            },
            info={
                "distance_to_target_mm": distance,
                "external_world_event": perturbation_active,
                "perturbation_active": perturbation_active,
            },
        )
