"""Navigation task — reach a target location using reduced descending control.

The agent must navigate the fly (or avatar) to a randomly placed target.
Rewards are shaped by distance reduction and stability maintenance.
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
class NavigationTask(WorldInterface):
    """Navigate to a target within an episode budget.

    Reward = −distance_to_target + stability_bonus.
    Terminates when the agent reaches the target within ``success_radius_mm``.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    _rng: np.random.Generator = field(init=False)
    _target_xy: np.ndarray = field(init=False)
    _prev_distance: float = field(default=0.0, init=False)
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
        self._prev_distance = float(
            np.linalg.norm(self._target_xy - raw_feedback.body_positions[0, :2])
        )
        self._step_count = 0
        return WorldState(
            mode="navigation_task",
            step_count=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_xy - raw_feedback.body_positions[0, :2],
            },
        )

    def step(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del command
        self._step_count += 1
        thorax_xy = raw_feedback.body_positions[0, :2]
        target_vector = self._target_xy - thorax_xy
        distance = float(np.linalg.norm(target_vector))
        # Reward: distance reduction + stability bonus.
        distance_improvement = self._prev_distance - distance
        stability_bonus = 0.1 * summary.features.get("stability", 0.0)
        reward = distance_improvement + stability_bonus
        self._prev_distance = distance
        success = distance <= self.config.success_radius_mm
        return WorldState(
            mode="navigation_task",
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
                "distance_improvement": distance_improvement,
                "external_world_event": False,
            },
        )
