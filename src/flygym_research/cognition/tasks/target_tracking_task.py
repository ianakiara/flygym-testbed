"""Target tracking task — follow a moving target.

The target drifts each step with optional random perturbations.  The agent
must continuously adjust its heading and speed to maintain proximity.
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
class TargetTrackingTask(WorldInterface):
    """Track a drifting target.

    Reward = −distance + tracking_bonus for staying within a proximity band.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    drift_speed: float = 0.15
    drift_noise: float = 0.05
    _rng: np.random.Generator = field(init=False)
    _target_xy: np.ndarray = field(init=False)
    _drift_dir: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._target_xy = np.zeros(2, dtype=np.float64)
        self._drift_dir = np.array([1.0, 0.0], dtype=np.float64)

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        self._target_xy = raw_feedback.body_positions[0, :2] + self._rng.uniform(
            1.0, 2.0, size=2
        )
        angle = self._rng.uniform(0, 2 * np.pi)
        self._drift_dir = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        self._step_count = 0
        return self._make_state(raw_feedback, reward=0.0, terminated=False)

    def step(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del command, summary
        self._step_count += 1
        # Drift the target.
        self._target_xy += self._drift_dir * self.drift_speed
        self._target_xy += self._rng.normal(0.0, self.drift_noise, size=2)
        # Occasionally change drift direction.
        if self._step_count % 11 == 0:
            angle = self._rng.uniform(0, 2 * np.pi)
            self._drift_dir = np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float64
            )
        thorax_xy = raw_feedback.body_positions[0, :2]
        distance = float(np.linalg.norm(self._target_xy - thorax_xy))
        tracking_bonus = 1.0 if distance < self.config.success_radius_mm * 3 else 0.0
        reward = -distance + tracking_bonus
        return WorldState(
            mode="target_tracking_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,  # Tracking tasks never terminate early.
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_xy - thorax_xy,
            },
            info={
                "distance_to_target_mm": distance,
                "tracking_bonus": tracking_bonus,
                "external_world_event": self._step_count % 11 == 0,
            },
        )

    def _make_state(
        self,
        raw_feedback: RawBodyFeedback,
        *,
        reward: float,
        terminated: bool,
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="target_tracking_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_xy - thorax_xy,
            },
            info={"external_world_event": False},
        )
