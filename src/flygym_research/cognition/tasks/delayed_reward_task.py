"""Delayed reward task — reward is withheld for *k* steps after the goal is reached.

Tests whether the controller can sustain goal-directed behaviour when
reinforcement is temporally displaced from the action that caused it.
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
class DelayedRewardTask(WorldInterface):
    """Navigate to a target but receive the reward only after a delay.

    Parameters
    ----------
    reward_delay : int
        Number of steps between reaching the goal zone and receiving the
        reward pulse.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    reward_delay: int = 5
    _rng: np.random.Generator = field(init=False)
    _target_xy: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)
    _reached_step: int | None = field(default=None, init=False)

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
        self._reached_step = None
        return self._make_state(raw_feedback, reward=0.0)

    def step(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del command, summary
        self._step_count += 1
        thorax_xy = raw_feedback.body_positions[0, :2]
        distance = float(np.linalg.norm(self._target_xy - thorax_xy))

        # Record when the agent first reaches the goal zone.
        if self._reached_step is None and distance <= self.config.success_radius_mm:
            self._reached_step = self._step_count

        # Deliver delayed reward.
        reward = 0.0
        if (
            self._reached_step is not None
            and self._step_count == self._reached_step + self.reward_delay
        ):
            reward = 10.0  # Delayed positive pulse.

        terminated = (
            self._reached_step is not None
            and self._step_count >= self._reached_step + self.reward_delay
        )
        return WorldState(
            mode="delayed_reward_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_xy - thorax_xy,
            },
            info={
                "distance_to_target_mm": distance,
                "reached_step": self._reached_step,
                "reward_delay": self.reward_delay,
                "external_world_event": False,
            },
        )

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="delayed_reward_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_xy - thorax_xy,
            },
            info={"external_world_event": False},
        )
