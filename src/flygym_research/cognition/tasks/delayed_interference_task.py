from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EnvConfig
from ..interfaces import AscendingSummary, DescendingCommand, RawBodyFeedback, WorldInterface, WorldState


@dataclass(slots=True)
class DelayedInterferenceTask(WorldInterface):
    config: EnvConfig = field(default_factory=EnvConfig)
    reward_delay: int = 5
    interference_start: int = 6
    _rng: np.random.Generator = field(init=False)
    _target_xy: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)
    _reached_step: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._target_xy = np.zeros(2, dtype=np.float64)

    def reset(self, seed: int | None, raw_feedback: RawBodyFeedback, summary: AscendingSummary) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        thorax_xy = raw_feedback.body_positions[0, :2]
        self._target_xy = thorax_xy + self._rng.uniform(1.2, 2.2, size=2)
        self._step_count = 0
        self._reached_step = None
        return self._make_state(raw_feedback, reward=0.0)

    def step(self, command: DescendingCommand, raw_feedback: RawBodyFeedback, summary: AscendingSummary) -> WorldState:
        del command, summary
        self._step_count += 1
        thorax_xy = raw_feedback.body_positions[0, :2]
        distance = float(np.linalg.norm(self._target_xy - thorax_xy))
        if self._reached_step is None and distance <= self.config.success_radius_mm:
            self._reached_step = self._step_count
        reward = 0.0
        if self._reached_step is not None and self._step_count == self._reached_step + self.reward_delay:
            reward = 8.0
        interference = self._step_count >= self.interference_start
        reward -= 0.1 * float(interference)
        terminated = self._reached_step is not None and self._step_count >= self._reached_step + self.reward_delay
        return WorldState(
            mode="delayed_interference_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_vector(thorax_xy, interference),
                "cue_vector": self._target_xy - thorax_xy if self._step_count <= 2 else np.zeros(2, dtype=np.float64),
            },
            info={
                "distance_to_target_mm": distance,
                "external_world_event": interference,
                "distractor_active": interference,
                "reward_delay": self.reward_delay,
            },
        )

    def _target_vector(self, thorax_xy: np.ndarray, interference: bool) -> np.ndarray:
        vector = self._target_xy - thorax_xy
        if not interference:
            return vector
        return vector + self._rng.normal(0.0, 0.4, size=2)

    def _make_state(self, raw_feedback: RawBodyFeedback, *, reward: float) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="delayed_interference_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": self._target_xy - thorax_xy,
                "cue_vector": self._target_xy - thorax_xy,
            },
            info={"external_world_event": False, "distractor_active": False},
        )
