from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EnvConfig
from ..interfaces import AscendingSummary, DescendingCommand, RawBodyFeedback, WorldInterface, WorldState


@dataclass(slots=True)
class DistractorCueRecallTask(WorldInterface):
    config: EnvConfig = field(default_factory=EnvConfig)
    cue_duration: int = 4
    distractor_start: int = 7
    _rng: np.random.Generator = field(init=False)
    _cue_target_xy: np.ndarray = field(init=False)
    _distractor_xy: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._cue_target_xy = np.zeros(2, dtype=np.float64)
        self._distractor_xy = np.zeros(2, dtype=np.float64)

    def reset(self, seed: int | None, raw_feedback: RawBodyFeedback, summary: AscendingSummary) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        thorax_xy = raw_feedback.body_positions[0, :2]
        cue_dir = self._rng.choice([-1.0, 1.0])
        self._cue_target_xy = thorax_xy + np.array([1.8 * cue_dir, 0.8], dtype=np.float64)
        self._distractor_xy = thorax_xy + np.array([-1.8 * cue_dir, -0.8], dtype=np.float64)
        self._step_count = 0
        return self._make_state(raw_feedback, reward=0.0)

    def step(self, command: DescendingCommand, raw_feedback: RawBodyFeedback, summary: AscendingSummary) -> WorldState:
        del command, summary
        self._step_count += 1
        thorax_xy = raw_feedback.body_positions[0, :2]
        true_distance = float(np.linalg.norm(self._cue_target_xy - thorax_xy))
        distractor_distance = float(np.linalg.norm(self._distractor_xy - thorax_xy))
        reward = -true_distance + 0.3 * max(distractor_distance - true_distance, 0.0)
        success = true_distance <= self.config.success_radius_mm
        return WorldState(
            mode="distractor_cue_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=success,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._active_target(thorax_xy),
                "target_vector": self._active_target(thorax_xy) - thorax_xy,
                "cue_vector": self._cue_vector(thorax_xy),
            },
            info={
                "true_distance_mm": true_distance,
                "distractor_distance_mm": distractor_distance,
                "external_world_event": self._step_count >= self.distractor_start,
                "distractor_active": self._step_count >= self.distractor_start,
            },
        )

    def _active_target(self, thorax_xy: np.ndarray) -> np.ndarray:
        if self._step_count >= self.distractor_start:
            return self._distractor_xy.copy()
        return self._cue_target_xy.copy()

    def _cue_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        if self._step_count >= self.cue_duration:
            return np.zeros(2, dtype=np.float64)
        return self._cue_target_xy - thorax_xy

    def _make_state(self, raw_feedback: RawBodyFeedback, *, reward: float) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="distractor_cue_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_xy_mm": self._cue_target_xy.copy(),
                "target_vector": self._cue_target_xy - thorax_xy,
                "cue_vector": self._cue_target_xy - thorax_xy,
            },
            info={"external_world_event": False, "distractor_active": False},
        )
