from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig
from .interfaces import (
    AscendingSummary,
    DescendingCommand,
    RawBodyFeedback,
    WorldInterface,
    WorldState,
)


@dataclass(slots=True)
class NativePhysicalWorld(WorldInterface):
    config: EnvConfig = field(default_factory=EnvConfig)
    _start_thorax: np.ndarray | None = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del seed, summary
        self._start_thorax = raw_feedback.body_positions[0, :2].copy()
        self._step_count = 0
        return WorldState(
            mode="native_physical",
            step_count=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            observables={
                "thorax_xy_mm": raw_feedback.body_positions[0, :2].copy(),
                "target_vector": np.array([1.0, 0.0], dtype=np.float64),
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
        delta_x = float(thorax_xy[0] - self._start_thorax[0])
        reward = delta_x + 0.1 * summary.features.get("stability", 0.0)
        truncated = self._step_count >= self.config.episode_steps
        return WorldState(
            mode="native_physical",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=truncated,
            observables={
                "thorax_xy_mm": thorax_xy.copy(),
                "target_vector": np.array([1.0, 0.0], dtype=np.float64),
            },
            info={"forward_progress_mm": delta_x, "external_world_event": False},
        )


@dataclass(slots=True)
class SimplifiedEmbodiedWorld(WorldInterface):
    config: EnvConfig = field(default_factory=EnvConfig)
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
            2.0, 4.0, size=2
        )
        self._step_count = 0
        target_vector = self._target_xy - raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="simplified_embodied",
            step_count=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": target_vector,
            },
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
        success = distance <= self.config.success_radius_mm
        reward = -distance
        return WorldState(
            mode="simplified_embodied",
            step_count=self._step_count,
            reward=reward,
            terminated=success,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_xy_mm": self._target_xy.copy(),
                "target_vector": target_vector,
            },
            info={"distance_to_target_mm": distance, "external_world_event": False},
        )


@dataclass(slots=True)
class AvatarRemappedWorld(WorldInterface):
    config: EnvConfig = field(default_factory=EnvConfig)
    _rng: np.random.Generator = field(init=False)
    _avatar_xy: np.ndarray = field(init=False)
    _target_xy: np.ndarray = field(init=False)
    _heading: float = field(default=0.0, init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._avatar_xy = np.zeros(2, dtype=np.float64)
        self._target_xy = np.zeros(2, dtype=np.float64)

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del raw_feedback, summary
        self._rng = np.random.default_rng(seed)
        self._avatar_xy = np.zeros(2, dtype=np.float64)
        self._target_xy = self._rng.uniform(-2.0, 2.0, size=2)
        self._heading = 0.0
        self._step_count = 0
        return WorldState(
            mode="avatar_remapped",
            step_count=0,
            reward=0.0,
            terminated=False,
            truncated=False,
            observables={
                "avatar_xy": self._avatar_xy.copy(),
                "heading": self._heading,
                "target_vector": self._target_xy - self._avatar_xy,
            },
        )

    def step(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del raw_feedback
        self._step_count += 1
        stability = summary.features.get("stability", 0.0)
        self._heading += command.turn_intent * self.config.avatar_turn_scale
        step_scale = self.config.avatar_step_scale * (0.25 + 0.75 * max(stability, 0.0))
        forward = np.array(
            [np.cos(self._heading), np.sin(self._heading)], dtype=np.float64
        )
        self._avatar_xy += forward * command.move_intent * step_scale
        external_event = self._step_count % 7 == 0
        if external_event:
            self._avatar_xy += self._rng.normal(
                0.0, self.config.avatar_noise_scale, size=2
            )
        target_vector = self._target_xy - self._avatar_xy
        distance = float(np.linalg.norm(target_vector))
        success = distance <= 0.2
        reward = -distance + 0.2 * stability
        return WorldState(
            mode="avatar_remapped",
            step_count=self._step_count,
            reward=reward,
            terminated=success,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "avatar_xy": self._avatar_xy.copy(),
                "heading": self._heading,
                "target_vector": target_vector,
                "partial_observation": np.clip(target_vector, -1.0, 1.0),
            },
            info={
                "distance_to_target": distance,
                "external_world_event": external_event,
            },
        )
