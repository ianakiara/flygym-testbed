"""Hidden cue recall task — a cue appears briefly and then disappears.

The correct action at later steps depends on remembering the cue.
A reactive (memoryless) controller must guess, while a memory controller
can recall the cue and act accordingly.

This is the canonical test for whether memory actually matters: the
observation at decision time is *identical* regardless of the cue, so
the only way to succeed is to have retained the cue in internal state.
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
class HiddenCueRecallTask(WorldInterface):
    """Navigation task where the target depends on a briefly shown cue.

    Parameters
    ----------
    cue_visible_steps : int
        Number of steps the cue is visible at the start of the episode.
    target_distance : float
        Distance from origin to the two candidate targets.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    cue_visible_steps: int = 3
    target_distance: float = 2.5
    _rng: np.random.Generator = field(init=False)
    _cue: int = field(default=0, init=False)  # 0 = left target, 1 = right target
    _targets: list[np.ndarray] = field(init=False)
    _step_count: int = field(default=0, init=False)
    _origin: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._targets = []
        self._origin = np.zeros(2, dtype=np.float64)

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        self._origin = raw_feedback.body_positions[0, :2].copy()

        # Randomise cue: 0 = go left, 1 = go right.
        self._cue = int(self._rng.integers(0, 2))

        # Place two targets symmetrically.
        d = self.target_distance
        self._targets = [
            self._origin + np.array([-d, 0.0], dtype=np.float64),  # left
            self._origin + np.array([d, 0.0], dtype=np.float64),   # right
        ]
        self._step_count = 0
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

        # The correct target is determined by the cue.
        correct_target = self._targets[self._cue]
        wrong_target = self._targets[1 - self._cue]

        dist_correct = float(np.linalg.norm(correct_target - thorax_xy))
        dist_wrong = float(np.linalg.norm(wrong_target - thorax_xy))

        # Reward: positive for approaching correct target, negative for wrong.
        reward = -dist_correct

        # Bonus for reaching correct target.
        reached_correct = dist_correct <= self.config.success_radius_mm
        reached_wrong = dist_wrong <= self.config.success_radius_mm
        if reached_correct:
            reward += 10.0
        if reached_wrong:
            reward -= 5.0

        terminated = reached_correct or reached_wrong

        return WorldState(
            mode="hidden_cue_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                # Both targets are always visible — the question is which is correct.
                "target_vector": self._ambiguous_target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                # Cue is only visible during the first few steps.
                "cue_signal": float(self._cue) if self._step_count <= self.cue_visible_steps else 0.5,
                "cue_visible": self._step_count <= self.cue_visible_steps,
            },
            info={
                "cue": self._cue,
                "dist_correct": dist_correct,
                "dist_wrong": dist_wrong,
                "reached_correct": reached_correct,
                "reached_wrong": reached_wrong,
                "external_world_event": False,
            },
        )

    def _ambiguous_target_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        """When cue is hidden, return midpoint vector (ambiguous).

        When cue is visible, return vector toward the correct target.
        """
        if self._step_count <= self.cue_visible_steps:
            return self._targets[self._cue] - thorax_xy
        # After cue disappears, the observation is ambiguous — midpoint
        # of the two targets.  A memoryless controller cannot distinguish.
        midpoint = (self._targets[0] + self._targets[1]) / 2.0
        return midpoint - thorax_xy

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="hidden_cue_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._ambiguous_target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": float(self._cue),
                "cue_visible": True,
            },
            info={"external_world_event": False},
        )
