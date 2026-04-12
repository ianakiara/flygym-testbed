"""Distractor cue recall task — a true cue, then a misleading distractor.

This is the *hard* version of :class:`HiddenCueRecallTask`.  It has three
phases:

1. **Cue phase** (steps 0–cue_visible_steps):
   True cue is shown (0 = left, 1 = right).  ``target_vector`` points to
   the correct target.

2. **Distractor phase** (steps cue_visible_steps+1 – distractor_end):
   A *distractor* cue is shown — always the **opposite** of the true cue.
   ``target_vector`` now points toward the **wrong** target.  A scalar
   memory trace (exponential average) will be corrupted by this phase.

3. **Ambiguous phase** (steps > distractor_end):
   Both cue and distractor are hidden.  ``target_vector`` = midpoint
   (ambiguous).  ``cue_signal`` = 0.5.

A memory controller that retains the full observation buffer can recall the
original cue from phase 1.  A reactive controller with only a scalar trace
will have its trace *overwritten* by the distractor in phase 2, so it will
either guess randomly or follow the wrong direction.
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
class DistractorCueRecallTask(WorldInterface):
    """Navigation task with a true cue followed by a misleading distractor.

    Parameters
    ----------
    cue_visible_steps : int
        Number of steps the *true* cue is visible.
    distractor_steps : int
        Number of steps the *distractor* (opposite) cue is visible after the
        true cue disappears.
    target_distance : float
        Distance from origin to each candidate target.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    cue_visible_steps: int = 3
    distractor_steps: int = 4
    target_distance: float = 2.5
    _rng: np.random.Generator = field(init=False)
    _cue: int = field(default=0, init=False)
    _targets: list[np.ndarray] = field(init=False)
    _step_count: int = field(default=0, init=False)
    _origin: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._targets = []
        self._origin = np.zeros(2, dtype=np.float64)

    @property
    def _distractor_end(self) -> int:
        return self.cue_visible_steps + self.distractor_steps

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        self._origin = raw_feedback.body_positions[0, :2].copy()
        self._cue = int(self._rng.integers(0, 2))
        d = self.target_distance
        self._targets = [
            self._origin + np.array([-d, 0.0], dtype=np.float64),
            self._origin + np.array([d, 0.0], dtype=np.float64),
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

        correct_target = self._targets[self._cue]
        wrong_target = self._targets[1 - self._cue]

        dist_correct = float(np.linalg.norm(correct_target - thorax_xy))
        dist_wrong = float(np.linalg.norm(wrong_target - thorax_xy))

        reward = -dist_correct
        reached_correct = dist_correct <= self.config.success_radius_mm
        reached_wrong = dist_wrong <= self.config.success_radius_mm
        if reached_correct:
            reward += 10.0
        if reached_wrong:
            reward -= 5.0

        terminated = reached_correct or reached_wrong

        return WorldState(
            mode="distractor_cue_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_vector": self._phase_target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": self._phase_cue_signal(),
                "cue_visible": self._step_count <= self.cue_visible_steps,
                "distractor_active": (
                    self.cue_visible_steps < self._step_count <= self._distractor_end
                ),
            },
            info={
                "cue": self._cue,
                "phase": self._current_phase(),
                "dist_correct": dist_correct,
                "dist_wrong": dist_wrong,
                "reached_correct": reached_correct,
                "reached_wrong": reached_wrong,
                "external_world_event": False,
            },
        )

    def _current_phase(self) -> str:
        if self._step_count <= self.cue_visible_steps:
            return "cue"
        if self._step_count <= self._distractor_end:
            return "distractor"
        return "ambiguous"

    def _phase_cue_signal(self) -> float:
        """Return the cue signal for the current phase.

        - Cue phase: true cue (0.0 or 1.0)
        - Distractor phase: opposite of true cue (misleading!)
        - Ambiguous phase: 0.5 (uninformative)
        """
        if self._step_count <= self.cue_visible_steps:
            return float(self._cue)
        if self._step_count <= self._distractor_end:
            return float(1 - self._cue)  # distractor = opposite
        return 0.5

    def _phase_target_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        """Return target_vector for the current phase.

        - Cue phase: points to the correct target.
        - Distractor phase: points to the WRONG target (misleading!).
        - Ambiguous phase: midpoint (uninformative).
        """
        if self._step_count <= self.cue_visible_steps:
            return self._targets[self._cue] - thorax_xy
        if self._step_count <= self._distractor_end:
            return self._targets[1 - self._cue] - thorax_xy  # wrong target!
        midpoint = (self._targets[0] + self._targets[1]) / 2.0
        return midpoint - thorax_xy

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="distractor_cue_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._phase_target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": float(self._cue),
                "cue_visible": True,
                "distractor_active": False,
            },
            info={"external_world_event": False},
        )
