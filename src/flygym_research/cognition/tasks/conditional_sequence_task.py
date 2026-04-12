"""Conditional sequence task — multi-stage dependency with misleading reward.

Tests whether the controller can recall an earlier *context signal* to make
the correct decision at a later stage, even when an intermediate reward
misleads a memoryless agent.

Protocol:

1. **Context phase** (steps 0–context_steps):
   A context signal is shown (``context_id`` ∈ {0, 1}).  The agent
   navigates toward a *context zone* and receives a small intermediate
   reward for arriving — this reward is **the same regardless of context**.

2. **Decision phase** (steps > context_steps):
   Context signal disappears.  Two decision targets appear at equal
   distance.  The *correct* target depends on the earlier context:
     - context 0 → go left
     - context 1 → go right

   ``target_vector`` points to the midpoint (ambiguous).  A memoryless
   controller cannot distinguish the two cases.

The intermediate reward in the context phase is *misleading* because it
rewards approach to the context zone equally for both contexts, creating a
false sense that the task is already solved.  The real discriminating
reward comes only in the decision phase.
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
class ConditionalSequenceTask(WorldInterface):
    """Multi-stage task where the correct final action depends on an
    earlier context signal that is no longer observable.

    Parameters
    ----------
    context_steps : int
        Number of steps the context signal is visible.
    context_zone_distance : float
        Distance from origin to the context zone.
    decision_distance : float
        Distance from context zone to the two decision targets.
    misleading_reward : float
        Small positive reward given for reaching the context zone
        (same for both contexts — deliberately misleading).
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    context_steps: int = 5
    context_zone_distance: float = 1.5
    decision_distance: float = 2.5
    misleading_reward: float = 2.0
    _rng: np.random.Generator = field(init=False)
    _context_id: int = field(default=0, init=False)
    _context_zone: np.ndarray = field(init=False)
    _decision_targets: list[np.ndarray] = field(init=False)
    _step_count: int = field(default=0, init=False)
    _origin: np.ndarray = field(init=False)
    _reached_context: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._context_zone = np.zeros(2, dtype=np.float64)
        self._decision_targets = []
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

        self._context_id = int(self._rng.integers(0, 2))

        # Context zone: directly ahead.
        self._context_zone = self._origin + np.array(
            [0.0, self.context_zone_distance], dtype=np.float64
        )

        # Decision targets: left and right of context zone.
        d = self.decision_distance
        self._decision_targets = [
            self._context_zone + np.array([-d, 0.0], dtype=np.float64),  # left
            self._context_zone + np.array([d, 0.0], dtype=np.float64),   # right
        ]

        self._step_count = 0
        self._reached_context = False
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
        in_decision_phase = self._step_count > self.context_steps

        reward = 0.0

        # Context phase: reward for reaching context zone (misleading —
        # same reward regardless of context).
        if not in_decision_phase:
            dist_to_context = float(np.linalg.norm(self._context_zone - thorax_xy))
            if not self._reached_context and dist_to_context <= self.config.success_radius_mm:
                self._reached_context = True
                reward = self.misleading_reward  # same for both contexts!

        # Decision phase: reward depends on context_id.
        correct_target = self._decision_targets[self._context_id]
        wrong_target = self._decision_targets[1 - self._context_id]
        dist_correct = float(np.linalg.norm(correct_target - thorax_xy))
        dist_wrong = float(np.linalg.norm(wrong_target - thorax_xy))

        reached_correct = in_decision_phase and dist_correct <= self.config.success_radius_mm
        reached_wrong = in_decision_phase and dist_wrong <= self.config.success_radius_mm

        if reached_correct:
            reward += 10.0
        if reached_wrong:
            reward -= 5.0

        terminated = reached_correct or reached_wrong

        return WorldState(
            mode="conditional_sequence_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_vector": self._current_target_vector(thorax_xy),
                "targets": [t.copy() for t in self._decision_targets],
                "cue_signal": (
                    float(self._context_id)
                    if self._step_count <= self.context_steps
                    else 0.5
                ),
                "cue_visible": self._step_count <= self.context_steps,
                "context_zone": self._context_zone.copy(),
                "in_decision_phase": in_decision_phase,
            },
            info={
                "context_id": self._context_id,
                "phase": "context" if not in_decision_phase else "decision",
                "reached_context": self._reached_context,
                "dist_correct": dist_correct,
                "dist_wrong": dist_wrong,
                "reached_correct": reached_correct,
                "reached_wrong": reached_wrong,
                "external_world_event": False,
            },
        )

    def _current_target_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        """Return target vector based on current phase.

        - Context phase: point toward context zone (same for both contexts).
        - Decision phase: midpoint of decision targets (ambiguous).
        """
        if self._step_count <= self.context_steps:
            return self._context_zone - thorax_xy
        midpoint = (self._decision_targets[0] + self._decision_targets[1]) / 2.0
        return midpoint - thorax_xy

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="conditional_sequence_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._current_target_vector(thorax_xy),
                "targets": [t.copy() for t in self._decision_targets]
                if self._decision_targets
                else [],
                "cue_signal": float(self._context_id),
                "cue_visible": True,
                "context_zone": self._context_zone.copy(),
                "in_decision_phase": False,
            },
            info={"external_world_event": False},
        )
