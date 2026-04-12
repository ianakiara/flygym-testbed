"""Same-state-different-history task — the hardest memory benchmark.

At step *k* a hidden event (A or B) occurs, visible for only 2 steps.
After step k+2 all observations are **identical** regardless of which
event occurred.  At step k+delay a decision point appears; the correct
action depends on the earlier event.

Why scalar trace fails
    By step k+delay the exponential-average trace has decayed/averaged
    away the event signal — current observations are identical for both
    event types, so a reactive controller is at chance (≈50 %).

Why slot memory succeeds
    The event is stored in a protected slot with high importance (novel
    signal with clear cue).  ``delay`` steps later the decision-point
    query retrieves it via attention.
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
class SameStateDifferentHistoryTask(WorldInterface):
    """Task where correct action depends on a long-gone hidden event.

    Parameters
    ----------
    event_step : int
        Step at which the hidden event first appears.
    event_visible_steps : int
        Number of steps the event is observable (default 2).
    decision_delay : int
        Gap between event disappearance and decision point.
    target_distance : float
        Distance from origin to each candidate target.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    event_step: int = 3
    event_visible_steps: int = 2
    decision_delay: int = 25
    target_distance: float = 2.5

    _rng: np.random.Generator = field(init=False)
    _event_type: int = field(default=0, init=False)  # 0 = A (left), 1 = B (right)
    _targets: list[np.ndarray] = field(init=False)
    _step_count: int = field(default=0, init=False)
    _origin: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._targets = []
        self._origin = np.zeros(2, dtype=np.float64)

    @property
    def _event_end(self) -> int:
        return self.event_step + self.event_visible_steps

    @property
    def _decision_step(self) -> int:
        return self._event_end + self.decision_delay

    # ── WorldInterface ───────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        self._origin = raw_feedback.body_positions[0, :2].copy()
        self._event_type = int(self._rng.integers(0, 2))
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

        correct_target = self._targets[self._event_type]
        wrong_target = self._targets[1 - self._event_type]

        dist_correct = float(np.linalg.norm(correct_target - thorax_xy))
        dist_wrong = float(np.linalg.norm(wrong_target - thorax_xy))

        in_decision = self._step_count >= self._decision_step
        reward = -dist_correct * 0.1  # mild shaping

        reached_correct = in_decision and dist_correct <= self.config.success_radius_mm
        reached_wrong = in_decision and dist_wrong <= self.config.success_radius_mm

        if reached_correct:
            reward += 10.0
        if reached_wrong:
            reward -= 5.0

        terminated = reached_correct or reached_wrong

        return WorldState(
            mode="same_state_different_history_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_vector": self._target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": self._cue_signal(),
                "cue_visible": self._is_event_visible(),
                "in_decision_phase": in_decision,
            },
            info={
                "event_type": self._event_type,
                "phase": self._current_phase(),
                "dist_correct": dist_correct,
                "dist_wrong": dist_wrong,
                "reached_correct": reached_correct,
                "reached_wrong": reached_wrong,
                "distance_to_target": dist_correct,
                "external_world_event": False,
            },
        )

    # ── Phase logic ──────────────────────────────────────────────────────

    def _is_event_visible(self) -> bool:
        return self.event_step <= self._step_count < self._event_end

    def _current_phase(self) -> str:
        if self._step_count < self.event_step:
            return "pre_event"
        if self._step_count < self._event_end:
            return "event"
        if self._step_count < self._decision_step:
            return "waiting"
        return "decision"

    def _cue_signal(self) -> float:
        """Return cue signal: informative only during event window."""
        if self._is_event_visible():
            return float(self._event_type)  # 0.0 or 1.0
        return 0.5  # uninformative

    def _target_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        """Return target vector.

        - Pre-event / waiting: midpoint (ambiguous — no directional info).
        - Event visible: points toward the correct target.
        - Decision phase: midpoint (must recall event to disambiguate).
        """
        if self._is_event_visible():
            return self._targets[self._event_type] - thorax_xy
        # All other phases: ambiguous midpoint
        midpoint = (self._targets[0] + self._targets[1]) / 2.0
        return midpoint - thorax_xy

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="same_state_different_history_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": self._cue_signal(),
                "cue_visible": self._is_event_visible(),
                "in_decision_phase": False,
            },
            info={
                "event_type": self._event_type,
                "phase": self._current_phase(),
                "distance_to_target": 0.0,
                "external_world_event": False,
            },
        )
