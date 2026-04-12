"""Branch-dependent recall task — episode branches on a hidden signal.

At step *k* a hidden branch signal (left/right) is visible for 2 steps.
Both branches produce **identical observations** from step k+5 onward.
The final action at step k+20 must match the branch taken.

Why scalar trace fails
    Identical observations from k+5 onward give identical trace values —
    the scalar trace cannot distinguish which branch was taken.

Why slot memory succeeds
    The branch signal is stored in a slot at step k with high importance.
    At the decision point the slot-attention read pathway retrieves it.
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
class BranchDependentRecallTask(WorldInterface):
    """Task where correct action depends on an earlier branch signal.

    Parameters
    ----------
    branch_step : int
        Step at which the branch signal appears.
    branch_visible_steps : int
        How many steps the branch signal is visible.
    convergence_steps : int
        Steps after branch disappears before observations fully converge.
    decision_delay : int
        Steps after convergence before the decision point.
    target_distance : float
        Distance from origin to each candidate target.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    branch_step: int = 3
    branch_visible_steps: int = 2
    convergence_steps: int = 3
    decision_delay: int = 15
    target_distance: float = 2.5

    _rng: np.random.Generator = field(init=False)
    _branch: int = field(default=0, init=False)  # 0 = left, 1 = right
    _targets: list[np.ndarray] = field(init=False)
    _step_count: int = field(default=0, init=False)
    _origin: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._targets = []
        self._origin = np.zeros(2, dtype=np.float64)

    @property
    def _branch_end(self) -> int:
        return self.branch_step + self.branch_visible_steps

    @property
    def _converge_step(self) -> int:
        return self._branch_end + self.convergence_steps

    @property
    def _decision_step(self) -> int:
        return self._converge_step + self.decision_delay

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
        self._branch = int(self._rng.integers(0, 2))
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

        correct_target = self._targets[self._branch]
        wrong_target = self._targets[1 - self._branch]
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
            mode="branch_dependent_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_vector": self._target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": self._cue_signal(),
                "cue_visible": self._is_branch_visible(),
                "in_decision_phase": in_decision,
            },
            info={
                "branch": self._branch,
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

    def _is_branch_visible(self) -> bool:
        return self.branch_step <= self._step_count < self._branch_end

    def _current_phase(self) -> str:
        if self._step_count < self.branch_step:
            return "pre_branch"
        if self._step_count < self._branch_end:
            return "branch_visible"
        if self._step_count < self._converge_step:
            return "converging"
        if self._step_count < self._decision_step:
            return "waiting"
        return "decision"

    def _cue_signal(self) -> float:
        """Return cue signal: informative only during branch window."""
        if self._is_branch_visible():
            return float(self._branch)  # 0.0 or 1.0
        # During convergence, signal fades linearly to 0.5
        if self._branch_end <= self._step_count < self._converge_step:
            progress = (self._step_count - self._branch_end) / max(self.convergence_steps, 1)
            return float(self._branch) * (1.0 - progress) + 0.5 * progress
        return 0.5  # fully ambiguous

    def _target_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        """Return target vector for current phase.

        - Pre-branch / waiting / decision: midpoint (ambiguous).
        - Branch visible: points toward correct target.
        - Converging: fades from correct toward midpoint.
        """
        midpoint = (self._targets[0] + self._targets[1]) / 2.0

        if self._is_branch_visible():
            return self._targets[self._branch] - thorax_xy

        if self._branch_end <= self._step_count < self._converge_step:
            progress = (self._step_count - self._branch_end) / max(self.convergence_steps, 1)
            correct_vec = self._targets[self._branch] - thorax_xy
            mid_vec = midpoint - thorax_xy
            return correct_vec * (1.0 - progress) + mid_vec * progress

        return midpoint - thorax_xy

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="branch_dependent_recall_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._target_vector(thorax_xy),
                "targets": [t.copy() for t in self._targets],
                "cue_signal": self._cue_signal(),
                "cue_visible": self._is_branch_visible(),
                "in_decision_phase": False,
            },
            info={
                "branch": self._branch,
                "phase": self._current_phase(),
                "distance_to_target": 0.0,
                "external_world_event": False,
            },
        )
