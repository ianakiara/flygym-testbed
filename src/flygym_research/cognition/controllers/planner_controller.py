"""Planner controller — uses lookahead subgoal generation to plan actions.

The planner maintains a queue of subgoals and generates actions toward the
current active subgoal.  When a subgoal is reached (or times out), the
planner advances to the next one.  This tests whether planning horizon
improves performance over reactive control.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..interfaces import BrainInterface, BrainObservation, DescendingCommand


@dataclass(slots=True)
class SubGoal:
    """A single subgoal in the planner's queue."""

    target_xy: np.ndarray
    timeout: int
    steps_active: int = 0


@dataclass(slots=True)
class PlannerController(BrainInterface):
    """Controller with explicit planning via subgoal decomposition.

    Parameters
    ----------
    num_subgoals : int
        Number of intermediate subgoals to generate along the path to
        the final target.
    subgoal_timeout : int
        Maximum steps spent pursuing a single subgoal before advancing.
    subgoal_radius : float
        Distance threshold for considering a subgoal reached.
    replan_interval : int
        Re-generate the plan every *replan_interval* steps.
    """

    num_subgoals: int = 3
    subgoal_timeout: int = 20
    subgoal_radius: float = 0.5
    replan_interval: int = 15
    _plan: deque[SubGoal] = field(init=False)
    _step_count: int = field(default=0, init=False)
    _total_replans: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._plan = deque()
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self._plan.clear()
        self._step_count = 0
        self._total_replans = 0
        self._rng = np.random.default_rng(seed)

    def act(self, observation: BrainObservation) -> DescendingCommand:
        self._step_count += 1

        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        stability = observation.summary.features.get("stability", 0.0)

        # Replan if plan is empty or at replan interval.
        if (
            len(self._plan) == 0
            or self._step_count % self.replan_interval == 0
        ):
            self._generate_plan(target_vector)

        # Advance subgoal if reached or timed out.
        if len(self._plan) > 0:
            current = self._plan[0]
            current.steps_active += 1
            dist_to_subgoal = float(np.linalg.norm(current.target_xy))
            if (
                dist_to_subgoal <= self.subgoal_radius
                or current.steps_active >= current.timeout
            ):
                self._plan.popleft()

        # Navigate toward the current subgoal (or final target if plan empty).
        if len(self._plan) > 0:
            nav_target = self._plan[0].target_xy
        else:
            nav_target = target_vector

        move_intent = float(np.clip(nav_target[0], -1.0, 1.0))
        turn_intent = float(np.clip(nav_target[1], -1.0, 1.0))

        return DescendingCommand(
            move_intent=move_intent,
            turn_intent=turn_intent,
            speed_modulation=0.3,
            stabilization_priority=float(np.clip(stability + 0.3, 0.0, 1.0)),
            target_bias=(float(target_vector[0]), float(target_vector[1])),
        )

    def _generate_plan(self, target_vector: np.ndarray) -> None:
        """Decompose the path to the target into intermediate subgoals."""
        self._plan.clear()
        self._total_replans += 1
        if np.linalg.norm(target_vector) < 1e-6:
            return
        for i in range(1, self.num_subgoals + 1):
            fraction = i / (self.num_subgoals + 1)
            subgoal_xy = target_vector * fraction
            # Add small lateral noise to avoid trivially straight paths.
            noise = self._rng.normal(0.0, 0.05, size=2)
            subgoal_xy = subgoal_xy + noise
            self._plan.append(
                SubGoal(
                    target_xy=subgoal_xy,
                    timeout=self.subgoal_timeout,
                )
            )

    def get_internal_state(self) -> dict[str, float]:
        return {
            "plan_length": float(len(self._plan)),
            "step_count": float(self._step_count),
            "total_replans": float(self._total_replans),
            "current_subgoal_steps": (
                float(self._plan[0].steps_active) if self._plan else 0.0
            ),
        }
