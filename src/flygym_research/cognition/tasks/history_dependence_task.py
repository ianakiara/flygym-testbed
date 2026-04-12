"""History dependence task — create matched-current-state scenarios with
different histories to test whether history is a necessary state variable.

Two trajectories are engineered to arrive at the same coarse state but via
different paths (left-then-right vs right-then-left).  The task rewards
differ based on the path taken, forcing the agent to use history to
maximise return.
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
class HistoryDependenceTask(WorldInterface):
    """Navigation task where reward depends on the *path* taken, not just
    the current state.

    The agent starts at origin.  Two waypoints are placed symmetrically.
    The agent must visit them in a specific order (randomised each episode)
    to receive the full reward.  Visiting in the wrong order gives a penalty.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    waypoint_distance: float = 2.0
    _rng: np.random.Generator = field(init=False)
    _waypoints: list[np.ndarray] = field(init=False)
    _required_order: list[int] = field(init=False)
    _visited: list[bool] = field(init=False)
    _visit_order: list[int] = field(init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng()
        self._waypoints = []
        self._required_order = []
        self._visited = []
        self._visit_order = []

    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        origin = raw_feedback.body_positions[0, :2].copy()
        d = self.waypoint_distance
        self._waypoints = [
            origin + np.array([-d, 0.0], dtype=np.float64),
            origin + np.array([d, 0.0], dtype=np.float64),
        ]
        # Randomise required visitation order.
        self._required_order = list(self._rng.permutation(2))
        self._visited = [False, False]
        self._visit_order = []
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

        # Check proximity to each waypoint.
        reward = 0.0
        for i, wp in enumerate(self._waypoints):
            if not self._visited[i]:
                dist = float(np.linalg.norm(wp - thorax_xy))
                if dist <= self.config.success_radius_mm:
                    self._visited[i] = True
                    self._visit_order.append(i)
                    # Reward depends on whether this matches the required order.
                    expected_idx = len(self._visit_order) - 1
                    if expected_idx < len(self._required_order):
                        if self._required_order[expected_idx] == i:
                            reward += 5.0  # Correct order.
                        else:
                            reward -= 3.0  # Wrong order.

        all_visited = all(self._visited)
        return WorldState(
            mode="history_dependence_task",
            step_count=self._step_count,
            reward=reward,
            terminated=all_visited,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_vector": self._closest_unvisited_vector(thorax_xy),
                "waypoints": [wp.copy() for wp in self._waypoints],
                "visited": list(self._visited),
            },
            info={
                "visit_order": list(self._visit_order),
                "required_order": list(self._required_order),
                "external_world_event": False,
            },
        )

    def _closest_unvisited_vector(self, thorax_xy: np.ndarray) -> np.ndarray:
        """Return vector to the closest unvisited waypoint."""
        best_vec = np.zeros(2, dtype=np.float64)
        best_dist = float("inf")
        for i, wp in enumerate(self._waypoints):
            if not self._visited[i]:
                vec = wp - thorax_xy
                dist = float(np.linalg.norm(vec))
                if dist < best_dist:
                    best_dist = dist
                    best_vec = vec
        return best_vec

    def _make_state(
        self, raw_feedback: RawBodyFeedback, *, reward: float
    ) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="history_dependence_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._closest_unvisited_vector(thorax_xy),
                "waypoints": [wp.copy() for wp in self._waypoints],
                "visited": list(self._visited),
            },
            info={"external_world_event": False},
        )
