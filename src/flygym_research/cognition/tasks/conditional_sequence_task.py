from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EnvConfig
from ..interfaces import AscendingSummary, DescendingCommand, RawBodyFeedback, WorldInterface, WorldState


@dataclass(slots=True)
class ConditionalSequenceTask(WorldInterface):
    config: EnvConfig = field(default_factory=EnvConfig)
    context_duration: int = 3
    waypoint_distance: float = 1.8
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

    def reset(self, seed: int | None, raw_feedback: RawBodyFeedback, summary: AscendingSummary) -> WorldState:
        del summary
        self._rng = np.random.default_rng(seed)
        origin = raw_feedback.body_positions[0, :2].copy()
        d = self.waypoint_distance
        self._waypoints = [
            origin + np.array([-d, 0.8], dtype=np.float64),
            origin + np.array([d, -0.8], dtype=np.float64),
        ]
        self._required_order = list(self._rng.permutation(2))
        self._visited = [False, False]
        self._visit_order = []
        self._step_count = 0
        return self._make_state(raw_feedback, reward=0.0)

    def step(self, command: DescendingCommand, raw_feedback: RawBodyFeedback, summary: AscendingSummary) -> WorldState:
        del command, summary
        self._step_count += 1
        thorax_xy = raw_feedback.body_positions[0, :2]
        reward = 0.0
        for i, wp in enumerate(self._waypoints):
            if self._visited[i]:
                continue
            distance = float(np.linalg.norm(wp - thorax_xy))
            if distance <= self.config.success_radius_mm:
                self._visited[i] = True
                self._visit_order.append(i)
                expected = len(self._visit_order) - 1
                reward += 6.0 if self._required_order[expected] == i else -4.0
        terminated = all(self._visited)
        return WorldState(
            mode="conditional_sequence_task",
            step_count=self._step_count,
            reward=reward,
            terminated=terminated,
            truncated=self._step_count >= self.config.episode_steps,
            observables={
                "target_vector": self._closest_unvisited(thorax_xy),
                "context_key": self._context_key(),
                "visited": list(self._visited),
            },
            info={
                "visit_order": list(self._visit_order),
                "required_order": list(self._required_order),
                "external_world_event": False,
                "distractor_active": False,
            },
        )

    def _context_key(self) -> float:
        if self._step_count >= self.context_duration:
            return 0.0
        return -1.0 if self._required_order[:1] == [0] else 1.0

    def _closest_unvisited(self, thorax_xy: np.ndarray) -> np.ndarray:
        best_vec = np.zeros(2, dtype=np.float64)
        best_dist = float("inf")
        for i, wp in enumerate(self._waypoints):
            if self._visited[i]:
                continue
            vec = wp - thorax_xy
            dist = float(np.linalg.norm(vec))
            if dist < best_dist:
                best_vec = vec
                best_dist = dist
        return best_vec

    def _make_state(self, raw_feedback: RawBodyFeedback, *, reward: float) -> WorldState:
        thorax_xy = raw_feedback.body_positions[0, :2]
        return WorldState(
            mode="conditional_sequence_task",
            step_count=self._step_count,
            reward=reward,
            terminated=False,
            truncated=False,
            observables={
                "target_vector": self._closest_unvisited(thorax_xy),
                "context_key": self._context_key(),
                "visited": list(self._visited),
            },
            info={"external_world_event": False, "distractor_active": False},
        )
