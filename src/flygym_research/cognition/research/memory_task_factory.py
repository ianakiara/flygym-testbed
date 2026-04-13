"""Memory task factory for PR #5 — 6 task families for deep memory benchmark.

Extends the existing 4 task families with:
  5. SelectiveRecallTask — multiple cues stored, only one matters later
  6. SparseRelevanceTask — few memory events matter among many distractors

All tasks use the WorldInterface protocol and work with _TaskEnvWrapper.
"""

from __future__ import annotations

import numpy as np

from ..config import EnvConfig
from ..interfaces import (
    AscendingSummary,
    RawBodyFeedback,
    WorldInterface,
    WorldState,
)

# Re-export existing task families for convenience
from ..experiments.exp_true_memory_benchmark import (
    ConflictingUpdatesTask,
    DelayedDisambiguationTask,
    MultiStageDependencyTask,
    MultiVariableMemoryTask,
)


# ---------------------------------------------------------------------------
# Task 5: Selective Recall
# ---------------------------------------------------------------------------

class SelectiveRecallTask(WorldInterface):
    """Multiple cues stored, but only one matters at decision time.

    Phase 1 (0–30%): Present 3 cues (A, B, C) sequentially.
    Phase 2 (30–70%): Distractors — random noise cues.
    Phase 3 (70–100%): Query phase — reveal which cue (A/B/C) to recall.
    Correct action depends on the VALUE of the queried cue.

    This specifically tests selective retrieval from memory.
    """

    def __init__(self, *, config: EnvConfig | None = None) -> None:
        self._config = config or EnvConfig(episode_steps=150)
        self._step = 0
        self._cues: dict[str, float] = {}
        self._query_key: str = ""
        self._rng = np.random.default_rng(0)

    def reset(
        self, seed: int, raw_body: RawBodyFeedback, ascending: AscendingSummary,
    ) -> WorldState:
        self._rng = np.random.default_rng(seed)
        self._step = 0
        # Generate 3 distinct cues
        self._cues = {
            "A": float(self._rng.choice([-1.0, 1.0])),
            "B": float(self._rng.choice([-1.0, 1.0])),
            "C": float(self._rng.choice([-1.0, 1.0])),
        }
        self._query_key = str(self._rng.choice(["A", "B", "C"]))
        return self._make_state(reward=0.0)

    def step(
        self, action: object, raw_body: RawBodyFeedback, ascending: AscendingSummary,
    ) -> WorldState:
        self._step += 1
        total = self._config.episode_steps
        frac = self._step / max(total, 1)

        move = float(getattr(action, "move_intent", 0.0))
        reward = 0.0

        if frac > 0.7:
            # Decision phase — correct action matches queried cue value
            correct_dir = self._cues[self._query_key]
            if abs(move - correct_dir) < 0.5:
                reward = 10.0
            else:
                reward = -5.0

        return self._make_state(reward=reward)

    def _make_state(self, *, reward: float) -> WorldState:
        total = self._config.episode_steps
        frac = self._step / max(total, 1)

        # Build cue vector based on phase
        if frac <= 0.1:
            cue_vector = np.array([self._cues["A"], 0.0])
            context = 1.0  # cue A
        elif frac <= 0.2:
            cue_vector = np.array([self._cues["B"], 0.0])
            context = 2.0  # cue B
        elif frac <= 0.3:
            cue_vector = np.array([self._cues["C"], 0.0])
            context = 3.0  # cue C
        elif frac <= 0.7:
            # Distractors
            cue_vector = self._rng.normal(0, 0.5, size=2)
            context = 0.0
        else:
            # Query phase — reveal which cue to recall
            query_idx = {"A": 1.0, "B": 2.0, "C": 3.0}[self._query_key]
            cue_vector = np.array([query_idx, 0.0])
            context = -query_idx  # negative = query mode

        return WorldState(
            mode="selective_recall",
            step_count=self._step,
            reward=reward,
            terminated=self._step >= total,
            truncated=False,
            observables={
                "target_vector": np.zeros(2, dtype=np.float64),
                "cue_vector": cue_vector.astype(np.float64),
                "context_key": context,
                "visited": [],
            },
            info={"external_world_event": False, "distractor_active": 0.3 < frac <= 0.7},
        )


# ---------------------------------------------------------------------------
# Task 6: Sparse Relevance
# ---------------------------------------------------------------------------

class SparseRelevanceTask(WorldInterface):
    """Only a few memory events matter among many distractors.

    Presents a stream of events. Only events with a specific "marker"
    are relevant.  At the end, the correct action depends on the COUNT
    or SUM of marked events.

    This tests the ability to filter and retain sparse relevant information.
    """

    def __init__(self, *, config: EnvConfig | None = None) -> None:
        self._config = config or EnvConfig(episode_steps=150)
        self._step = 0
        self._n_relevant = 0
        self._relevant_sum = 0.0
        self._rng = np.random.default_rng(0)
        self._events: list[tuple[bool, float]] = []

    def reset(
        self, seed: int, raw_body: RawBodyFeedback, ascending: AscendingSummary,
    ) -> WorldState:
        self._rng = np.random.default_rng(seed)
        self._step = 0
        total = self._config.episode_steps

        # Pre-generate event stream
        self._events = []
        for _ in range(total):
            is_relevant = self._rng.random() < 0.15  # 15% are relevant
            value = float(self._rng.uniform(-1, 1))
            self._events.append((is_relevant, value))

        self._n_relevant = sum(1 for r, _ in self._events if r)
        self._relevant_sum = sum(v for r, v in self._events if r)
        return self._make_state(reward=0.0)

    def step(
        self, action: object, raw_body: RawBodyFeedback, ascending: AscendingSummary,
    ) -> WorldState:
        self._step += 1
        total = self._config.episode_steps
        frac = self._step / max(total, 1)

        move = float(getattr(action, "move_intent", 0.0))
        reward = 0.0

        if frac > 0.9:
            # Decision phase — move positive if relevant_sum > 0, else negative
            correct_dir = 1.0 if self._relevant_sum > 0 else -1.0
            if abs(move - correct_dir) < 0.5:
                reward = 10.0
            else:
                reward = -5.0

        return self._make_state(reward=reward)

    def _make_state(self, *, reward: float) -> WorldState:
        total = self._config.episode_steps
        frac = self._step / max(total, 1)

        # Current event info
        if self._step < len(self._events):
            is_relevant, value = self._events[self._step]
        else:
            is_relevant, value = False, 0.0

        # Marker: relevant events have context_key = 1.0, others = 0.0
        marker = 1.0 if is_relevant else 0.0
        cue_vector = np.array([value, marker], dtype=np.float64)

        return WorldState(
            mode="sparse_relevance",
            step_count=self._step,
            reward=reward,
            terminated=self._step >= total,
            truncated=False,
            observables={
                "target_vector": np.zeros(2, dtype=np.float64),
                "cue_vector": cue_vector,
                "context_key": marker,
                "visited": [],
            },
            info={
                "external_world_event": is_relevant,
                "distractor_active": not is_relevant and frac < 0.9,
            },
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

ALL_TASK_FAMILIES: dict[str, type] = {
    "delayed_disambiguation": DelayedDisambiguationTask,
    "multi_stage_dependency": MultiStageDependencyTask,
    "conflicting_updates": ConflictingUpdatesTask,
    "multi_variable_memory": MultiVariableMemoryTask,
    "selective_recall": SelectiveRecallTask,
    "sparse_relevance": SparseRelevanceTask,
}


def create_task(
    name: str,
    *,
    config: EnvConfig | None = None,
) -> WorldInterface:
    """Create a task instance by name."""
    cls = ALL_TASK_FAMILIES.get(name)
    if cls is None:
        raise ValueError(f"Unknown task family: {name}")
    return cls(config=config)
