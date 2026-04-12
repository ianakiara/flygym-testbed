"""Observation adapter — converts raw body feedback into brain-friendly summaries.

This adapter sits between the body/reflex layer and the brain layer, composing
the ascending summary with world state and history into a single
:class:`BrainObservation`.  It is designed to be composable: callers may swap
the underlying :class:`AscendingAdapter` or change the history length without
touching the brain controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import EnvConfig
from ..interfaces import (
    AscendingSummary,
    BrainObservation,
    RawBodyFeedback,
    WorldState,
)


@dataclass(slots=True)
class ObservationAdapter:
    """Compose raw feedback, ascending summary, world state, and history
    into a single :class:`BrainObservation` consumed by brain controllers.

    Parameters
    ----------
    config : EnvConfig
        Environment config controlling history length.
    flatten : bool
        If *True*, :meth:`flatten_observation` returns a 1-D numpy vector
        suitable for simple feed-forward controllers.
    """

    config: EnvConfig = field(default_factory=EnvConfig)
    flatten: bool = False
    _history: deque[dict[str, float]] = field(init=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.config.history_length)

    def reset(self) -> None:
        """Clear observation history."""
        self._history.clear()

    def compose(
        self,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
        world_state: WorldState,
    ) -> BrainObservation:
        """Build a :class:`BrainObservation` and record history."""
        self._history.append(
            {
                "reward": float(world_state.reward),
                "stability": float(summary.features.get("stability", 0.0)),
                "step_count": float(world_state.step_count),
            }
        )
        return BrainObservation(
            raw_body=raw_feedback,
            summary=summary,
            world=world_state,
            history=tuple(self._history),
        )

    def flatten_observation(self, obs: BrainObservation) -> np.ndarray:
        """Convert a :class:`BrainObservation` into a flat numpy vector.

        Layout: [summary_features | world_observables_flat | history_flat].
        Useful for simple controllers that expect a 1-D input.
        """
        parts: list[float] = []
        # Summary features in sorted-key order for determinism.
        for key in sorted(obs.summary.features):
            val = obs.summary.features[key]
            parts.append(float(val) if np.isfinite(val) else 0.0)
        # World observables — flatten any arrays.
        for key in sorted(obs.world.observables):
            val = obs.world.observables[key]
            if isinstance(val, np.ndarray):
                parts.extend(float(v) for v in val.flat)
            elif isinstance(val, (int, float)):
                parts.append(float(val))
        # History — flatten each dict in order.
        for entry in obs.history:
            for key in sorted(entry):
                parts.append(float(entry[key]))
        return np.array(parts, dtype=np.float64)

    def observation_spec(self) -> dict[str, Any]:
        """Return a human-readable description of the observation layout."""
        return {
            "raw_body_fields": [
                "joint_angles",
                "joint_velocities",
                "body_positions",
                "body_rotations",
                "contact_active",
                "contact_forces",
                "actuator_forces",
            ],
            "summary_features": "dict[str, float] — ascending summary features",
            "world_observables": "dict[str, Any] — world-mode-specific observables",
            "history_length": self.config.history_length,
            "flatten_available": self.flatten,
        }
