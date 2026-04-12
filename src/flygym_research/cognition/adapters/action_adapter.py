"""Action adapter — converts brain action format into environment-consumable actions.

This adapter normalises and validates brain-layer outputs before they reach
the body/reflex layer.  It supports both :class:`DescendingCommand` (reduced
control) and :class:`RawControlCommand` (direct actuator control), and can
construct a :class:`DescendingCommand` from a flat numpy vector or a dictionary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..interfaces import DescendingCommand, RawControlCommand


# Canonical field order used by flat-vector conversion.
_DESCENDING_FIELDS = (
    "move_intent",
    "turn_intent",
    "speed_modulation",
    "stabilization_priority",
    "approach_avoid",
    "interaction_trigger",
)

_FIELD_RANGES: dict[str, tuple[float, float]] = {
    "move_intent": (-1.0, 1.0),
    "turn_intent": (-1.0, 1.0),
    "speed_modulation": (-1.0, 1.0),
    "stabilization_priority": (0.0, 1.0),
    "approach_avoid": (-1.0, 1.0),
    "interaction_trigger": (0.0, 1.0),
}


@dataclass(slots=True)
class ActionAdapter:
    """Validate, clip, and convert brain outputs into body-layer commands.

    Parameters
    ----------
    clip : bool
        If *True* (default), field values are clipped to their valid ranges
        before constructing the command.
    """

    clip: bool = True

    def from_descending(self, command: DescendingCommand) -> DescendingCommand:
        """Validate and optionally clip a :class:`DescendingCommand`."""
        if not self.clip:
            return command
        return DescendingCommand(
            move_intent=float(np.clip(command.move_intent, -1.0, 1.0)),
            turn_intent=float(np.clip(command.turn_intent, -1.0, 1.0)),
            speed_modulation=float(np.clip(command.speed_modulation, -1.0, 1.0)),
            stabilization_priority=float(
                np.clip(command.stabilization_priority, 0.0, 1.0)
            ),
            approach_avoid=float(np.clip(command.approach_avoid, -1.0, 1.0)),
            interaction_trigger=float(np.clip(command.interaction_trigger, 0.0, 1.0)),
            target_bias=command.target_bias,
            state_mode=command.state_mode,
            metadata=command.metadata,
        )

    def from_raw(self, command: RawControlCommand) -> RawControlCommand:
        """Pass-through for raw commands (no clipping — actuator limits
        are enforced by MuJoCo)."""
        return command

    def from_dict(self, d: dict[str, Any]) -> DescendingCommand:
        """Build a :class:`DescendingCommand` from a plain dictionary.

        Missing keys default to the :class:`DescendingCommand` defaults.
        """
        kwargs: dict[str, Any] = {}
        for field_name in _DESCENDING_FIELDS:
            if field_name in d:
                lo, hi = _FIELD_RANGES[field_name]
                val = float(d[field_name])
                kwargs[field_name] = float(np.clip(val, lo, hi)) if self.clip else val
        if "target_bias" in d:
            tb = d["target_bias"]
            kwargs["target_bias"] = (float(tb[0]), float(tb[1]))
        if "state_mode" in d:
            kwargs["state_mode"] = str(d["state_mode"])
        return DescendingCommand(**kwargs)

    def from_flat(self, vector: np.ndarray) -> DescendingCommand:
        """Build a :class:`DescendingCommand` from a flat numpy vector.

        Expected layout: ``[move_intent, turn_intent, speed_modulation,
        stabilization_priority, approach_avoid, interaction_trigger]``.
        Extra elements are silently ignored.
        """
        d: dict[str, float] = {}
        for i, field_name in enumerate(_DESCENDING_FIELDS):
            if i < len(vector):
                d[field_name] = float(vector[i])
        return self.from_dict(d)

    def to_flat(self, command: DescendingCommand) -> np.ndarray:
        """Flatten a :class:`DescendingCommand` into a numpy vector."""
        return np.array(
            [
                command.move_intent,
                command.turn_intent,
                command.speed_modulation,
                command.stabilization_priority,
                command.approach_avoid,
                command.interaction_trigger,
            ],
            dtype=np.float64,
        )

    def action_spec(self) -> dict[str, tuple[float, float]]:
        """Return field names and their valid ranges."""
        return dict(_FIELD_RANGES)
