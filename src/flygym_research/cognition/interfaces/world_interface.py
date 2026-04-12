from __future__ import annotations

from abc import ABC, abstractmethod

from .types import AscendingSummary, DescendingCommand, RawBodyFeedback, WorldState


class WorldInterface(ABC):
    @abstractmethod
    def reset(
        self,
        seed: int | None,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        """Reset world state around the current body state."""

    @abstractmethod
    def step(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        summary: AscendingSummary,
    ) -> WorldState:
        """Advance world dynamics and return the next task state."""
