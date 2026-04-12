from __future__ import annotations

from abc import ABC, abstractmethod

from .types import BrainObservation, DescendingCommand, RawControlCommand


class BrainInterface(ABC):
    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Reset controller state."""

    @abstractmethod
    def act(
        self, observation: BrainObservation
    ) -> DescendingCommand | RawControlCommand:
        """Produce the next descending or direct-control command."""

    @abstractmethod
    def get_internal_state(self) -> dict[str, float]:
        """Return a compact numeric view of persistent controller state."""
