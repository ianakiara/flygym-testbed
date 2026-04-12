from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import (
    AdapterOutput,
    AscendingSummary,
    DescendingCommand,
    RawBodyFeedback,
    RawControlCommand,
    WorldState,
)


class BodyInterface(ABC):
    @abstractmethod
    def reset(
        self, seed: int | None = None
    ) -> tuple[RawBodyFeedback, AscendingSummary]:
        """Reset the body/reflex substrate."""

    @abstractmethod
    def step(
        self,
        command: DescendingCommand,
        world_state: WorldState | None = None,
    ) -> tuple[RawBodyFeedback, AscendingSummary, AdapterOutput]:
        """Advance the substrate using a reduced descending command."""

    @abstractmethod
    def step_raw(
        self,
        command: RawControlCommand,
        world_state: WorldState | None = None,
    ) -> tuple[RawBodyFeedback, AscendingSummary, dict[str, Any]]:
        """Advance the substrate using direct low-level control."""

    @abstractmethod
    def get_action_spec(self) -> dict[str, tuple[float, float]]:
        """Return the reduced descending action ranges."""

    @abstractmethod
    def get_internal_state(self) -> dict[str, float]:
        """Return compact body/reflex state summaries."""
