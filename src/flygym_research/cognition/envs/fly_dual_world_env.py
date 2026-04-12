from __future__ import annotations

from ..config import EnvConfig
from ..interfaces import (
    BodyInterface,
    BrainObservation,
    DescendingCommand,
    RawControlCommand,
    StepTransition,
    WorldInterface,
)
from .fly_body_world_env import FlyBodyWorldEnv


class FlyDualWorldEnv(FlyBodyWorldEnv):
    def __init__(
        self,
        body: BodyInterface,
        worlds: dict[str, WorldInterface],
        *,
        default_mode: str = "native",
        config: EnvConfig | None = None,
    ) -> None:
        if default_mode not in worlds:
            raise ValueError(f"Unknown default mode: {default_mode}")
        self.worlds = worlds
        self.mode = default_mode
        super().__init__(body=body, world=worlds[default_mode], config=config)

    def set_mode(self, mode: str) -> None:
        if mode not in self.worlds:
            raise ValueError(f"Unknown world mode: {mode}")
        self.mode = mode
        self.world = self.worlds[mode]

    def reset(
        self, seed: int | None = None, mode: str | None = None
    ) -> BrainObservation:
        if mode is not None:
            self.set_mode(mode)
        return super().reset(seed)

    def step(
        self,
        command: DescendingCommand | RawControlCommand,
    ) -> StepTransition:
        return super().step(command)
