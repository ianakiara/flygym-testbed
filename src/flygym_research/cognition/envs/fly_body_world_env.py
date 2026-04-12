from __future__ import annotations

from collections import deque

from ..config import EnvConfig
from ..interfaces import (
    BodyInterface,
    BrainObservation,
    DescendingCommand,
    RawControlCommand,
    StepTransition,
    WorldInterface,
)


class FlyBodyWorldEnv:
    def __init__(
        self,
        body: BodyInterface,
        world: WorldInterface,
        *,
        config: EnvConfig | None = None,
    ) -> None:
        self.body = body
        self.world = world
        self.config = config or EnvConfig()
        self._history: deque[dict[str, float]] = deque(
            maxlen=self.config.history_length
        )
        self._last_observation: BrainObservation | None = None

    def reset(self, seed: int | None = None) -> BrainObservation:
        raw_feedback, summary = self.body.reset(seed)
        world_state = self.world.reset(seed, raw_feedback, summary)
        self._history.clear()
        self._last_observation = BrainObservation(
            raw_body=raw_feedback,
            summary=summary,
            world=world_state,
            history=tuple(),
        )
        return self._last_observation

    def step(
        self,
        command: DescendingCommand | RawControlCommand,
    ) -> StepTransition:
        if self._last_observation is None:
            raise RuntimeError("Call reset() before step().")
        if isinstance(command, RawControlCommand):
            raw_feedback, summary, body_log = self.body.step_raw(
                command, self._last_observation.world
            )
            world_state = self.world.step(DescendingCommand(), raw_feedback, summary)
        else:
            raw_feedback, summary, body_log = self.body.step(
                command, self._last_observation.world
            )
            world_state = self.world.step(command, raw_feedback, summary)
        self._history.append(
            {
                "reward": float(world_state.reward),
                "stability": float(summary.features.get("stability", 0.0)),
                "mode": float(world_state.step_count),
            }
        )
        observation = BrainObservation(
            raw_body=raw_feedback,
            summary=summary,
            world=world_state,
            history=tuple(self._history),
        )
        info = dict(world_state.info)
        info["body_log"] = body_log.log if hasattr(body_log, "log") else body_log
        self._last_observation = observation
        return StepTransition(
            observation=observation,
            action=command,
            reward=float(world_state.reward),
            terminated=bool(world_state.terminated),
            truncated=bool(world_state.truncated),
            info=info,
        )

    def action_spec(self) -> dict[str, tuple[float, float]]:
        return self.body.get_action_spec()
