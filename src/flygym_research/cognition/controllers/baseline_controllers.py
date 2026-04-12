from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import (
    BrainInterface,
    BrainObservation,
    DescendingCommand,
    RawControlCommand,
)


@dataclass(slots=True)
class ReflexOnlyController(BrainInterface):
    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, observation: BrainObservation) -> DescendingCommand:
        del observation
        return DescendingCommand(stabilization_priority=1.0)

    def get_internal_state(self) -> dict[str, float]:
        return {"memory": 0.0}


@dataclass(slots=True)
class RandomController(BrainInterface):
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: BrainObservation) -> DescendingCommand:
        del observation
        return DescendingCommand(
            move_intent=float(self.rng.uniform(-1.0, 1.0)),
            turn_intent=float(self.rng.uniform(-1.0, 1.0)),
            speed_modulation=float(self.rng.uniform(-1.0, 1.0)),
            stabilization_priority=float(self.rng.uniform(0.0, 1.0)),
        )

    def get_internal_state(self) -> dict[str, float]:
        return {"entropy": 1.0}


@dataclass(slots=True)
class ReducedDescendingController(BrainInterface):
    memory_trace: float = 0.0

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.memory_trace = 0.0

    def act(self, observation: BrainObservation) -> DescendingCommand:
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        stability = observation.summary.features.get("stability", 0.0)
        self.memory_trace = 0.8 * self.memory_trace + 0.2 * float(
            np.linalg.norm(target_vector)
        )
        move_intent = float(np.clip(target_vector[0], -1.0, 1.0))
        turn_intent = float(np.clip(target_vector[1], -1.0, 1.0))
        return DescendingCommand(
            move_intent=move_intent,
            turn_intent=turn_intent,
            speed_modulation=float(np.clip(self.memory_trace, -1.0, 1.0)),
            stabilization_priority=float(np.clip(stability + 0.25, 0.0, 1.0)),
            target_bias=(float(target_vector[0]), float(target_vector[1])),
        )

    def get_internal_state(self) -> dict[str, float]:
        return {"memory_trace": float(self.memory_trace)}


@dataclass(slots=True)
class NoAscendingFeedbackController(BrainInterface):
    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, observation: BrainObservation) -> DescendingCommand:
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        return DescendingCommand(
            move_intent=float(np.clip(target_vector[0], -1.0, 1.0)),
            turn_intent=float(np.clip(target_vector[1], -1.0, 1.0)),
            stabilization_priority=0.0,
            target_bias=(float(target_vector[0]), float(target_vector[1])),
        )

    def get_internal_state(self) -> dict[str, float]:
        return {"memory": 0.0}


@dataclass(slots=True)
class RawControlController(BrainInterface):
    amplitude: float = 0.15

    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, observation: BrainObservation) -> RawControlCommand:
        n = observation.raw_body.joint_angles.shape[0]
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        command = np.zeros(n, dtype=np.float64)
        if n > 0:
            command += self.amplitude * np.clip(target_vector[0], -1.0, 1.0)
        adhesion = np.ones(6, dtype=np.float64)
        return RawControlCommand(actuator_inputs=command, adhesion_states=adhesion)

    def get_internal_state(self) -> dict[str, float]:
        return {"amplitude": float(self.amplitude)}


@dataclass(slots=True)
class BodylessAvatarController(ReducedDescendingController):
    pass
