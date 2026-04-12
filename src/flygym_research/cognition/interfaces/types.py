from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class DescendingCommand:
    move_intent: float = 0.0
    turn_intent: float = 0.0
    speed_modulation: float = 0.0
    stabilization_priority: float = 1.0
    approach_avoid: float = 0.0
    interaction_trigger: float = 0.0
    target_bias: tuple[float, float] = (0.0, 0.0)
    state_mode: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RawControlCommand:
    actuator_inputs: np.ndarray
    adhesion_states: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdapterOutput:
    actuator_inputs: np.ndarray
    adhesion_states: np.ndarray
    log: dict[str, Any]


@dataclass(slots=True)
class RawBodyFeedback:
    time: float
    joint_angles: np.ndarray
    joint_velocities: np.ndarray
    body_positions: np.ndarray
    body_rotations: np.ndarray
    contact_active: np.ndarray
    contact_forces: np.ndarray
    contact_torques: np.ndarray
    contact_positions: np.ndarray
    contact_normals: np.ndarray
    contact_tangents: np.ndarray
    actuator_forces: np.ndarray


@dataclass(slots=True)
class AscendingSummary:
    features: dict[str, float]
    active_channels: tuple[str, ...]
    disabled_channels: tuple[str, ...]


@dataclass(slots=True)
class WorldState:
    mode: str
    step_count: int
    reward: float
    terminated: bool
    truncated: bool
    observables: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrainObservation:
    raw_body: RawBodyFeedback
    summary: AscendingSummary
    world: WorldState
    history: tuple[dict[str, float], ...]


@dataclass(slots=True)
class StepTransition:
    observation: BrainObservation
    action: DescendingCommand | RawControlCommand
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
