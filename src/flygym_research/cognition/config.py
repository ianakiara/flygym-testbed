from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BodyLayerConfig:
    timestep: float = 1e-4
    warmup_duration_s: float = 0.01
    phase_increment: float = 0.35
    gait_amplitude: float = 0.2
    turn_gain: float = 0.15
    stabilization_gain: float = 0.1
    adhesion_on: float = 1.0
    adhesion_off: float = 0.0
    kp: float = 45.0
    force_range: tuple[float, float] = (-30.0, 30.0)
    disabled_feedback_channels: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class EnvConfig:
    episode_steps: int = 64
    history_length: int = 8
    success_radius_mm: float = 1.0
    avatar_step_scale: float = 0.35
    avatar_turn_scale: float = 0.4
    avatar_noise_scale: float = 0.02
