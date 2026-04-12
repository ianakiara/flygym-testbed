from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import BodyLayerConfig
from ..interfaces import AscendingSummary, RawBodyFeedback

CHANNEL_GROUPS = {
    "pose": {"stability", "thorax_height_mm", "body_speed_mm_s"},
    "contact": {"contact_fraction", "slip_risk", "collision_load"},
    "locomotion": {"locomotion_quality", "actuator_effort"},
    "target": {"target_distance", "target_salience"},
    "internal": {"phase", "phase_velocity"},
}


@dataclass(slots=True)
class AscendingAdapter:
    thorax_index: int
    config: BodyLayerConfig
    _prev_thorax_position: np.ndarray | None = field(default=None, init=False)

    def reset(self) -> None:
        self._prev_thorax_position = None

    def summarize(
        self,
        raw_feedback: RawBodyFeedback,
        *,
        internal_state: dict[str, float] | None = None,
        target_vector: np.ndarray | None = None,
    ) -> AscendingSummary:
        thorax_position = raw_feedback.body_positions[self.thorax_index]
        thorax_quat = raw_feedback.body_rotations[self.thorax_index]
        if self._prev_thorax_position is None:
            speed = 0.0
        else:
            dt = max(raw_feedback.time, 1e-6)
            speed = float(
                np.linalg.norm(thorax_position - self._prev_thorax_position) / dt
            )
        self._prev_thorax_position = thorax_position.copy()

        orientation_error = float(np.linalg.norm(thorax_quat[1:3]))
        contact_fraction = float(np.mean(raw_feedback.contact_active > 0.5))
        tangential_force = np.linalg.norm(raw_feedback.contact_forces[:, :2], axis=1)
        normal_force = np.abs(raw_feedback.contact_forces[:, 2]) + 1e-6
        slip_risk = float(np.mean(tangential_force / normal_force))
        collision_load = float(
            np.mean(np.linalg.norm(raw_feedback.contact_forces, axis=1))
        )
        actuator_effort = float(np.mean(np.abs(raw_feedback.actuator_forces)))
        stability = float(
            1.0
            / (
                1.0
                + orientation_error
                + max(0.0, 1.2 - float(thorax_position[2]))
                + slip_risk
            )
        )
        locomotion_quality = float(speed * (0.5 + contact_fraction) / (1.0 + slip_risk))

        features = {
            "stability": stability,
            "thorax_height_mm": float(thorax_position[2]),
            "body_speed_mm_s": speed,
            "contact_fraction": contact_fraction,
            "slip_risk": slip_risk,
            "collision_load": collision_load,
            "locomotion_quality": locomotion_quality,
            "actuator_effort": actuator_effort,
            "target_distance": (
                float(np.linalg.norm(target_vector))
                if target_vector is not None
                else np.nan
            ),
            "target_salience": (
                float(1.0 / (1.0 + np.linalg.norm(target_vector)))
                if target_vector is not None
                else 0.0
            ),
            "phase": float((internal_state or {}).get("phase", 0.0)),
            "phase_velocity": float((internal_state or {}).get("phase_velocity", 0.0)),
        }

        disabled = set(self.config.disabled_feedback_channels)
        if disabled:
            features = {
                key: value
                for key, value in features.items()
                if not any(key in CHANNEL_GROUPS[group] for group in disabled)
            }
        active_channels = tuple(sorted(features.keys()))
        return AscendingSummary(
            features=features,
            active_channels=active_channels,
            disabled_channels=tuple(sorted(disabled)),
        )
