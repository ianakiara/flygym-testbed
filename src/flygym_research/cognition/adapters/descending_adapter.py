from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flygym.anatomy import JointDOF

from ..config import BodyLayerConfig
from ..interfaces import AdapterOutput, DescendingCommand, RawBodyFeedback

# Alternating tripod groups using FlyGym leg identifiers:
# left-front/right-middle/left-hind and right-front/left-middle/right-hind.
# This preserves the standard insect alternating-tripod gait pattern in a compact form.
TRIPOD_A = {"lf", "rm", "lh"}
TRIPOD_B = {"rf", "lm", "rh"}


@dataclass(slots=True)
class DescendingAdapter:
    actuator_order: list[JointDOF]
    thorax_index: int
    neutral_inputs: np.ndarray
    config: BodyLayerConfig

    def action_spec(self) -> dict[str, tuple[float, float]]:
        return {
            "move_intent": (-1.0, 1.0),
            "turn_intent": (-1.0, 1.0),
            "speed_modulation": (-1.0, 1.0),
            "stabilization_priority": (0.0, 1.0),
            "approach_avoid": (-1.0, 1.0),
            "interaction_trigger": (0.0, 1.0),
        }

    def map_command(
        self,
        command: DescendingCommand,
        raw_feedback: RawBodyFeedback,
        phase: float,
    ) -> AdapterOutput:
        move_intent = float(np.clip(command.move_intent, -1.0, 1.0))
        turn_intent = float(np.clip(command.turn_intent, -1.0, 1.0))
        speed_modulation = float(np.clip(command.speed_modulation, -1.0, 1.0))
        stabilization_priority = float(
            np.clip(command.stabilization_priority, 0.0, 1.0)
        )
        target_delta = np.array(command.target_bias, dtype=np.float64)
        targets = self.neutral_inputs.copy()
        adhesion_states = np.full(6, self.config.adhesion_on, dtype=np.float64)

        thorax_quat = raw_feedback.body_rotations[self.thorax_index]
        roll_proxy = float(thorax_quat[1])
        pitch_proxy = float(thorax_quat[2])
        gait_amplitude = self.config.gait_amplitude * (0.25 + 0.75 * abs(move_intent))
        gait_amplitude *= 1.0 + 0.5 * max(speed_modulation, 0.0)

        if abs(move_intent) < 1e-3 and abs(turn_intent) < 1e-3:
            return AdapterOutput(
                actuator_inputs=targets,
                adhesion_states=adhesion_states,
                log={
                    "phase": phase,
                    "move_intent": move_intent,
                    "turn_intent": turn_intent,
                    "mode": command.state_mode,
                    "stance_legs": ("lf", "lm", "lh", "rf", "rm", "rh"),
                },
            )

        stance_legs: list[str] = []
        for idx, jointdof in enumerate(self.actuator_order):
            leg = jointdof.child.pos
            phase_offset = 0.0 if leg in TRIPOD_A else np.pi
            gait_phase = np.sin(phase + phase_offset)
            if gait_phase <= 0:
                stance_legs.append(leg)
            else:
                adhesion_states[self._leg_index(leg)] = self.config.adhesion_off

            side_sign = 1.0 if leg.startswith("l") else -1.0
            swing_term = gait_phase * gait_amplitude * np.sign(move_intent or 1.0)
            turn_term = side_sign * turn_intent * self.config.turn_gain
            target_term = float(target_delta[1]) * 0.05 * side_sign
            stabilization_roll = (
                -roll_proxy * stabilization_priority * self.config.stabilization_gain
            )
            stabilization_pitch = (
                -pitch_proxy * stabilization_priority * self.config.stabilization_gain
            )

            if jointdof.axis.value == "pitch":
                if jointdof.child.link == "coxa":
                    targets[idx] += 0.12 * swing_term + stabilization_pitch
                elif jointdof.child.link == "trochanterfemur":
                    targets[idx] += swing_term
                elif jointdof.child.link == "tibia":
                    targets[idx] -= 0.7 * swing_term
                else:
                    targets[idx] -= 0.25 * swing_term
            elif jointdof.axis.value == "yaw":
                targets[idx] += turn_term + target_term
            elif jointdof.axis.value == "roll":
                targets[idx] += turn_term + stabilization_roll * side_sign

        return AdapterOutput(
            actuator_inputs=targets,
            adhesion_states=adhesion_states,
            log={
                "phase": phase,
                "move_intent": move_intent,
                "turn_intent": turn_intent,
                "speed_modulation": speed_modulation,
                "stabilization_priority": stabilization_priority,
                "mean_target_delta": float(
                    np.mean(np.abs(targets - self.neutral_inputs))
                ),
                "stance_legs": tuple(sorted(set(stance_legs))),
            },
        )

    @staticmethod
    def _leg_index(leg: str) -> int:
        return ("lf", "lm", "lh", "rf", "rm", "rh").index(leg)
