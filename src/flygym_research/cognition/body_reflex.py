from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import ActuatorType, FlatGroundWorld, Fly, KinematicPosePreset
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D

from .adapters import AscendingAdapter, DescendingAdapter
from .config import BodyLayerConfig
from .interfaces import (
    AdapterOutput,
    AscendingSummary,
    BodyInterface,
    DescendingCommand,
    RawBodyFeedback,
    RawControlCommand,
    WorldState,
)


@dataclass(slots=True)
class FlyBodyLayer(BodyInterface):
    config: BodyLayerConfig = field(default_factory=BodyLayerConfig)

    def __post_init__(self) -> None:
        neutral_pose = KinematicPosePreset.NEUTRAL.get_pose_by_axis_order(
            AxisOrder.YAW_PITCH_ROLL
        )
        skeleton = Skeleton(
            axis_order=AxisOrder.YAW_PITCH_ROLL,
            joint_preset=JointPreset.LEGS_ONLY,
        )
        fly = Fly(name="cognition_fly")
        fly.add_joints(skeleton, neutral_pose=neutral_pose)
        actuated_dofs = skeleton.get_actuated_dofs_from_preset(
            ActuatedDOFPreset.LEGS_ACTIVE_ONLY
        )
        fly.add_actuators(
            actuated_dofs,
            ActuatorType.POSITION,
            neutral_input=neutral_pose,
            kp=self.config.kp,
            forcerange=self.config.force_range,
        )
        fly.add_leg_adhesion(gain=1.0)
        world = FlatGroundWorld(name="cognition_world")
        world.add_fly(
            fly,
            spawn_position=[0.0, 0.0, 1.5],
            spawn_rotation=Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        )
        self.fly = fly
        self.world = world
        self.sim = Simulation(world)
        self.phase = 0.0
        self.phase_velocity = self.config.phase_increment
        self.actuator_order = fly.get_actuated_jointdofs_order(ActuatorType.POSITION)
        neutral_lookup = fly.jointdof_to_neutralaction_by_type[ActuatorType.POSITION]
        self.neutral_inputs = np.array(
            [neutral_lookup[jointdof] for jointdof in self.actuator_order],
            dtype=np.float64,
        )
        thorax_index = next(
            i
            for i, seg in enumerate(fly.get_bodysegs_order())
            if seg.name == "c_thorax"
        )
        self.descending_adapter = DescendingAdapter(
            actuator_order=list(self.actuator_order),
            thorax_index=thorax_index,
            neutral_inputs=self.neutral_inputs,
            config=self.config,
        )
        self.ascending_adapter = AscendingAdapter(
            thorax_index=thorax_index, config=self.config
        )

    def reset(
        self, seed: int | None = None
    ) -> tuple[RawBodyFeedback, AscendingSummary]:
        del seed
        self.sim.reset()
        self.sim.set_leg_adhesion_states(
            self.fly.name, np.full(6, self.config.adhesion_on, dtype=np.float64)
        )
        self.sim.warmup(self.config.warmup_duration_s)
        self.phase = 0.0
        self.phase_velocity = self.config.phase_increment
        self.ascending_adapter.reset()
        raw = self._collect_raw_feedback()
        summary = self.ascending_adapter.summarize(
            raw, internal_state=self.get_internal_state()
        )
        return raw, summary

    def step(
        self,
        command: DescendingCommand,
        world_state: WorldState | None = None,
    ) -> tuple[RawBodyFeedback, AscendingSummary, AdapterOutput]:
        raw_before = self._collect_raw_feedback()
        target_vector = self._target_vector(world_state)
        adapter_output = self.descending_adapter.map_command(
            command, raw_before, self.phase
        )
        self.sim.set_actuator_inputs(
            self.fly.name, ActuatorType.POSITION, adapter_output.actuator_inputs
        )
        self.sim.set_leg_adhesion_states(self.fly.name, adapter_output.adhesion_states)
        self.sim.step()
        self.phase += self.phase_velocity * (
            1.0 + 0.5 * max(command.speed_modulation, -0.5)
        )
        raw_after = self._collect_raw_feedback()
        summary = self.ascending_adapter.summarize(
            raw_after,
            internal_state=self.get_internal_state(),
            target_vector=target_vector,
        )
        return raw_after, summary, adapter_output

    def step_raw(
        self,
        command: RawControlCommand,
        world_state: WorldState | None = None,
    ) -> tuple[RawBodyFeedback, AscendingSummary, dict[str, Any]]:
        del world_state
        self.sim.set_actuator_inputs(
            self.fly.name, ActuatorType.POSITION, command.actuator_inputs
        )
        if command.adhesion_states is not None:
            self.sim.set_leg_adhesion_states(self.fly.name, command.adhesion_states)
        self.sim.step()
        raw_after = self._collect_raw_feedback()
        summary = self.ascending_adapter.summarize(
            raw_after,
            internal_state=self.get_internal_state(),
        )
        log = {
            "mode": "raw_control",
            "mean_abs_input": float(np.mean(np.abs(command.actuator_inputs))),
        }
        return raw_after, summary, log

    def get_action_spec(self) -> dict[str, tuple[float, float]]:
        return self.descending_adapter.action_spec()

    def get_internal_state(self) -> dict[str, float]:
        return {
            "phase": float(self.phase),
            "phase_velocity": float(self.phase_velocity),
        }

    def _collect_raw_feedback(self) -> RawBodyFeedback:
        contact_active, forces, torques, positions, normals, tangents = (
            self.sim.get_ground_contact_info(self.fly.name)
        )
        return RawBodyFeedback(
            time=self.sim.time,
            joint_angles=self.sim.get_joint_angles(self.fly.name),
            joint_velocities=self.sim.get_joint_velocities(self.fly.name),
            body_positions=self.sim.get_body_positions(self.fly.name),
            body_rotations=self.sim.get_body_rotations(self.fly.name),
            contact_active=contact_active,
            contact_forces=forces,
            contact_torques=torques,
            contact_positions=positions,
            contact_normals=normals,
            contact_tangents=tangents,
            actuator_forces=self.sim.get_actuator_forces(
                self.fly.name, ActuatorType.POSITION
            ),
        )

    @staticmethod
    def _target_vector(world_state: WorldState | None) -> np.ndarray | None:
        if world_state is None:
            return None
        target_vector = world_state.observables.get("target_vector")
        if target_vector is None:
            return None
        return np.asarray(target_vector, dtype=np.float64)


@dataclass(slots=True)
class BodylessBodyLayer(BodyInterface):
    config: BodyLayerConfig = field(default_factory=BodyLayerConfig)

    def __post_init__(self) -> None:
        self.time = 0.0
        self.phase = 0.0
        self._position = np.zeros(3, dtype=np.float64)

    def reset(
        self, seed: int | None = None
    ) -> tuple[RawBodyFeedback, AscendingSummary]:
        del seed
        self.time = 0.0
        self.phase = 0.0
        self._position = np.zeros(3, dtype=np.float64)
        raw = self._make_feedback(np.zeros(3, dtype=np.float64))
        summary = AscendingSummary(
            features={
                "stability": 1.0,
                "thorax_height_mm": 0.0,
                "body_speed_mm_s": 0.0,
                "contact_fraction": 0.0,
                "slip_risk": 0.0,
                "locomotion_quality": 0.0,
                "phase": 0.0,
                "phase_velocity": self.config.phase_increment,
            },
            active_channels=(
                "body_speed_mm_s",
                "contact_fraction",
                "locomotion_quality",
                "phase",
                "phase_velocity",
                "slip_risk",
                "stability",
                "thorax_height_mm",
            ),
            disabled_channels=tuple(),
        )
        return raw, summary

    def step(
        self,
        command: DescendingCommand,
        world_state: WorldState | None = None,
    ) -> tuple[RawBodyFeedback, AscendingSummary, AdapterOutput]:
        del world_state
        delta = (
            np.array([command.move_intent, command.turn_intent, 0.0], dtype=np.float64)
            * self.config.bodyless_position_scale
        )
        self._position += delta
        self.time += self.config.phase_increment
        self.phase += self.config.phase_increment
        raw = self._make_feedback(delta)
        summary = AscendingSummary(
            features={
                "stability": 1.0,
                "thorax_height_mm": 0.0,
                "body_speed_mm_s": float(np.linalg.norm(delta)),
                "contact_fraction": 0.0,
                "slip_risk": 0.0,
                "locomotion_quality": float(np.linalg.norm(delta)),
                "phase": float(self.phase),
                "phase_velocity": self.config.phase_increment,
            },
            active_channels=(
                "body_speed_mm_s",
                "contact_fraction",
                "locomotion_quality",
                "phase",
                "phase_velocity",
                "slip_risk",
                "stability",
                "thorax_height_mm",
            ),
            disabled_channels=tuple(),
        )
        return (
            raw,
            summary,
            AdapterOutput(
                actuator_inputs=np.zeros(1, dtype=np.float64),
                adhesion_states=np.zeros(6, dtype=np.float64),
                log={"mode": "bodyless"},
            ),
        )

    def step_raw(
        self,
        command: RawControlCommand,
        world_state: WorldState | None = None,
    ) -> tuple[RawBodyFeedback, AscendingSummary, dict[str, Any]]:
        del world_state
        # Interpret the first two actuator values as positional deltas
        # to maintain parity with the descending interface.
        move = (
            float(command.actuator_inputs[0])
            if len(command.actuator_inputs) > 0
            else 0.0
        )
        turn = (
            float(command.actuator_inputs[1])
            if len(command.actuator_inputs) > 1
            else 0.0
        )
        delta = (
            np.array([move, turn, 0.0], dtype=np.float64)
            * self.config.bodyless_position_scale
        )
        self._position += delta
        self.time += self.config.phase_increment
        self.phase += self.config.phase_increment
        raw = self._make_feedback(delta)
        summary = AscendingSummary(
            features={
                "stability": 1.0,
                "thorax_height_mm": 0.0,
                "body_speed_mm_s": float(np.linalg.norm(delta)),
                "contact_fraction": 0.0,
                "slip_risk": 0.0,
                "locomotion_quality": float(np.linalg.norm(delta)),
                "phase": float(self.phase),
                "phase_velocity": self.config.phase_increment,
            },
            active_channels=(
                "body_speed_mm_s",
                "contact_fraction",
                "locomotion_quality",
                "phase",
                "phase_velocity",
                "slip_risk",
                "stability",
                "thorax_height_mm",
            ),
            disabled_channels=tuple(),
        )
        return raw, summary, {"mode": "bodyless_raw"}

    def get_action_spec(self) -> dict[str, tuple[float, float]]:
        return {
            "move_intent": (-1.0, 1.0),
            "turn_intent": (-1.0, 1.0),
            "speed_modulation": (-1.0, 1.0),
            "stabilization_priority": (0.0, 1.0),
            "approach_avoid": (-1.0, 1.0),
            "interaction_trigger": (0.0, 1.0),
        }

    def get_internal_state(self) -> dict[str, float]:
        return {
            "phase": float(self.phase),
            "phase_velocity": self.config.phase_increment,
        }

    def _make_feedback(self, delta: np.ndarray) -> RawBodyFeedback:
        return RawBodyFeedback(
            time=self.time,
            joint_angles=np.zeros(1, dtype=np.float64),
            joint_velocities=np.zeros(1, dtype=np.float64),
            body_positions=self._position.reshape(1, -1),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            contact_active=np.zeros(6, dtype=np.float64),
            contact_forces=np.zeros((6, 3), dtype=np.float64),
            contact_torques=np.zeros((6, 3), dtype=np.float64),
            contact_positions=np.zeros((6, 3), dtype=np.float64),
            contact_normals=np.zeros((6, 3), dtype=np.float64),
            contact_tangents=np.zeros((6, 3), dtype=np.float64),
            actuator_forces=np.array([float(np.linalg.norm(delta))], dtype=np.float64),
        )
