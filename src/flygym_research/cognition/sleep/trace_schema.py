from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np

from ..interfaces import (
    AscendingSummary,
    BrainObservation,
    DescendingCommand,
    RawBodyFeedback,
    RawControlCommand,
    StepTransition,
    WorldState,
)
from ..metrics import summarize_metrics


ArrayLikeKeys = {
    "target_vector",
    "target_xy_mm",
    "thorax_xy_mm",
    "avatar_xy",
    "partial_observation",
    "joint_angles",
    "joint_velocities",
    "body_positions",
    "body_rotations",
    "contact_active",
    "contact_forces",
    "contact_torques",
    "contact_positions",
    "contact_normals",
    "contact_tangents",
    "actuator_forces",
    "actuator_inputs",
    "adhesion_states",
}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(v) for v in value]
    return value



def _maybe_array(key: str, value: Any) -> Any:
    if key in ArrayLikeKeys and isinstance(value, list):
        return np.array(value, dtype=np.float64)
    return value



def serialize_transition(transition: StepTransition) -> dict[str, Any]:
    action_type = "descending" if isinstance(transition.action, DescendingCommand) else "raw"
    return {
        "reward": float(transition.reward),
        "terminated": bool(transition.terminated),
        "truncated": bool(transition.truncated),
        "info": _normalize_value(transition.info),
        "observation": {
            "raw_body": {
                "time": float(transition.observation.raw_body.time),
                "joint_angles": _normalize_value(transition.observation.raw_body.joint_angles),
                "joint_velocities": _normalize_value(
                    transition.observation.raw_body.joint_velocities
                ),
                "body_positions": _normalize_value(
                    transition.observation.raw_body.body_positions
                ),
                "body_rotations": _normalize_value(
                    transition.observation.raw_body.body_rotations
                ),
                "contact_active": _normalize_value(
                    transition.observation.raw_body.contact_active
                ),
                "contact_forces": _normalize_value(
                    transition.observation.raw_body.contact_forces
                ),
                "contact_torques": _normalize_value(
                    transition.observation.raw_body.contact_torques
                ),
                "contact_positions": _normalize_value(
                    transition.observation.raw_body.contact_positions
                ),
                "contact_normals": _normalize_value(
                    transition.observation.raw_body.contact_normals
                ),
                "contact_tangents": _normalize_value(
                    transition.observation.raw_body.contact_tangents
                ),
                "actuator_forces": _normalize_value(
                    transition.observation.raw_body.actuator_forces
                ),
            },
            "summary": {
                "features": _normalize_value(transition.observation.summary.features),
                "active_channels": list(transition.observation.summary.active_channels),
                "disabled_channels": list(transition.observation.summary.disabled_channels),
            },
            "world": {
                "mode": transition.observation.world.mode,
                "step_count": int(transition.observation.world.step_count),
                "reward": float(transition.observation.world.reward),
                "terminated": bool(transition.observation.world.terminated),
                "truncated": bool(transition.observation.world.truncated),
                "observables": _normalize_value(transition.observation.world.observables),
                "info": _normalize_value(transition.observation.world.info),
            },
            "history": [_normalize_value(entry) for entry in transition.observation.history],
        },
        "action": {
            "type": action_type,
            "payload": _normalize_value(transition.action.__dict__),
        },
    }



def deserialize_transition(payload: dict[str, Any]) -> StepTransition:
    raw_payload = payload["observation"]["raw_body"]
    raw_body = RawBodyFeedback(
        time=float(raw_payload["time"]),
        joint_angles=_maybe_array("joint_angles", raw_payload["joint_angles"]),
        joint_velocities=_maybe_array(
            "joint_velocities", raw_payload["joint_velocities"]
        ),
        body_positions=_maybe_array("body_positions", raw_payload["body_positions"]),
        body_rotations=_maybe_array("body_rotations", raw_payload["body_rotations"]),
        contact_active=_maybe_array("contact_active", raw_payload["contact_active"]),
        contact_forces=_maybe_array("contact_forces", raw_payload["contact_forces"]),
        contact_torques=_maybe_array("contact_torques", raw_payload["contact_torques"]),
        contact_positions=_maybe_array(
            "contact_positions", raw_payload["contact_positions"]
        ),
        contact_normals=_maybe_array("contact_normals", raw_payload["contact_normals"]),
        contact_tangents=_maybe_array(
            "contact_tangents", raw_payload["contact_tangents"]
        ),
        actuator_forces=_maybe_array(
            "actuator_forces", raw_payload["actuator_forces"]
        ),
    )
    summary_payload = payload["observation"]["summary"]
    summary = AscendingSummary(
        features={str(k): float(v) for k, v in summary_payload["features"].items()},
        active_channels=tuple(summary_payload["active_channels"]),
        disabled_channels=tuple(summary_payload["disabled_channels"]),
    )
    world_payload = payload["observation"]["world"]
    world = WorldState(
        mode=world_payload["mode"],
        step_count=int(world_payload["step_count"]),
        reward=float(world_payload["reward"]),
        terminated=bool(world_payload["terminated"]),
        truncated=bool(world_payload["truncated"]),
        observables={
            str(k): _maybe_array(str(k), v)
            for k, v in world_payload["observables"].items()
        },
        info=world_payload.get("info", {}),
    )
    observation = BrainObservation(
        raw_body=raw_body,
        summary=summary,
        world=world,
        history=tuple(payload["observation"]["history"]),
    )
    action_payload = payload["action"]
    if action_payload["type"] == "descending":
        action = DescendingCommand(**action_payload["payload"])
    else:
        raw_action = action_payload["payload"]
        action = RawControlCommand(
            actuator_inputs=_maybe_array("actuator_inputs", raw_action["actuator_inputs"]),
            adhesion_states=_maybe_array(
                "adhesion_states", raw_action.get("adhesion_states")
            )
            if raw_action.get("adhesion_states") is not None
            else None,
            metadata=raw_action.get("metadata", {}),
        )
    return StepTransition(
        observation=observation,
        action=action,
        reward=float(payload["reward"]),
        terminated=bool(payload["terminated"]),
        truncated=bool(payload["truncated"]),
        info=payload.get("info", {}),
    )



def _episode_id(
    controller_name: str,
    world_mode: str,
    seed: int,
    ablation_channels: tuple[str, ...],
    perturbation_tag: str,
) -> str:
    raw = "|".join(
        [controller_name, world_mode, str(seed), ",".join(ablation_channels), perturbation_tag]
    )
    digest = sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"ep-{digest}"


@dataclass(slots=True)
class TraceEpisode:
    controller_name: str
    world_mode: str
    seed: int
    transitions: list[StepTransition]
    controller_state: dict[str, float] = field(default_factory=dict)
    body_state: dict[str, float] = field(default_factory=dict)
    ablation_channels: tuple[str, ...] = ()
    perturbation_tag: str = "baseline"
    metadata: dict[str, Any] = field(default_factory=dict)
    episode_id: str = ""

    def __post_init__(self) -> None:
        if not self.episode_id:
            self.episode_id = _episode_id(
                self.controller_name,
                self.world_mode,
                self.seed,
                self.ablation_channels,
                self.perturbation_tag,
            )

    @property
    def episode_steps(self) -> int:
        return len(self.transitions)

    @property
    def summary_metrics(self) -> dict[str, float]:
        return summarize_metrics(self.transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "controller_name": self.controller_name,
            "world_mode": self.world_mode,
            "seed": self.seed,
            "ablation_channels": list(self.ablation_channels),
            "perturbation_tag": self.perturbation_tag,
            "episode_steps": self.episode_steps,
            "controller_state": _normalize_value(self.controller_state),
            "body_state": _normalize_value(self.body_state),
            "metadata": _normalize_value(self.metadata),
            "summary_metrics": _normalize_value(self.summary_metrics),
            "transitions": [serialize_transition(t) for t in self.transitions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TraceEpisode":
        return cls(
            episode_id=payload["episode_id"],
            controller_name=payload["controller_name"],
            world_mode=payload["world_mode"],
            seed=int(payload["seed"]),
            ablation_channels=tuple(payload.get("ablation_channels", [])),
            perturbation_tag=payload.get("perturbation_tag", "baseline"),
            controller_state={
                str(k): float(v) for k, v in payload.get("controller_state", {}).items()
            },
            body_state={str(k): float(v) for k, v in payload.get("body_state", {}).items()},
            metadata=payload.get("metadata", {}),
            transitions=[deserialize_transition(t) for t in payload["transitions"]],
        )


@dataclass(slots=True)
class TraceSegment:
    source_episode_id: str
    start_step: int
    end_step: int
    transitions: list[StepTransition]
    controller_name: str
    world_mode: str
    seed: int
    perturbation_tag: str = "baseline"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def segment_id(self) -> str:
        return f"{self.source_episode_id}:{self.start_step}-{self.end_step}"

    @property
    def summary_metrics(self) -> dict[str, float]:
        return summarize_metrics(self.transitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source_episode_id": self.source_episode_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "controller_name": self.controller_name,
            "world_mode": self.world_mode,
            "seed": self.seed,
            "perturbation_tag": self.perturbation_tag,
            "metadata": _normalize_value(self.metadata),
            "summary_metrics": _normalize_value(self.summary_metrics),
            "transitions": [serialize_transition(t) for t in self.transitions],
        }


@dataclass(slots=True)
class SleepCandidate:
    candidate_id: str
    representative_episode_id: str
    member_episode_ids: list[str]
    evidence: dict[str, Any]
    score_components: dict[str, float]
    residual_episode_ids: list[str] = field(default_factory=list)
    decision: str = "review"
    retained_exception_rationale: dict[str, str] = field(default_factory=dict)

    @property
    def compressed_episode_ids(self) -> list[str]:
        residuals = set(self.residual_episode_ids)
        return [episode_id for episode_id in self.member_episode_ids if episode_id not in residuals]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "representative_episode_id": self.representative_episode_id,
            "member_episode_ids": self.member_episode_ids,
            "evidence": _normalize_value(self.evidence),
            "score_components": _normalize_value(self.score_components),
            "residual_episode_ids": self.residual_episode_ids,
            "decision": self.decision,
            "retained_exception_rationale": self.retained_exception_rationale,
        }


@dataclass(slots=True)
class SleepArtifact:
    artifact_id: str
    trace_bank_path: str | None
    candidates: list[SleepCandidate]
    compressed_episode_ids: list[str]
    residual_episode_ids: list[str]
    validation: dict[str, Any]
    reports: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "trace_bank_path": self.trace_bank_path,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "compressed_episode_ids": self.compressed_episode_ids,
            "residual_episode_ids": self.residual_episode_ids,
            "validation": _normalize_value(self.validation),
            "reports": _normalize_value(self.reports),
            "metadata": _normalize_value(self.metadata),
        }



def episode_bank_fingerprint(episodes: list[TraceEpisode]) -> str:
    manifest = "|".join(sorted(ep.episode_id for ep in episodes))
    return sha1(manifest.encode("utf-8")).hexdigest()[:12]



def default_trace_bank_path(root: str | Path, episodes: list[TraceEpisode]) -> Path:
    root = Path(root)
    return root / f"trace_bank_{episode_bank_fingerprint(episodes)}.json"
