"""Experiment 6 — Observer Interoperability.

Validates:  translation >> raw similarity

Creates observer variants (noise, scaling, bias, partial observation)
and compares raw agreement vs translated agreement using MSE,
correlation, and AUC metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics import (
    interoperability_score,
    raw_latent_alignment,
    translated_latent_alignment,
)
from ..interfaces import AscendingSummary, BrainObservation, RawBodyFeedback, StepTransition, WorldState
from ..metrics.interoperability_metrics import extract_state_matrix
from .exp_sleep_trace_compressor import collect_trace_bank


def _clone_transition(
    transition,
    *,
    raw_body: RawBodyFeedback | None = None,
    features: dict[str, float] | None = None,
    observables: dict | None = None,
    info: dict | None = None,
) -> StepTransition:
    summary = AscendingSummary(
        features=features or dict(transition.observation.summary.features),
        active_channels=transition.observation.summary.active_channels,
        disabled_channels=transition.observation.summary.disabled_channels,
    )
    world = WorldState(
        mode=transition.observation.world.mode,
        step_count=transition.observation.world.step_count,
        reward=transition.observation.world.reward,
        terminated=transition.observation.world.terminated,
        truncated=transition.observation.world.truncated,
        observables=observables or dict(transition.observation.world.observables),
        info=info or dict(transition.observation.world.info),
    )
    observation = BrainObservation(
        raw_body=raw_body or transition.observation.raw_body,
        summary=summary,
        world=world,
        history=transition.observation.history,
    )
    return StepTransition(
        observation=observation,
        action=transition.action,
        reward=transition.reward,
        terminated=transition.terminated,
        truncated=transition.truncated,
        info=world.info,
    )


def _copy_world_observables(transition, *, avatar_xy=None, heading=None, target_vector=None) -> dict:
    observables = dict(transition.observation.world.observables)
    if avatar_xy is not None:
        observables["avatar_xy"] = np.asarray(avatar_xy, dtype=np.float64)
    if heading is not None:
        observables["heading"] = float(heading)
    if target_vector is not None:
        observables["target_vector"] = np.asarray(target_vector, dtype=np.float64)
    return observables


# ---------------------------------------------------------------------------
# Observer perturbation generators
# ---------------------------------------------------------------------------

def _perturb_transitions_noise(transitions: list, *, scale: float = 0.3, rng: np.random.Generator | None = None) -> list:
    """Add Gaussian noise to observations."""
    rng = rng or np.random.default_rng(0)
    perturbed = []
    for t in transitions:
        noisy_positions = t.observation.raw_body.body_positions + rng.normal(0, scale, t.observation.raw_body.body_positions.shape)
        noisy_feedback = RawBodyFeedback(
            time=t.observation.raw_body.time,
            joint_angles=t.observation.raw_body.joint_angles + rng.normal(0, scale * 0.5, t.observation.raw_body.joint_angles.shape),
            joint_velocities=t.observation.raw_body.joint_velocities,
            body_positions=noisy_positions.astype(np.float64),
            body_rotations=t.observation.raw_body.body_rotations,
            contact_active=t.observation.raw_body.contact_active,
            contact_forces=t.observation.raw_body.contact_forces,
            contact_torques=t.observation.raw_body.contact_torques,
            contact_positions=t.observation.raw_body.contact_positions,
            contact_normals=t.observation.raw_body.contact_normals,
            contact_tangents=t.observation.raw_body.contact_tangents,
            actuator_forces=t.observation.raw_body.actuator_forces,
        )
        features = dict(t.observation.summary.features)
        for key in (
            "body_speed_mm_s",
            "locomotion_quality",
            "actuator_effort",
            "phase",
            "phase_velocity",
        ):
            features[key] = float(features.get(key, 0.0) + rng.normal(0.0, scale))
        target_vec = np.asarray(
            t.observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        noisy_target = target_vec + rng.normal(0.0, scale, target_vec.shape)
        avatar_xy = np.asarray(
            t.observation.world.observables.get("avatar_xy", np.zeros(2)),
            dtype=np.float64,
        ) + rng.normal(0.0, scale, 2)
        info = dict(t.info)
        info["distance_to_target"] = float(info.get("distance_to_target", 0.0) + rng.normal(0.0, scale))
        perturbed.append(
            _clone_transition(
                t,
                raw_body=noisy_feedback,
                features=features,
                observables=_copy_world_observables(
                    t,
                    avatar_xy=avatar_xy,
                    heading=t.observation.world.observables.get("heading", 0.0) + rng.normal(0.0, scale * 0.25),
                    target_vector=noisy_target,
                ),
                info=info,
            )
        )
    return perturbed


def _perturb_transitions_scaling(transitions: list, *, factor: float = 2.0) -> list:
    """Scale observations by a constant factor."""
    perturbed = []
    for t in transitions:
        scaled_feedback = RawBodyFeedback(
            time=t.observation.raw_body.time,
            joint_angles=t.observation.raw_body.joint_angles * factor,
            joint_velocities=t.observation.raw_body.joint_velocities * factor,
            body_positions=t.observation.raw_body.body_positions * factor,
            body_rotations=t.observation.raw_body.body_rotations,
            contact_active=t.observation.raw_body.contact_active,
            contact_forces=t.observation.raw_body.contact_forces * factor,
            contact_torques=t.observation.raw_body.contact_torques,
            contact_positions=t.observation.raw_body.contact_positions,
            contact_normals=t.observation.raw_body.contact_normals,
            contact_tangents=t.observation.raw_body.contact_tangents,
            actuator_forces=t.observation.raw_body.actuator_forces,
        )
        features = {
            key: float(value * factor)
            for key, value in t.observation.summary.features.items()
        }
        target_vec = np.asarray(
            t.observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        ) * factor
        avatar_xy = np.asarray(
            t.observation.world.observables.get("avatar_xy", np.zeros(2)),
            dtype=np.float64,
        ) * factor
        info = dict(t.info)
        info["distance_to_target"] = float(info.get("distance_to_target", 0.0) * factor)
        perturbed.append(
            _clone_transition(
                t,
                raw_body=scaled_feedback,
                features=features,
                observables=_copy_world_observables(
                    t,
                    avatar_xy=avatar_xy,
                    target_vector=target_vec,
                ),
                info=info,
            )
        )
    return perturbed


def _perturb_transitions_bias(transitions: list, *, bias: float = 1.5) -> list:
    """Add constant bias to observations."""
    perturbed = []
    for t in transitions:
        biased_feedback = RawBodyFeedback(
            time=t.observation.raw_body.time,
            joint_angles=t.observation.raw_body.joint_angles + bias,
            joint_velocities=t.observation.raw_body.joint_velocities,
            body_positions=t.observation.raw_body.body_positions + bias,
            body_rotations=t.observation.raw_body.body_rotations,
            contact_active=t.observation.raw_body.contact_active,
            contact_forces=t.observation.raw_body.contact_forces,
            contact_torques=t.observation.raw_body.contact_torques,
            contact_positions=t.observation.raw_body.contact_positions,
            contact_normals=t.observation.raw_body.contact_normals,
            contact_tangents=t.observation.raw_body.contact_tangents,
            actuator_forces=t.observation.raw_body.actuator_forces,
        )
        features = {
            key: float(value + bias)
            for key, value in t.observation.summary.features.items()
        }
        target_vec = np.asarray(
            t.observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        ) + bias
        avatar_xy = np.asarray(
            t.observation.world.observables.get("avatar_xy", np.zeros(2)),
            dtype=np.float64,
        ) + bias
        info = dict(t.info)
        info["distance_to_target"] = float(info.get("distance_to_target", 0.0) + bias)
        perturbed.append(
            _clone_transition(
                t,
                raw_body=biased_feedback,
                features=features,
                observables=_copy_world_observables(
                    t,
                    avatar_xy=avatar_xy,
                    target_vector=target_vec,
                ),
                info=info,
            )
        )
    return perturbed


def _perturb_transitions_partial(transitions: list, *, drop_fraction: float = 0.3, rng: np.random.Generator | None = None) -> list:
    """Drop a fraction of observation channels (partial observability)."""
    rng = rng or np.random.default_rng(0)
    perturbed = []
    for t in transitions:
        mask = rng.random(t.observation.raw_body.joint_angles.shape) > drop_fraction
        partial_feedback = RawBodyFeedback(
            time=t.observation.raw_body.time,
            joint_angles=t.observation.raw_body.joint_angles * mask.astype(np.float64),
            joint_velocities=t.observation.raw_body.joint_velocities * mask.astype(np.float64),
            body_positions=t.observation.raw_body.body_positions,
            body_rotations=t.observation.raw_body.body_rotations,
            contact_active=t.observation.raw_body.contact_active,
            contact_forces=t.observation.raw_body.contact_forces,
            contact_torques=t.observation.raw_body.contact_torques,
            contact_positions=t.observation.raw_body.contact_positions,
            contact_normals=t.observation.raw_body.contact_normals,
            contact_tangents=t.observation.raw_body.contact_tangents,
            actuator_forces=t.observation.raw_body.actuator_forces,
        )
        features = dict(t.observation.summary.features)
        for key in (
            "body_speed_mm_s",
            "locomotion_quality",
            "actuator_effort",
            "phase",
            "phase_velocity",
        ):
            if rng.random() < drop_fraction:
                features[key] = 0.0
        target_vec = np.asarray(
            t.observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        keep_target = rng.random(target_vec.shape) > drop_fraction
        avatar_xy = np.asarray(
            t.observation.world.observables.get("avatar_xy", np.zeros(2)),
            dtype=np.float64,
        )
        keep_avatar = rng.random(avatar_xy.shape) > drop_fraction
        info = dict(t.info)
        if rng.random() < drop_fraction:
            info["distance_to_target"] = 0.0
        perturbed.append(
            _clone_transition(
                t,
                raw_body=partial_feedback,
                features=features,
                observables=_copy_world_observables(
                    t,
                    avatar_xy=avatar_xy * keep_avatar.astype(np.float64),
                    target_vector=target_vec * keep_target.astype(np.float64),
                ),
                info=info,
            )
        )
    return perturbed


PERTURBATION_TYPES = {
    "noise": _perturb_transitions_noise,
    "scaling": _perturb_transitions_scaling,
    "bias": _perturb_transitions_bias,
    "partial_observation": _perturb_transitions_partial,
}


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def _compute_raw_agreement(original: list, perturbed: list) -> dict[str, float]:
    """Compute raw (untranslated) agreement between original and perturbed.

    raw_latent_alignment() returns:
      - raw_alignment: mean per-dimension Pearson correlation
      - raw_dims_active: number of non-constant dimensions
    We also compute element-wise MSE on the state matrices directly.
    """
    try:
        raw = raw_latent_alignment(original, perturbed)
        # Compute actual MSE on state matrices
        n = min(len(original), len(perturbed))
        Z_orig = extract_state_matrix(original[:n])
        Z_pert = extract_state_matrix(perturbed[:n])
        raw_mse = float(np.mean((Z_orig - Z_pert) ** 2)) if n > 0 else 1.0
        return {
            "raw_mse": raw_mse,
            "raw_correlation": float(raw.get("raw_alignment", 0.0)),
        }
    except Exception:
        return {"raw_mse": 1.0, "raw_correlation": 0.0}


def _compute_translated_agreement(original: list, perturbed: list) -> dict[str, float]:
    """Compute translated agreement (after learned alignment).

    translated_latent_alignment() returns:
      - translated_alignment: max(R²_ab, R²_ba)
      - translation_r2_ab / translation_r2_ba: directional R² values
      - translation_residual_norm: residual per sample after linear map
    """
    try:
        translated = translated_latent_alignment(original, perturbed)
        # Compute translated MSE from the residual norm
        # residual_norm is already per-sample, square it for MSE proxy
        res_norm = float(translated.get("translation_residual_norm", 1.0))
        translated_mse = res_norm ** 2
        return {
            "translated_mse": translated_mse,
            "translated_correlation": float(translated.get("translated_alignment", 0.0)),
            "translation_r2": float(translated.get("translated_alignment", 0.0)),
        }
    except Exception:
        return {"translated_mse": 1.0, "translated_correlation": 0.0, "translation_r2": 0.0}


def _compute_interop_auc(original: list, perturbed: list) -> float:
    """Compute interoperability score as a proxy for AUC."""
    try:
        interop = interoperability_score(original, perturbed)
        return float(interop.get("interoperability_score", 0.5))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_observer_interoperability",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect baseline episodes
    episodes = collect_trace_bank(max_steps=24, seeds=[0, 1])

    # Select representative episodes for comparison
    selected = episodes[:min(12, len(episodes))]

    rng = np.random.default_rng(42)

    results_by_perturbation = {}
    all_raw_scores = []
    all_translated_scores = []

    for pert_name, pert_fn in PERTURBATION_TYPES.items():
        pert_results = []
        for ep in selected:
            original_transitions = ep.transitions
            if not original_transitions:
                continue

            kwargs = {}
            if pert_name in ("noise", "partial_observation"):
                kwargs["rng"] = rng
            perturbed_transitions = pert_fn(original_transitions, **kwargs)

            raw = _compute_raw_agreement(original_transitions, perturbed_transitions)
            translated = _compute_translated_agreement(original_transitions, perturbed_transitions)
            interop_auc = _compute_interop_auc(original_transitions, perturbed_transitions)

            row = {
                "episode_id": ep.episode_id,
                "controller": ep.controller_name,
                "world_mode": ep.world_mode,
                **raw,
                **translated,
                "interop_auc": interop_auc,
                "translation_improvement_mse": raw["raw_mse"] - translated["translated_mse"],
                "translation_improvement_corr": translated["translated_correlation"] - raw["raw_correlation"],
            }
            pert_results.append(row)
            all_raw_scores.append(raw["raw_correlation"])
            all_translated_scores.append(translated["translated_correlation"])

        if pert_results:
            results_by_perturbation[pert_name] = {
                "n_episodes": len(pert_results),
                "mean_raw_mse": float(np.mean([r["raw_mse"] for r in pert_results])),
                "mean_raw_correlation": float(np.mean([r["raw_correlation"] for r in pert_results])),
                "mean_translated_mse": float(np.mean([r["translated_mse"] for r in pert_results])),
                "mean_translated_correlation": float(np.mean([r["translated_correlation"] for r in pert_results])),
                "mean_interop_auc": float(np.mean([r["interop_auc"] for r in pert_results])),
                "mean_translation_improvement_mse": float(np.mean([r["translation_improvement_mse"] for r in pert_results])),
                "mean_translation_improvement_corr": float(np.mean([r["translation_improvement_corr"] for r in pert_results])),
                "details": pert_results[:5],
            }

    # Pass criteria
    overall_raw_corr = float(np.mean(all_raw_scores)) if all_raw_scores else 0.0
    overall_translated_corr = float(np.mean(all_translated_scores)) if all_translated_scores else 0.0
    mean_raw_mse = float(
        np.mean([result["mean_raw_mse"] for result in results_by_perturbation.values()])
    ) if results_by_perturbation else 0.0
    mean_translated_mse = float(
        np.mean([result["mean_translated_mse"] for result in results_by_perturbation.values()])
    ) if results_by_perturbation else 0.0
    translation_consistently_better = mean_translated_mse < mean_raw_mse

    # Check consistency across all perturbation types using the primary MSE signal.
    consistently_better = all(
        results_by_perturbation[p]["mean_translation_improvement_mse"] > 0
        for p in results_by_perturbation
    )

    payload = {
        "results_by_perturbation": results_by_perturbation,
        "pass_criteria": {
            "translation_gt_raw_consistently": translation_consistently_better and consistently_better,
            "overall_raw_mse": mean_raw_mse,
            "overall_translated_mse": mean_translated_mse,
            "overall_raw_correlation": overall_raw_corr,
            "overall_translated_correlation": overall_translated_corr,
            "improvement_mse": mean_raw_mse - mean_translated_mse,
            "improvement_corr": overall_translated_corr - overall_raw_corr,
        },
        "summary": {
            "n_perturbation_types": len(PERTURBATION_TYPES),
            "n_episodes_tested": len(selected),
            "n_total_comparisons": len(all_raw_scores),
        },
    }

    (output_dir / "observer_interoperability.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
