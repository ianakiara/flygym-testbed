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
from ..metrics.interoperability_metrics import extract_state_matrix
from .exp_sleep_trace_compressor import collect_trace_bank


# ---------------------------------------------------------------------------
# Observer perturbation generators
# ---------------------------------------------------------------------------

def _perturb_transitions_noise(transitions: list, *, scale: float = 0.3, rng: np.random.Generator | None = None) -> list:
    """Add Gaussian noise to observations."""
    rng = rng or np.random.default_rng(0)
    perturbed = []
    for t in transitions:
        from ..interfaces import StepTransition, BrainObservation, RawBodyFeedback
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
        noisy_obs = BrainObservation(
            raw_body=noisy_feedback,
            summary=t.observation.summary,
            world=t.observation.world,
            history=t.observation.history,
        )
        perturbed.append(StepTransition(
            observation=noisy_obs, action=t.action,
            reward=t.reward, terminated=t.terminated,
            truncated=t.truncated, info=t.info,
        ))
    return perturbed


def _perturb_transitions_scaling(transitions: list, *, factor: float = 2.0) -> list:
    """Scale observations by a constant factor."""
    perturbed = []
    for t in transitions:
        from ..interfaces import StepTransition, BrainObservation, RawBodyFeedback
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
        scaled_obs = BrainObservation(
            raw_body=scaled_feedback,
            summary=t.observation.summary,
            world=t.observation.world,
            history=t.observation.history,
        )
        perturbed.append(StepTransition(
            observation=scaled_obs, action=t.action,
            reward=t.reward, terminated=t.terminated,
            truncated=t.truncated, info=t.info,
        ))
    return perturbed


def _perturb_transitions_bias(transitions: list, *, bias: float = 1.5) -> list:
    """Add constant bias to observations."""
    perturbed = []
    for t in transitions:
        from ..interfaces import StepTransition, BrainObservation, RawBodyFeedback
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
        biased_obs = BrainObservation(
            raw_body=biased_feedback,
            summary=t.observation.summary,
            world=t.observation.world,
            history=t.observation.history,
        )
        perturbed.append(StepTransition(
            observation=biased_obs, action=t.action,
            reward=t.reward, terminated=t.terminated,
            truncated=t.truncated, info=t.info,
        ))
    return perturbed


def _perturb_transitions_partial(transitions: list, *, drop_fraction: float = 0.3, rng: np.random.Generator | None = None) -> list:
    """Drop a fraction of observation channels (partial observability)."""
    rng = rng or np.random.default_rng(0)
    perturbed = []
    for t in transitions:
        from ..interfaces import StepTransition, BrainObservation, RawBodyFeedback
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
        partial_obs = BrainObservation(
            raw_body=partial_feedback,
            summary=t.observation.summary,
            world=t.observation.world,
            history=t.observation.history,
        )
        perturbed.append(StepTransition(
            observation=partial_obs, action=t.action,
            reward=t.reward, terminated=t.terminated,
            truncated=t.truncated, info=t.info,
        ))
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
    translation_consistently_better = overall_translated_corr > overall_raw_corr

    # Check consistency across all perturbation types
    consistently_better = all(
        results_by_perturbation[p]["mean_translation_improvement_corr"] > 0
        for p in results_by_perturbation
    )

    payload = {
        "results_by_perturbation": results_by_perturbation,
        "pass_criteria": {
            "translation_gt_raw_consistently": translation_consistently_better and consistently_better,
            "overall_raw_correlation": overall_raw_corr,
            "overall_translated_correlation": overall_translated_corr,
            "improvement": overall_translated_corr - overall_raw_corr,
        },
        "summary": {
            "n_perturbation_types": len(PERTURBATION_TYPES),
            "n_episodes_tested": len(selected),
            "n_total_comparisons": len(all_raw_scores),
        },
    }

    (output_dir / "observer_interoperability.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
