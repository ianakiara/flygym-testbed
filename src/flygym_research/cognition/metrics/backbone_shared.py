"""BackboneShared composite metric — assembles all component metrics.

    BackboneShared(X) = Ω(X) − λ·InteropLoss(X) − μ·SeamFragility(X) − ν·ScaleDrift(X)

All component metrics already exist as individual modules. This module
assembles them into the unified composite score and provides:
- Compression gating (gate = BackboneShared ≥ threshold)
- Cross-world validation (same-world vs cross-world comparison)
- Component contribution breakdown
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..interfaces import StepTransition
from ..metrics import (
    interoperability_score,
    seam_fragility,
)
from ..metrics.quotient_metrics import degeneracy_score


@dataclass(slots=True)
class BackboneSharedConfig:
    """Weights for BackboneShared composite score."""

    lambda_interop: float = 0.6    # interop loss penalty
    mu_seam: float = 0.6           # seam fragility penalty
    nu_drift: float = 0.5          # scale drift penalty
    alpha_quotient: float = 1.0    # quotient redundancy weight
    gate_threshold: float = 0.1    # minimum score for compression approval


def compute_backbone_shared(
    transitions_dict: dict[str, list[StepTransition]],
    *,
    cross_world_transitions: dict[str, list[StepTransition]] | None = None,
    config: BackboneSharedConfig | None = None,
) -> dict[str, float]:
    """Compute the BackboneShared composite score.

    Parameters
    ----------
    transitions_dict : dict
        Controller name → transitions for the primary world.
    cross_world_transitions : dict, optional
        Controller name → transitions for a different world (for scale drift).
    config : BackboneSharedConfig, optional
        Weight configuration.

    Returns
    -------
    dict with:
    - backbone_shared: the composite score
    - quotient_redundancy: Ω component
    - interop_loss: 1 − mean interoperability
    - seam_fragility_mean: mean seam fragility across controllers
    - scale_drift: cross-world divergence (0 if no cross-world data)
    - gate_approved: whether score exceeds gate_threshold
    """
    config = config or BackboneSharedConfig()
    names = sorted(transitions_dict.keys())

    if len(names) < 2:
        return {
            "backbone_shared": 0.0,
            "quotient_redundancy": 0.0,
            "interop_loss": 0.0,
            "seam_fragility_mean": 0.0,
            "scale_drift": 0.0,
            "gate_approved": False,
            "components": {},
        }

    # ── Component 1: Quotient Redundancy (Ω) ────────────────────────────
    degeneracy = degeneracy_score(transitions_dict)
    q_red = float(
        np.clip(
            0.5 * degeneracy.get("degeneracy_ratio", 0.0)
            + 0.5 * degeneracy.get("information_loss", 0.0),
            0.0,
            1.0,
        )
    )

    # ── Component 2: Interop Loss ────────────────────────────────────────
    interop_scores = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            try:
                result = interoperability_score(
                    transitions_dict[n1], transitions_dict[n2],
                )
                interop_scores.append(result["interoperability_score"])
            except Exception:
                interop_scores.append(0.0)
    mean_interop = float(np.mean(interop_scores)) if interop_scores else 0.5
    interop_loss = 1.0 - mean_interop

    # ── Component 3: Seam Fragility ──────────────────────────────────────
    seam_scores = []
    for name in names:
        try:
            sf = seam_fragility(transitions_dict[name])
            seam_scores.append(sf["seam_fragility"])
        except Exception:
            seam_scores.append(0.0)
    seam_mean = float(np.mean(seam_scores)) if seam_scores else 0.0

    # ── Component 4: Scale Drift ─────────────────────────────────────────
    scale_drift = 0.0
    if cross_world_transitions:
        from ..metrics.quotient_metrics import counterfactual_divergence

        common = sorted(set(names) & set(cross_world_transitions.keys()))
        if len(common) >= 2:
            try:
                div = counterfactual_divergence(
                    {n: transitions_dict[n] for n in common},
                    {n: cross_world_transitions[n] for n in common},
                )
                scale_drift = div.get("mean_reward_divergence", 0.0)
            except Exception:
                scale_drift = 0.0

    # ── Composite Score ──────────────────────────────────────────────────
    backbone_shared = (
        config.alpha_quotient * q_red
        - config.lambda_interop * interop_loss
        - config.mu_seam * seam_mean
        - config.nu_drift * scale_drift
    )

    gate_approved = backbone_shared >= config.gate_threshold

    return {
        "backbone_shared": float(backbone_shared),
        "quotient_redundancy": float(q_red),
        "interop_loss": float(interop_loss),
        "mean_interop": float(mean_interop),
        "seam_fragility_mean": float(seam_mean),
        "scale_drift": float(scale_drift),
        "gate_approved": bool(gate_approved),
        "gate_threshold": config.gate_threshold,
        "components": {
            "alpha_quotient": config.alpha_quotient,
            "lambda_interop": config.lambda_interop,
            "mu_seam": config.mu_seam,
            "nu_drift": config.nu_drift,
            "Omega_contribution": float(config.alpha_quotient * q_red),
            "interop_penalty": float(-config.lambda_interop * interop_loss),
            "seam_penalty": float(-config.mu_seam * seam_mean),
            "drift_penalty": float(-config.nu_drift * scale_drift),
        },
    }


def compare_world_modes(
    same_world_transitions: dict[str, list[StepTransition]],
    cross_world_transitions: dict[str, list[StepTransition]],
    *,
    config: BackboneSharedConfig | None = None,
) -> dict[str, Any]:
    """Compare BackboneShared scores between same-world and cross-world.

    If BackboneShared correctly distinguishes the two modes, the same-world
    score should be higher (less drift penalty, more coherent interop).
    """
    same_result = compute_backbone_shared(
        same_world_transitions, config=config,
    )
    cross_result = compute_backbone_shared(
        same_world_transitions,
        cross_world_transitions=cross_world_transitions,
        config=config,
    )

    return {
        "same_world_backbone_shared": same_result["backbone_shared"],
        "cross_world_backbone_shared": cross_result["backbone_shared"],
        "backbone_shared_delta": same_result["backbone_shared"] - cross_result["backbone_shared"],
        "same_world_gate": same_result["gate_approved"],
        "cross_world_gate": cross_result["gate_approved"],
        "distinguishes_modes": same_result["backbone_shared"] > cross_result["backbone_shared"],
        "same_world_detail": same_result,
        "cross_world_detail": cross_result,
    }
