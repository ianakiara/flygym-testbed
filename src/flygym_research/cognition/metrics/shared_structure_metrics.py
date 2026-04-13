from __future__ import annotations

import numpy as np

# Shared-structure regimes are intentionally conservative:
# - coherent requires a clearly positive backbone score with low degeneracy
# - degraded still requires portable support but tolerates mild score erosion
# The numeric cutoffs are calibrated to the bounded [-1, 1]-ish scoring surface
# produced by shared_structure_profile: 0.2 marks a meaningful positive margin,
# while -0.05 still allows near-neutral but portable candidates to survive as
# degraded rather than being collapsed into degenerate convergence.
COHERENT_BACKBONE_THRESHOLD = 0.2
DEGENERACY_CEILING = 0.2
DEGRADED_PORTABLE_FLOOR = 0.2
DEGRADED_BACKBONE_FLOOR = -0.05


def shared_structure_profile(
    *,
    redundancy: float,
    portability_fraction: float,
    functional_transfer_gain: float,
    seam_risk: float,
    interop_loss: float,
    scale_drift: float,
    degeneracy_penalty: float,
    redundancy_weight: float = 0.4,
    portability_weight: float = 0.25,
    function_weight: float = 0.35,
) -> dict[str, float | str]:
    # Use authentic portability: high portability combined with high interop-loss
    # indicates false portability and should not be rewarded.
    authentic_portability = float(
        np.clip(portability_fraction * (1.0 - interop_loss), 0.0, 1.0)
    )
    coherent_core = float(
        np.clip(
            redundancy_weight * redundancy
            + portability_weight * authentic_portability
            + function_weight * max(functional_transfer_gain, 0.0),
            0.0,
            1.0,
        )
    )
    # Cross-term penalties penalise the characteristic failure modes:
    #   portability × interop_loss  → false-portable (high portability but bad cross-ctrl)
    #   portability × scale_drift   → transfer-fragile (seems portable but drifts under scale)
    portability_interop_penalty = float(
        np.clip(portability_fraction * interop_loss, 0.0, 1.0)
    )
    portability_scale_penalty = float(
        np.clip(portability_fraction * scale_drift, 0.0, 1.0)
    )
    # Penalise negative functional_transfer_gain explicitly (false/adversarial portables)
    negative_transfer_penalty = float(np.clip(-functional_transfer_gain, 0.0, 1.0))
    backbone_shared = float(
        coherent_core
        - 0.6 * seam_risk
        - 0.6 * interop_loss
        - 0.5 * scale_drift
        - 0.35 * degeneracy_penalty
        - 0.8 * portability_interop_penalty
        - 0.5 * portability_scale_penalty
        - 0.4 * negative_transfer_penalty
    )
    portable_floor = float(np.clip(portability_fraction - 0.5 * scale_drift, 0.0, 1.0))
    if (
        backbone_shared >= COHERENT_BACKBONE_THRESHOLD
        and degeneracy_penalty <= DEGENERACY_CEILING
    ):
        regime = "coherent_shared_structure"
    elif (
        portable_floor >= DEGRADED_PORTABLE_FLOOR
        and backbone_shared > DEGRADED_BACKBONE_FLOOR
    ):
        regime = "portable_but_degraded"
    else:
        regime = "degenerate_convergence"
    return {
        "coherent_core": coherent_core,
        "portable_floor": portable_floor,
        "backbone_shared": backbone_shared,
        "shared_structure_regime": regime,
    }


def transfer_hierarchy(
    *, functional_transfer_gain: float, portability_fraction: float
) -> dict[str, float | str]:
    if portability_fraction >= 0.95 and functional_transfer_gain >= 0.15:
        tier = "universal"
    elif portability_fraction >= 0.34 and functional_transfer_gain >= 0.0:
        tier = "portable"
    else:
        tier = "local"
    return {
        "transfer_hierarchy_tier": tier,
        "functional_transfer_gain": functional_transfer_gain,
        "portability_fraction": portability_fraction,
    }
