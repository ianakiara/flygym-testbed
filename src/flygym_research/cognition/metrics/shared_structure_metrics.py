from __future__ import annotations

import numpy as np


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
    coherent_core = float(
        np.clip(
            redundancy_weight * redundancy
            + portability_weight * portability_fraction
            + function_weight * functional_transfer_gain,
            0.0,
            1.0,
        )
    )
    backbone_shared = float(
        coherent_core
        - 0.6 * seam_risk
        - 0.6 * interop_loss
        - 0.5 * scale_drift
        - 0.7 * degeneracy_penalty
    )
    portable_floor = float(np.clip(portability_fraction - 0.5 * scale_drift, 0.0, 1.0))
    if backbone_shared >= 0.2 and degeneracy_penalty <= 0.2:
        regime = "coherent_shared_structure"
    elif portable_floor >= 0.2 and backbone_shared > -0.05:
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
