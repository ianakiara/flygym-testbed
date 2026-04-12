from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import StepTransition
from .core_metrics import seam_fragility, summarize_metrics
from .disruption_metrics import global_disruption_signature
from .interoperability_metrics import (
    interoperability_score,
    reward_trajectory_similarity,
)
from .quotient_metrics import translation_preserves_environment



def trajectory_equivalence_strength(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    reward = reward_trajectory_similarity(transitions_a, transitions_b)
    interop = interoperability_score(transitions_a, transitions_b)
    translation = translation_preserves_environment(transitions_a, transitions_b)
    success_a = float(any(t.terminated for t in transitions_a))
    success_b = float(any(t.terminated for t in transitions_b))
    success_match = float(success_a == success_b)
    mae_penalty = 1.0 / (1.0 + reward["reward_mae"])
    score = float(
        np.clip(
            0.35 * max(reward["reward_correlation"], 0.0)
            + 0.35 * interop["interoperability_score"]
            + 0.2 * translation["environment_preservation_r2"]
            + 0.1 * success_match * mae_penalty,
            0.0,
            1.0,
        )
    )
    return {
        "trajectory_equivalence_strength": score,
        "reward_correlation": reward["reward_correlation"],
        "reward_mae": reward["reward_mae"],
        "interoperability_score": interop["interoperability_score"],
        "environment_preservation_r2": translation["environment_preservation_r2"],
        "success_match": success_match,
    }



def seam_critical_exception_score(
    transitions: list[StepTransition],
    reference_transitions: list[StepTransition] | None = None,
) -> dict[str, float]:
    seam = seam_fragility(transitions)["seam_fragility"]
    mismatch = 0.0
    if reference_transitions is not None:
        reference_seam = seam_fragility(reference_transitions)["seam_fragility"]
        mismatch = abs(seam - reference_seam)
    score = float(np.clip(seam + mismatch, 0.0, 1.0))
    return {
        "seam_critical_exception_score": score,
        "seam_fragility": seam,
        "seam_delta_vs_reference": mismatch,
    }



def compression_gain(original_count: int, compressed_count: int) -> dict[str, float]:
    if original_count <= 0:
        return {"compression_gain": 0.0, "compression_ratio": 1.0}
    ratio = compressed_count / original_count
    return {
        "compression_gain": float(np.clip(1.0 - ratio, 0.0, 1.0)),
        "compression_ratio": float(ratio),
    }



# Metrics where *lower* values indicate better performance.  The delta
# calculation must negate these so that improvement is always positive.
_LOWER_IS_BETTER = frozenset({"seam_fragility"})


def post_compression_robustness_delta(
    baseline_metrics: dict[str, float],
    compressed_metrics: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(baseline_metrics) | set(compressed_metrics))
    deltas: list[float] = []
    for key in keys:
        raw = compressed_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
        # Negate "lower is better" metrics so improvement is always positive.
        deltas.append(-raw if key in _LOWER_IS_BETTER else raw)
    mean_delta = float(np.mean(deltas)) if deltas else 0.0
    return {
        "post_compression_robustness_delta": mean_delta,
        "robustness_regressed": float(mean_delta < -0.05),
    }



def residual_utility(
    original_count: int,
    residual_count: int,
    seam_exception_count: int,
) -> dict[str, float]:
    if original_count <= 0:
        return {"residual_utility": 0.0, "residual_coverage": 0.0}
    coverage = residual_count / original_count
    if residual_count == 0:
        utility = 0.0
    else:
        utility = float(np.clip(seam_exception_count / residual_count, 0.0, 1.0))
    return {
        "residual_utility": utility,
        "residual_coverage": float(coverage),
    }



def drift_staleness_score(
    metadata: dict[str, Any],
    validation: dict[str, Any],
) -> dict[str, float]:
    age_days = float(metadata.get("age_days", 0.0))
    reuse_count = float(metadata.get("reuse_count", 0.0))
    pass_rate = float(validation.get("pass_rate", 1.0))
    staleness = age_days / (age_days + reuse_count + 1.0)
    score = float(np.clip(0.6 * staleness + 0.4 * (1.0 - pass_rate), 0.0, 1.0))
    return {
        "drift_staleness_score": score,
        "age_days": age_days,
        "reuse_count": reuse_count,
        "validation_pass_rate": pass_rate,
    }



def repairability_score(seam_report: dict[str, Any]) -> dict[str, float]:
    failures = float(seam_report.get("n_failures", 0.0))
    patchable = float(seam_report.get("n_patchable_failures", 0.0))
    if failures <= 0:
        score = 1.0
    else:
        score = float(np.clip(patchable / failures, 0.0, 1.0))
    return {
        "repairability_score": score,
        "n_failures": failures,
        "n_patchable_failures": patchable,
    }



def sleep_validation_vector(
    baseline_transitions: list[StepTransition],
    compressed_transitions: list[StepTransition],
) -> dict[str, float]:
    baseline = summarize_metrics(baseline_transitions)
    compressed = summarize_metrics(compressed_transitions)
    disruption = global_disruption_signature(baseline_transitions, compressed_transitions)
    delta = post_compression_robustness_delta(baseline, compressed)
    return {
        **baseline,
        **{f"compressed_{k}": v for k, v in compressed.items()},
        **delta,
        "fragmentation_score": disruption["fragmentation_score"],
        "global_disruption": disruption["is_global_disruption"],
    }
