"""Global disruption signature metrics — tests whether disrupting a central
integrative channel fragments the whole system or only one skill.

If the system has genuine integration, disrupting a central channel
(e.g., ascending feedback, stabilization, or internal state) should degrade
MULTIPLE metrics simultaneously rather than just one.  This is the
'global disruption' marker.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition
from .core_metrics import (
    history_dependence,
    seam_fragility,
    self_world_separation,
    stabilization_quality,
    state_persistence,
    task_performance,
)


def compute_metric_vector(transitions: list[StepTransition]) -> np.ndarray:
    """Compute all core metrics and return as a standardised vector.

    Order: [return, stability_mean, state_autocorrelation,
    history_dependence, self_world_marker, seam_fragility].
    """
    m = {}
    m.update(task_performance(transitions))
    m.update(stabilization_quality(transitions))
    m.update(state_persistence(transitions))
    m.update(history_dependence(transitions))
    m.update(self_world_separation(transitions))
    m.update(seam_fragility(transitions))
    return np.array(
        [
            m.get("return", 0.0),
            m.get("stability_mean", 0.0),
            m.get("state_autocorrelation", 0.0),
            m.get("history_dependence", 0.0),
            m.get("self_world_marker", 0.0),
            m.get("seam_fragility", 0.0),
        ],
        dtype=np.float64,
    )


_METRIC_NAMES = (
    "return",
    "stability_mean",
    "state_autocorrelation",
    "history_dependence",
    "self_world_marker",
    "seam_fragility",
)


def global_disruption_signature(
    baseline_transitions: list[StepTransition],
    disrupted_transitions: list[StepTransition],
) -> dict[str, float]:
    """Compare baseline and disrupted metric vectors.

    Returns:
    - per-metric degradation (absolute and relative)
    - count of metrics that degraded
    - fragmentation score (1 metric degraded = local, many = global)
    """
    baseline = compute_metric_vector(baseline_transitions)
    disrupted = compute_metric_vector(disrupted_transitions)

    abs_change = disrupted - baseline
    # Relative change — use safe denominator to avoid RuntimeWarning from
    # numpy evaluating the division for all elements before masking.
    safe_baseline = np.where(np.abs(baseline) > 1e-8, np.abs(baseline), 1.0)
    rel_change = np.where(
        np.abs(baseline) > 1e-8, abs_change / safe_baseline, 0.0
    )

    degradation_threshold = -0.1  # >10% relative degradation.
    degraded_mask = rel_change < degradation_threshold
    n_degraded = int(degraded_mask.sum())
    n_total = len(baseline)

    result: dict[str, float] = {}
    for i, name in enumerate(_METRIC_NAMES):
        result[f"disruption_{name}_abs"] = float(abs_change[i])
        result[f"disruption_{name}_rel"] = float(rel_change[i])
        result[f"disruption_{name}_degraded"] = float(degraded_mask[i])

    # Fragmentation score: 0 = no degradation, 0.5 = one metric, 1.0 = all metrics.
    result["n_metrics_degraded"] = float(n_degraded)
    result["fragmentation_score"] = float(n_degraded / n_total) if n_total > 0 else 0.0
    result["is_global_disruption"] = float(n_degraded >= 3)  # Threshold: 3+ metrics.
    result["is_local_disruption"] = float(1 <= n_degraded < 3)
    result["mean_relative_change"] = float(np.mean(rel_change))

    return result
