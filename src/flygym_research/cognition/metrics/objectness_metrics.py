"""Shared objectness proxy metrics — measures whether stable target or goal
representations survive controller swaps, viewpoint changes, world changes,
and sensory noise.

Objectness here is an engineering proxy: does the controller maintain a
coherent, persistent representation of tracked targets across perturbations?
This is NOT a consciousness claim — it is a control-integration marker.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition


def target_representation_stability(
    transitions: list[StepTransition],
) -> dict[str, float]:
    """Measure how stably the controller tracks target_vector over time.

    Returns the coefficient of variation of the target distance trajectory
    (lower = more stable tracking) and the autocorrelation of the target
    direction angle.
    """
    distances: list[float] = []
    angles: list[float] = []
    for t in transitions:
        tv = t.observation.world.observables.get("target_vector")
        if tv is not None:
            tv_arr = np.asarray(tv, dtype=np.float64)
            distances.append(float(np.linalg.norm(tv_arr)))
            angles.append(float(np.arctan2(tv_arr[1], tv_arr[0])))

    if len(distances) < 2:
        return {
            "target_distance_cv": 0.0,
            "target_angle_autocorr": 0.0,
            "target_persistence": 0.0,
        }

    d = np.array(distances, dtype=np.float64)
    a = np.array(angles, dtype=np.float64)

    # Coefficient of variation of distance.
    cv = float(d.std() / (d.mean() + 1e-8))

    # Lag-1 autocorrelation of angle.
    if np.allclose(a[:-1], a[0]) or np.allclose(a[1:], a[0]):
        angle_autocorr = 0.0
    else:
        corr = float(np.corrcoef(a[:-1], a[1:])[0, 1])
        angle_autocorr = 0.0 if np.isnan(corr) else corr

    # Persistence: fraction of steps where target distance decreases.
    improvements = np.sum(np.diff(d) < 0)
    persistence = float(improvements / max(len(d) - 1, 1))

    return {
        "target_distance_cv": cv,
        "target_angle_autocorr": angle_autocorr,
        "target_persistence": persistence,
    }


def cross_condition_objectness(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Compare target representation stability across two conditions.

    Useful for checking whether objectness survives controller swap,
    world change, or sensory noise injection.
    """
    metrics_a = target_representation_stability(transitions_a)
    metrics_b = target_representation_stability(transitions_b)

    return {
        "objectness_persistence_diff": abs(
            metrics_a["target_persistence"] - metrics_b["target_persistence"]
        ),
        "objectness_cv_diff": abs(
            metrics_a["target_distance_cv"] - metrics_b["target_distance_cv"]
        ),
        "objectness_autocorr_diff": abs(
            metrics_a["target_angle_autocorr"] - metrics_b["target_angle_autocorr"]
        ),
        "condition_a_persistence": metrics_a["target_persistence"],
        "condition_b_persistence": metrics_b["target_persistence"],
    }


def shared_objectness_score(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Composite shared objectness score.

    Higher values indicate that target representations are more similar
    across conditions — a candidate marker for controller-invariant
    object tracking.
    """
    cross = cross_condition_objectness(transitions_a, transitions_b)
    # Score: 1.0 minus normalised differences (clamped to [0, 1]).
    raw = 1.0 - (
        cross["objectness_persistence_diff"]
        + cross["objectness_cv_diff"] * 0.5
        + cross["objectness_autocorr_diff"] * 0.5
    )
    return {
        **cross,
        "shared_objectness_score": float(np.clip(raw, 0.0, 1.0)),
    }
