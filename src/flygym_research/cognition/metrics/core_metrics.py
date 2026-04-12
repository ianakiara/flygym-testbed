from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..interfaces import StepTransition


def task_performance(transitions: list[StepTransition]) -> dict[str, float]:
    rewards = np.array(
        [transition.reward for transition in transitions], dtype=np.float64
    )
    return {
        "return": float(rewards.sum()) if len(rewards) else 0.0,
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "success": float(any(transition.terminated for transition in transitions)),
    }


def stabilization_quality(transitions: list[StepTransition]) -> dict[str, float]:
    stability = np.array(
        [
            transition.observation.summary.features.get("stability", 0.0)
            for transition in transitions
        ],
        dtype=np.float64,
    )
    return {
        "stability_mean": float(stability.mean()) if len(stability) else 0.0,
        "stability_min": float(stability.min()) if len(stability) else 0.0,
    }


def state_persistence(
    transitions: list[StepTransition],
    *,
    key: str = "phase",
) -> dict[str, float]:
    values = np.array(
        [
            transition.observation.summary.features.get(key, 0.0)
            for transition in transitions
        ],
        dtype=np.float64,
    )
    if len(values) < 2:
        return {"state_autocorrelation": 0.0}
    return {"state_autocorrelation": float(np.corrcoef(values[:-1], values[1:])[0, 1])}


def history_dependence(transitions: list[StepTransition]) -> dict[str, float]:
    grouped_actions: dict[tuple[int, int], list[float]] = defaultdict(list)
    for transition in transitions:
        target_vector = np.asarray(
            transition.observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        key = tuple(np.round(target_vector, 0).astype(int))
        if hasattr(transition.action, "move_intent"):
            grouped_actions[key].append(float(transition.action.move_intent))
    if not grouped_actions:
        return {"history_dependence": 0.0}
    variances = [
        np.var(actions) for actions in grouped_actions.values() if len(actions) > 1
    ]
    return {"history_dependence": float(np.mean(variances)) if variances else 0.0}


def self_world_separation(transitions: list[StepTransition]) -> dict[str, float]:
    external = np.array(
        [
            float(transition.info.get("external_world_event", False))
            for transition in transitions
        ],
        dtype=np.float64,
    )
    speed = np.array(
        [
            transition.observation.summary.features.get("body_speed_mm_s", 0.0)
            for transition in transitions
        ],
        dtype=np.float64,
    )
    if len(external) < 2 or np.all(external == external[0]):
        return {"self_world_marker": 0.0}
    return {"self_world_marker": float(np.corrcoef(external, speed)[0, 1])}


def seam_fragility(transitions: list[StepTransition]) -> dict[str, float]:
    mean_delta = []
    for transition in transitions:
        body_log = transition.info.get("body_log", {})
        if isinstance(body_log, dict) and "mean_target_delta" in body_log:
            mean_delta.append(float(body_log["mean_target_delta"]))
    return {"seam_fragility": float(np.mean(mean_delta)) if mean_delta else 0.0}


def summarize_metrics(transitions: list[StepTransition]) -> dict[str, float]:
    metrics = {}
    for fn in (
        task_performance,
        stabilization_quality,
        state_persistence,
        history_dependence,
        self_world_separation,
        seam_fragility,
    ):
        metrics.update(fn(transitions))
    return metrics
