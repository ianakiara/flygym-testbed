from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..interfaces import StepTransition


def task_performance(transitions: list[StepTransition]) -> dict[str, float]:
    """Compute episode return, mean reward, and binary success flag."""
    rewards = np.array(
        [transition.reward for transition in transitions], dtype=np.float64
    )
    return {
        "return": float(rewards.sum()) if len(rewards) else 0.0,
        "mean_reward": float(rewards.mean()) if len(rewards) else 0.0,
        "success": float(any(transition.terminated for transition in transitions)),
    }


def stabilization_quality(transitions: list[StepTransition]) -> dict[str, float]:
    """Mean and minimum stability over an episode."""
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
    """Lag-1 autocorrelation of an ascending-summary feature over the episode."""
    values = np.array(
        [
            transition.observation.summary.features.get(key, 0.0)
            for transition in transitions
        ],
        dtype=np.float64,
    )
    if len(values) < 2:
        return {"state_autocorrelation": 0.0}
    left = values[:-1]
    right = values[1:]
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return {"state_autocorrelation": 0.0}
    corr = float(np.corrcoef(left, right)[0, 1])
    if not np.isfinite(corr):
        return {"state_autocorrelation": 0.0}
    return {"state_autocorrelation": float(np.clip(corr, -1.0, 1.0))}


def history_dependence(transitions: list[StepTransition]) -> dict[str, float]:
    """Temporal autocorrelation of actions within coarse-state buckets.

    A controller whose actions depend on *history* (not just current state)
    will show high within-bucket autocorrelation: given the same coarse
    state, consecutive visits produce similar actions because the hidden
    state carries forward.  A random controller will show ~zero
    autocorrelation because each action is drawn independently.

    Previous implementation used *variance* within buckets, which is
    inverted — random maximises variance by definition.  Autocorrelation
    correctly captures temporal structure.
    """
    grouped_actions: dict[tuple[int, int], list[float]] = defaultdict(list)
    for transition in transitions:
        target_vector = np.asarray(
            transition.observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        # Round target vectors onto a 0.5-unit grid for finer bucketing
        # while still grouping similar states together.
        key = tuple(np.round(target_vector * 2, 0).astype(int))
        if hasattr(transition.action, "move_intent"):
            grouped_actions[key].append(float(transition.action.move_intent))
    if not grouped_actions:
        return {"history_dependence": 0.0}

    autocorrs: list[float] = []
    for actions in grouped_actions.values():
        if len(actions) < 3:
            continue
        a = np.array(actions, dtype=np.float64)
        left, right = a[:-1], a[1:]
        if np.allclose(left, left[0]) or np.allclose(right, right[0]):
            # Constant actions — perfect persistence but uninformative
            autocorrs.append(1.0)
            continue
        corr = float(np.corrcoef(left, right)[0, 1])
        if not np.isnan(corr):
            autocorrs.append(corr)

    if not autocorrs:
        return {"history_dependence": 0.0}
    # Clamp to [0, 1] — negative autocorrelation means oscillation, not
    # history dependence, so treat it as zero.
    return {"history_dependence": float(np.clip(np.mean(autocorrs), 0.0, 1.0))}


def self_world_separation(transitions: list[StepTransition]) -> dict[str, float]:
    """Measure whether the controller distinguishes self-caused from world-caused changes.

    Uses two complementary signals:

    1. **Action response to events** — correlation between external world
       events and the *change* in move_intent at the next step.  A controller
       that distinguishes self from world should modulate its actions
       differently when the world perturbs vs. when it moves itself.

    2. **Target-vector disruption** — mean absolute target-vector change on
       event steps vs. non-event steps.  If external events disrupt target
       tracking, the ratio reveals whether the controller compensates.

    Previous implementation used ``body_speed_mm_s`` which is disconnected
    from world events in BodylessBodyLayer (body_speed comes from synthetic
    body positions, not from world perturbations).
    """
    if len(transitions) < 4:
        return {"self_world_marker": 0.0}

    external = np.array(
        [float(t.info.get("external_world_event", False)) for t in transitions],
        dtype=np.float64,
    )
    if np.all(external == external[0]):
        return {"self_world_marker": 0.0}

    # --- Signal 1: action change response to events ---
    moves = np.array(
        [
            float(t.action.move_intent) if hasattr(t.action, "move_intent") else 0.0
            for t in transitions
        ],
        dtype=np.float64,
    )
    action_changes = np.abs(np.diff(moves))
    # Align: event at step i → action change at step i (diff between i and i-1)
    event_flags = external[1:]  # skip first since diff is one shorter

    if len(action_changes) < 2 or np.allclose(action_changes, action_changes[0]):
        action_signal = 0.0
    else:
        event_mask = event_flags > 0.5
        non_event_mask = ~event_mask
        if event_mask.sum() > 0 and non_event_mask.sum() > 0:
            mean_event = float(action_changes[event_mask].mean())
            mean_non_event = float(action_changes[non_event_mask].mean())
            # Ratio: how much MORE the action changes on event steps
            action_signal = (mean_event - mean_non_event) / (
                mean_non_event + 1e-8
            )
        else:
            action_signal = 0.0

    # --- Signal 2: target vector disruption ---
    tv_changes: list[float] = []
    for i in range(1, len(transitions)):
        tv_prev = np.asarray(
            transitions[i - 1].observation.world.observables.get(
                "target_vector", np.zeros(2)
            ),
            dtype=np.float64,
        )
        tv_curr = np.asarray(
            transitions[i].observation.world.observables.get(
                "target_vector", np.zeros(2)
            ),
            dtype=np.float64,
        )
        tv_changes.append(float(np.linalg.norm(tv_curr - tv_prev)))
    tv_arr = np.array(tv_changes, dtype=np.float64)
    event_flags_tv = external[1:]

    if len(tv_arr) >= 2 and not np.all(event_flags_tv == event_flags_tv[0]):
        ev_mask = event_flags_tv > 0.5
        nev_mask = ~ev_mask
        if ev_mask.sum() > 0 and nev_mask.sum() > 0:
            tv_event = float(tv_arr[ev_mask].mean())
            tv_non = float(tv_arr[nev_mask].mean())
            tv_signal = (tv_event - tv_non) / (tv_non + 1e-8)
        else:
            tv_signal = 0.0
    else:
        tv_signal = 0.0

    # --- Activity gate ---
    # A controller that barely moves (std(move_intent) ≈ 0) trivially
    # separates self from world because it produces zero self-caused
    # changes.  Discount tv_signal by the controller's activity level
    # so that standing still does not score as "good separation."
    move_std = float(np.std(moves))
    # Sigmoid-like gate: activity < 0.01 → weight ≈ 0; activity > 0.1 → weight ≈ 1
    activity_weight = float(np.clip(move_std / 0.05, 0.0, 1.0))

    # Composite: action_signal (always valid) + activity-gated tv_signal
    composite = float(
        np.clip(0.5 * action_signal + 0.5 * tv_signal * activity_weight, -1.0, 1.0)
    )
    return {"self_world_marker": composite}


def seam_fragility(transitions: list[StepTransition]) -> dict[str, float]:
    """Boundary-sensitive seam score from target, action, and state discontinuities."""
    if not transitions:
        return {"seam_fragility": 0.0}

    components = []
    body_deltas = []
    for transition in transitions:
        body_log = transition.info.get("body_log", {})
        if isinstance(body_log, dict) and "mean_target_delta" in body_log:
            body_deltas.append(float(body_log["mean_target_delta"]))

    if body_deltas:
        components.append(float(np.mean(body_deltas)))

    boundary_scores = []
    for prev, curr in zip(transitions[:-1], transitions[1:]):
        prev_obs = prev.observation
        curr_obs = curr.observation

        prev_target = np.asarray(
            prev_obs.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        curr_target = np.asarray(
            curr_obs.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        target_delta = float(np.linalg.norm(curr_target - prev_target))

        prev_pos = np.asarray(prev_obs.raw_body.body_positions, dtype=np.float64)
        curr_pos = np.asarray(curr_obs.raw_body.body_positions, dtype=np.float64)
        body_delta = float(np.linalg.norm(curr_pos - prev_pos) / max(prev_pos.size, 1))

        prev_action = np.array(
            [
                float(getattr(prev.action, "move_intent", 0.0)),
                float(getattr(prev.action, "turn_intent", 0.0)),
                float(getattr(prev.action, "speed_modulation", 0.0)),
            ],
            dtype=np.float64,
        )
        curr_action = np.array(
            [
                float(getattr(curr.action, "move_intent", 0.0)),
                float(getattr(curr.action, "turn_intent", 0.0)),
                float(getattr(curr.action, "speed_modulation", 0.0)),
            ],
            dtype=np.float64,
        )
        action_delta = float(np.linalg.norm(curr_action - prev_action))

        prev_reward = float(prev.reward)
        curr_reward = float(curr.reward)
        reward_delta = abs(curr_reward - prev_reward)

        boundary_weight = 1.0
        if curr.info.get("seam_corruption_applied") or prev.info.get("seam_corruption_applied"):
            boundary_weight += float(curr.info.get("seam_corruption_magnitude", 0.0))
        if curr.info.get("delayed_target_mismatch") or prev.info.get("delayed_target_mismatch"):
            boundary_weight += 0.5
        if prev_obs.world.mode != curr_obs.world.mode:
            boundary_weight += 0.75

        boundary_scores.append(
            boundary_weight * (
                0.4 * target_delta
                + 0.3 * action_delta
                + 0.2 * body_delta
                + 0.1 * reward_delta
            )
        )

    if boundary_scores:
        components.append(float(np.mean(boundary_scores)))

    return {"seam_fragility": float(np.mean(components)) if components else 0.0}


def summarize_metrics(transitions: list[StepTransition]) -> dict[str, float]:
    """Run all core metric functions and merge results into one dictionary."""
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
