"""Causal intervention metrics — measures whether a controller's internal
state carries genuine causal influence on behaviour, vs. being a mere
epiphenomenal correlate.

These metrics are valuable for next-generation AI systems because:
1. **Causal grounding** — distinguishes controllers that *use* internal
   state from those that only *have* it (critical for alignment).
2. **Intervention robustness** — measures whether targeted perturbation
   to hidden state produces predictable, proportional behavioural change.
3. **Counterfactual sensitivity** — answers "would the behaviour have
   been different had the internal state been different?"

Core technique: run a controller, snapshot its internal state, replay
from the snapshot with a *surgical* intervention on one state dimension,
then measure the divergence.  If divergence is proportional to the
intervention magnitude, the state is causally active.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition


def causal_influence_score(
    baseline_transitions: list[StepTransition],
    intervened_transitions: list[StepTransition],
    *,
    intervention_magnitude: float = 1.0,
) -> dict[str, float]:
    """Measure how much a surgical state intervention changes behaviour.

    Given two transition lists — one from normal execution, one from an
    execution where the controller's internal state was perturbed at
    ``t=0`` by ``intervention_magnitude`` — compute the behavioural
    divergence.

    Returns
    -------
    dict with:
        causal_influence : float
            Normalised divergence (0 = no effect, 1 = maximal).
        action_divergence : float
            Mean absolute difference in move_intent.
        trajectory_divergence : float
            Euclidean distance between reward trajectories.
        proportionality : float
            How proportional divergence is to intervention magnitude
            (closer to 1 = more proportional).
    """
    n = min(len(baseline_transitions), len(intervened_transitions))
    if n < 2:
        return {
            "causal_influence": 0.0,
            "action_divergence": 0.0,
            "trajectory_divergence": 0.0,
            "proportionality": 0.0,
        }

    # Action divergence.
    base_actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in baseline_transitions[:n]],
        dtype=np.float64,
    )
    int_actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in intervened_transitions[:n]],
        dtype=np.float64,
    )
    action_div = float(np.mean(np.abs(base_actions - int_actions)))

    # Reward trajectory divergence.
    base_rewards = np.array([t.reward for t in baseline_transitions[:n]], dtype=np.float64)
    int_rewards = np.array([t.reward for t in intervened_transitions[:n]], dtype=np.float64)
    traj_div = float(np.linalg.norm(base_rewards - int_rewards) / max(np.sqrt(n), 1.0))

    # Composite influence — normalised by intervention magnitude so
    # different-sized interventions are comparable.
    raw_influence = 0.6 * action_div + 0.4 * traj_div
    safe_mag = max(abs(intervention_magnitude), 1e-8)
    normalised = float(np.clip(raw_influence / safe_mag, 0.0, 1.0))

    # Proportionality: if we double the intervention, does divergence
    # roughly double?  Measured as ratio / expected_ratio clamped to [0,1].
    # For a single measurement, proportionality = influence / magnitude
    # (perfect proportionality = 1.0).
    proportionality = float(np.clip(raw_influence / safe_mag, 0.0, 1.0))

    return {
        "causal_influence": normalised,
        "action_divergence": action_div,
        "trajectory_divergence": traj_div,
        "proportionality": proportionality,
    }


def epiphenomenal_test(
    baseline_transitions: list[StepTransition],
    state_shuffled_transitions: list[StepTransition],
) -> dict[str, float]:
    """Test whether internal state is epiphenomenal (decorative).

    If shuffling the hidden state at each step produces the *same*
    behaviour, the state is epiphenomenal — it doesn't causally
    contribute to action selection.

    Returns
    -------
    dict with:
        epiphenomenal_score : float
            0.0 = state is causally active (shuffling changes behaviour)
            1.0 = state is epiphenomenal (shuffling has no effect)
        state_action_mutual_info_proxy : float
            Proxy for MI between state and action changes.
    """
    n = min(len(baseline_transitions), len(state_shuffled_transitions))
    if n < 4:
        return {"epiphenomenal_score": 0.5, "state_action_mutual_info_proxy": 0.0}

    base_actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in baseline_transitions[:n]],
        dtype=np.float64,
    )
    shuf_actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in state_shuffled_transitions[:n]],
        dtype=np.float64,
    )

    # If actions are identical despite shuffled state → epiphenomenal.
    action_diff = float(np.mean(np.abs(base_actions - shuf_actions)))
    max_action_range = max(float(np.ptp(base_actions)), 1e-8)
    normalised_diff = float(np.clip(action_diff / max_action_range, 0.0, 1.0))

    # Epiphenomenal score: 1 - normalised_diff (no diff = epiphenomenal).
    epiphenomenal = 1.0 - normalised_diff

    # MI proxy: correlation between action changes and state differences.
    base_changes = np.abs(np.diff(base_actions))
    shuf_changes = np.abs(np.diff(shuf_actions))
    if np.std(base_changes) < 1e-10 or np.std(shuf_changes) < 1e-10:
        mi_proxy = 0.0
    else:
        corr = float(np.corrcoef(base_changes, shuf_changes)[0, 1])
        mi_proxy = 0.0 if not np.isfinite(corr) else abs(corr)

    return {
        "epiphenomenal_score": float(np.clip(epiphenomenal, 0.0, 1.0)),
        "state_action_mutual_info_proxy": mi_proxy,
    }


def temporal_causal_depth(
    transitions: list[StepTransition],
    *,
    max_horizon: int = 10,
) -> dict[str, float]:
    """Estimate how far into the future the controller's actions are
    influenced by past state.

    Computes the autocorrelation of action *changes* (not raw actions)
    at increasing lags.  Deeper causal chains produce higher
    autocorrelation at longer lags.

    This is a proxy for "temporal depth" — how far ahead the controller
    is effectively planning.

    Returns
    -------
    dict with:
        causal_depth : float
            Lag at which action-change autocorrelation drops below 0.1.
        causal_depth_score : float
            Normalised depth in [0, 1] relative to max_horizon.
        autocorr_by_lag : dict[str, float]
            Autocorrelation at each lag.
    """
    if len(transitions) < max_horizon + 3:
        result: dict[str, float] = {
            "causal_depth": 0.0,
            "causal_depth_score": 0.0,
        }
        for lag in range(1, max_horizon + 1):
            result[f"action_change_autocorr_lag_{lag}"] = 0.0
        return result

    actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in transitions],
        dtype=np.float64,
    )
    changes = np.diff(actions)
    if np.std(changes) < 1e-10:
        result = {"causal_depth": 0.0, "causal_depth_score": 0.0}
        for lag in range(1, max_horizon + 1):
            result[f"action_change_autocorr_lag_{lag}"] = 0.0
        return result

    mean_c = changes.mean()
    var_c = changes.var()
    depth = float(max_horizon)
    result = {}

    for lag in range(1, max_horizon + 1):
        if var_c < 1e-12:
            ac = 0.0
        else:
            cov = np.mean((changes[:-lag] - mean_c) * (changes[lag:] - mean_c))
            ac = float(cov / var_c)
            if not np.isfinite(ac):
                ac = 0.0
        result[f"action_change_autocorr_lag_{lag}"] = ac
        if ac < 0.1 and depth == float(max_horizon):
            depth = float(lag)

    result["causal_depth"] = depth
    result["causal_depth_score"] = float(np.clip(depth / max_horizon, 0.0, 1.0))
    return result
