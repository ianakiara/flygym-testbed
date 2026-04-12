"""Controller interoperability metrics — measures whether different controller
families produce translatable latent/object representations across the same
tasks and worlds.

These metrics compare internal state trajectories, action distributions, and
observation-encoding similarity across controller families run on identical
episodes.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition


def controller_action_distribution(
    transitions: list[StepTransition],
) -> dict[str, float]:
    """Summarise the action distribution for a controller run.

    Returns mean, std, and range of move_intent and turn_intent.
    """
    moves: list[float] = []
    turns: list[float] = []
    for t in transitions:
        if hasattr(t.action, "move_intent"):
            moves.append(float(t.action.move_intent))
            turns.append(float(t.action.turn_intent))
    if not moves:
        return {
            "action_move_mean": 0.0,
            "action_move_std": 0.0,
            "action_turn_mean": 0.0,
            "action_turn_std": 0.0,
            "action_move_range": 0.0,
            "action_turn_range": 0.0,
        }
    m = np.array(moves, dtype=np.float64)
    t = np.array(turns, dtype=np.float64)
    return {
        "action_move_mean": float(m.mean()),
        "action_move_std": float(m.std()),
        "action_turn_mean": float(t.mean()),
        "action_turn_std": float(t.std()),
        "action_move_range": float(m.max() - m.min()),
        "action_turn_range": float(t.max() - t.min()),
    }


def latent_state_similarity(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
    *,
    key: str = "stability",
) -> dict[str, float]:
    """Compare ascending-summary feature trajectories from two controller runs.

    Uses correlation and mean-absolute-error between the feature time series
    to quantify whether controllers produce similar internal representations.
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 2:
        return {"latent_correlation": 0.0, "latent_mae": 0.0}
    vals_a = np.array(
        [t.observation.summary.features.get(key, 0.0) for t in transitions_a[:n]],
        dtype=np.float64,
    )
    vals_b = np.array(
        [t.observation.summary.features.get(key, 0.0) for t in transitions_b[:n]],
        dtype=np.float64,
    )
    if np.allclose(vals_a, vals_a[0]) or np.allclose(vals_b, vals_b[0]):
        return {"latent_correlation": 0.0, "latent_mae": float(np.mean(np.abs(vals_a - vals_b)))}
    corr = float(np.corrcoef(vals_a, vals_b)[0, 1])
    return {
        "latent_correlation": 0.0 if np.isnan(corr) else corr,
        "latent_mae": float(np.mean(np.abs(vals_a - vals_b))),
    }


def reward_trajectory_similarity(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Compare reward trajectories from two controller runs."""
    n = min(len(transitions_a), len(transitions_b))
    if n < 2:
        return {"reward_correlation": 0.0, "reward_mae": 0.0}
    r_a = np.array([t.reward for t in transitions_a[:n]], dtype=np.float64)
    r_b = np.array([t.reward for t in transitions_b[:n]], dtype=np.float64)
    if np.allclose(r_a, r_a[0]) or np.allclose(r_b, r_b[0]):
        return {"reward_correlation": 0.0, "reward_mae": float(np.mean(np.abs(r_a - r_b)))}
    corr = float(np.corrcoef(r_a, r_b)[0, 1])
    return {
        "reward_correlation": 0.0 if np.isnan(corr) else corr,
        "reward_mae": float(np.mean(np.abs(r_a - r_b))),
    }


def interoperability_score(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Composite interoperability score: average of reward and latent correlations."""
    latent = latent_state_similarity(transitions_a, transitions_b)
    reward = reward_trajectory_similarity(transitions_a, transitions_b)
    composite = 0.5 * abs(latent["latent_correlation"]) + 0.5 * abs(
        reward["reward_correlation"]
    )
    return {
        **latent,
        **reward,
        "interoperability_score": composite,
    }
