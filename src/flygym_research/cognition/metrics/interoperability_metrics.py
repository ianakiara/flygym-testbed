"""Controller interoperability metrics — measures whether different controller
families produce translatable latent/object representations across the same
tasks and worlds.

The key distinction:
  - **raw alignment**: direct element-wise correlation of state vectors.
    Measures whether controllers produce *identical* internal trajectories.
  - **translated alignment**: fit a linear map T: z_a → z_b and measure R²
    of the prediction.  Measures whether shared *structure* exists even when
    raw representations differ.

Reward is deliberately excluded from the composite interoperability score
because it is an environment-imposed global scalar that inflates agreement
without revealing internal structural similarity.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition


# ── State vector extraction ─────────────────────────────────────────────


def _extract_state_vector(t: StepTransition) -> np.ndarray:
    """Build a rich state vector z_t from a single step transition.

    Components (14-dimensional):
      [0:2]  avatar_xy          — world position
      [2]    heading             — orientation
      [3:5]  target_vector       — relative target direction
      [5]    move_intent         — action: forward/back
      [6]    turn_intent         — action: left/right
      [7]    speed_modulation    — action: speed modifier
      [8]    distance_to_target  — scalar distance from info
      [9]    body_speed          — ascending: movement rate
      [10]   locomotion_quality  — ascending: gait quality
      [11]   actuator_effort     — ascending: effort
      [12]   phase               — ascending: locomotion phase
      [13]   phase_velocity      — ascending: phase rate
    """
    obs = t.observation
    world_obs = obs.world.observables
    feats = obs.summary.features

    avatar_xy = np.asarray(world_obs.get("avatar_xy", np.zeros(2)), dtype=np.float64)
    heading = float(world_obs.get("heading", 0.0))
    target_vec = np.asarray(
        world_obs.get("target_vector", np.zeros(2)), dtype=np.float64,
    )

    # Actions — handle both DescendingCommand and RawControlCommand
    move = float(getattr(t.action, "move_intent", 0.0))
    turn = float(getattr(t.action, "turn_intent", 0.0))
    speed = float(getattr(t.action, "speed_modulation", 0.0))

    dist = float(t.info.get("distance_to_target", 0.0))
    body_speed = feats.get("body_speed_mm_s", 0.0)
    loco_quality = feats.get("locomotion_quality", 0.0)
    effort = feats.get("actuator_effort", 0.0)
    phase = feats.get("phase", 0.0)
    phase_vel = feats.get("phase_velocity", 0.0)

    return np.array(
        [
            avatar_xy[0], avatar_xy[1],
            heading,
            target_vec[0], target_vec[1],
            move, turn, speed,
            dist,
            body_speed, loco_quality, effort,
            phase, phase_vel,
        ],
        dtype=np.float64,
    )


def extract_state_matrix(
    transitions: list[StepTransition],
) -> np.ndarray:
    """Extract T×D state matrix from a list of transitions."""
    if not transitions:
        return np.zeros((0, 14), dtype=np.float64)
    return np.array(
        [_extract_state_vector(t) for t in transitions],
        dtype=np.float64,
    )


# ── Action distribution summary ─────────────────────────────────────────


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
    t_arr = np.array(turns, dtype=np.float64)
    return {
        "action_move_mean": float(m.mean()),
        "action_move_std": float(m.std()),
        "action_turn_mean": float(t_arr.mean()),
        "action_turn_std": float(t_arr.std()),
        "action_move_range": float(m.max() - m.min()),
        "action_turn_range": float(t_arr.max() - t_arr.min()),
    }


# ── Raw alignment (no translation) ──────────────────────────────────────


def raw_latent_alignment(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Element-wise correlation of state-vector trajectories.

    For each dimension d of the state vector, computes Pearson correlation
    between the two controller runs.  Returns the mean of non-constant
    dimension correlations.  Constant dimensions are excluded.

    This measures whether controllers produce *identical* time-series
    in each state dimension.
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 4:
        return {"raw_alignment": 0.0, "raw_dims_active": 0, "raw_per_dim": []}

    Z_a = extract_state_matrix(transitions_a[:n])  # (n, D)
    Z_b = extract_state_matrix(transitions_b[:n])  # (n, D)

    correlations = []
    for d in range(Z_a.shape[1]):
        col_a, col_b = Z_a[:, d], Z_b[:, d]
        if np.std(col_a) < 1e-10 or np.std(col_b) < 1e-10:
            continue  # skip constant dimensions
        corr = float(np.corrcoef(col_a, col_b)[0, 1])
        if not np.isnan(corr):
            correlations.append(corr)

    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    return {
        "raw_alignment": mean_corr,
        "raw_dims_active": len(correlations),
        "raw_per_dim": correlations,
    }


# ── Translated alignment (linear map T: z_a → z_b) ──────────────────────


def translated_latent_alignment(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Train a linear translation T: z_a → z_b and measure R².

    Uses ordinary least squares:  T = (Z_a^T Z_a)^{-1} Z_a^T Z_b
    Then computes R² = 1 - SS_res / SS_tot on the full dataset.

    Also computes the reverse map T': z_b → z_a and reports the better
    of the two R² values (symmetric interoperability).

    R² interpretation:
      0.0 → no linear structure shared
      0.5 → half the variance in z_b is predictable from z_a
      1.0 → perfect linear translation exists
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 6:  # Need more samples than dimensions for stable regression
        return {
            "translated_alignment": 0.0,
            "translation_r2_ab": 0.0,
            "translation_r2_ba": 0.0,
            "translation_residual_norm": 0.0,
        }

    Z_a = extract_state_matrix(transitions_a[:n])  # (n, D)
    Z_b = extract_state_matrix(transitions_b[:n])  # (n, D)

    # Remove constant columns (they break regression)
    active_a = np.std(Z_a, axis=0) > 1e-10
    active_b = np.std(Z_b, axis=0) > 1e-10
    Za = Z_a[:, active_a]
    Zb = Z_b[:, active_b]

    if Za.shape[1] < 1 or Zb.shape[1] < 1:
        return {
            "translated_alignment": 0.0,
            "translation_r2_ab": 0.0,
            "translation_r2_ba": 0.0,
            "translation_residual_norm": 0.0,
        }

    # Standardize for numerical stability
    mean_a, std_a = Za.mean(axis=0), Za.std(axis=0)
    mean_b, std_b = Zb.mean(axis=0), Zb.std(axis=0)
    std_a[std_a < 1e-10] = 1.0
    std_b[std_b < 1e-10] = 1.0
    Za_norm = (Za - mean_a) / std_a
    Zb_norm = (Zb - mean_b) / std_b

    def _r2(source: np.ndarray, target: np.ndarray) -> float:
        """Fit T: source → target via least squares, return R²."""
        # Add bias column
        src_bias = np.hstack([source, np.ones((source.shape[0], 1))])
        try:
            T, residuals, rank, sv = np.linalg.lstsq(src_bias, target, rcond=None)
        except np.linalg.LinAlgError:
            return 0.0
        pred = src_bias @ T
        ss_res = float(np.sum((target - pred) ** 2))
        ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
        if ss_tot < 1e-12:
            return 0.0
        return max(0.0, 1.0 - ss_res / ss_tot)

    r2_ab = _r2(Za_norm, Zb_norm)
    r2_ba = _r2(Zb_norm, Za_norm)

    # Residual norm (for diagnostics)
    src_bias = np.hstack([Za_norm, np.ones((Za_norm.shape[0], 1))])
    try:
        T, _, _, _ = np.linalg.lstsq(src_bias, Zb_norm, rcond=None)
        pred = src_bias @ T
        res_norm = float(np.linalg.norm(Zb_norm - pred)) / n
    except np.linalg.LinAlgError:
        res_norm = 0.0

    return {
        "translated_alignment": max(r2_ab, r2_ba),
        "translation_r2_ab": r2_ab,
        "translation_r2_ba": r2_ba,
        "translation_residual_norm": res_norm,
    }


# ── Action trajectory similarity ─────────────────────────────────────────


def action_trajectory_similarity(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Compare action trajectories (move_intent, turn_intent) directly.

    Separate from latent comparison — this measures behavioral similarity
    without involving observations or rewards.
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 4:
        return {"action_corr_move": 0.0, "action_corr_turn": 0.0, "action_mae": 0.0}

    def _get_actions(trans: list[StepTransition]) -> tuple[np.ndarray, np.ndarray]:
        moves = np.array(
            [float(getattr(t.action, "move_intent", 0.0)) for t in trans[:n]],
            dtype=np.float64,
        )
        turns = np.array(
            [float(getattr(t.action, "turn_intent", 0.0)) for t in trans[:n]],
            dtype=np.float64,
        )
        return moves, turns

    m_a, t_a = _get_actions(transitions_a)
    m_b, t_b = _get_actions(transitions_b)

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        c = float(np.corrcoef(x, y)[0, 1])
        return 0.0 if np.isnan(c) else c

    return {
        "action_corr_move": _safe_corr(m_a, m_b),
        "action_corr_turn": _safe_corr(t_a, t_b),
        "action_mae": float(0.5 * np.mean(np.abs(m_a - m_b)) + 0.5 * np.mean(np.abs(t_a - t_b))),
    }


# ── Reward trajectory similarity (reported separately, NOT in composite) ──


def reward_trajectory_similarity(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Compare reward trajectories from two controller runs.

    Reported for diagnostic purposes but deliberately excluded from the
    composite interoperability score to avoid environment-imposed inflation.
    """
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


# ── Legacy single-key comparison (kept for backward compatibility) ───────


def latent_state_similarity(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
    *,
    key: str = "stability",
) -> dict[str, float]:
    """Compare a single ascending-summary feature between two runs.

    Legacy API — prefer raw_latent_alignment or translated_latent_alignment.
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


# ── Composite interoperability score ─────────────────────────────────────


def interoperability_score(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Composite interoperability score: translated_alignment only.

    Measures whether a linear translation exists between controller state
    trajectories.  Reward is excluded.  Raw alignment is reported for
    contrast but does not enter the composite.

    The composite IS the translated R² — the core question is:
    "does a structure-preserving map exist between these controllers?"
    """
    raw = raw_latent_alignment(transitions_a, transitions_b)
    translated = translated_latent_alignment(transitions_a, transitions_b)
    action_sim = action_trajectory_similarity(transitions_a, transitions_b)
    reward = reward_trajectory_similarity(transitions_a, transitions_b)

    return {
        # Core interop score — translated R² only (no reward leakage)
        "interoperability_score": translated["translated_alignment"],
        # Detailed sub-metrics
        "raw_alignment": raw["raw_alignment"],
        "raw_dims_active": raw["raw_dims_active"],
        "translated_alignment": translated["translated_alignment"],
        "translation_r2_ab": translated["translation_r2_ab"],
        "translation_r2_ba": translated["translation_r2_ba"],
        "translation_residual_norm": translated["translation_residual_norm"],
        # Action similarity (separate from latent)
        "action_corr_move": action_sim["action_corr_move"],
        "action_corr_turn": action_sim["action_corr_turn"],
        "action_mae": action_sim["action_mae"],
        # Reward (diagnostic only — not in composite)
        "reward_correlation": reward["reward_correlation"],
        "reward_mae": reward["reward_mae"],
    }
