"""Environment-as-quotient-operator metrics.

Formalizes the insight that the environment E acts as a many→one projection:

    E: Z (controller/internal space) → R (outcome space)

This means E *destroys information* but *preserves task-relevant invariants*.

Key concepts:
  - **Equivalence class**: (z_i, a_i) ~_E (z_j, a_j) iff E(z_i,a_i) ≈ E(z_j,a_j)
  - **Degeneracy**: many internal states map to same outcome (information loss)
  - **Counterfactual divergence**: E-equivalent controllers diverge under E'
  - **Environment-invariant subspace**: dimensions of z that E preserves
  - **Translation validity**: E ∘ T_ij ≈ E (translation preserves equivalence)

Reference equation:
    Valid translations T_ij must satisfy: E(T_ij(z_i)) ≈ E(z_i)
    i.e., translation must preserve environment equivalence classes.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition
from .interoperability_metrics import extract_state_matrix


# ── Degeneracy Detection ────────────────────────────────────────────────


def degeneracy_score(
    transitions_dict: dict[str, list[StepTransition]],
) -> dict[str, float | dict]:
    """Measure the degeneracy of E: how much latent variance is lost in reward.

    For a set of controllers all running in the same environment:
    1. Extract reward trajectories r_i(t) and state matrices Z_i(t)
    2. Bin timesteps by reward quantile
    3. Within each reward bin, compute variance of state vectors across controllers
    4. degeneracy_ratio = var(z | r) / var(z)
       - High ratio → controllers have different states but same outcomes
       - Low ratio → reward fully determines internal state (no degeneracy)

    Also computes:
    - reward_entropy: how spread out reward values are
    - state_entropy: how spread out state values are
    - information_loss: 1 - I(z;r)/H(z) estimated via correlation
    """
    if len(transitions_dict) < 2:
        return {"degeneracy_ratio": 0.0, "information_loss": 0.0}

    # Find common length
    min_len = min(len(t) for t in transitions_dict.values())
    if min_len < 4:
        return {"degeneracy_ratio": 0.0, "information_loss": 0.0}

    names = sorted(transitions_dict.keys())

    # Extract state matrices and reward vectors for each controller
    state_matrices: dict[str, np.ndarray] = {}
    reward_vectors: dict[str, np.ndarray] = {}
    for name in names:
        trans = transitions_dict[name][:min_len]
        state_matrices[name] = extract_state_matrix(trans)
        reward_vectors[name] = np.array(
            [t.reward for t in trans], dtype=np.float64,
        )

    # Stack all state matrices: (n_controllers × T, D)
    all_states = np.vstack([state_matrices[n] for n in names])
    all_rewards = np.concatenate([reward_vectors[n] for n in names])

    # Total state variance (across all controllers and time)
    total_state_var = float(np.var(all_states, axis=0).sum())
    if total_state_var < 1e-12:
        return {"degeneracy_ratio": 0.0, "information_loss": 0.0}

    # Bin rewards into quantiles
    n_bins = min(8, len(all_rewards) // 4)
    if n_bins < 2:
        return {"degeneracy_ratio": 0.0, "information_loss": 0.0}

    try:
        bin_edges = np.quantile(all_rewards, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(all_rewards, bin_edges[1:-1])
    except (ValueError, IndexError):
        return {"degeneracy_ratio": 0.0, "information_loss": 0.0}

    # Within-bin state variance (conditional variance)
    conditional_var = 0.0
    total_count = 0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() < 2:
            continue
        bin_states = all_states[mask]
        conditional_var += float(np.var(bin_states, axis=0).sum()) * mask.sum()
        total_count += mask.sum()

    if total_count == 0:
        return {"degeneracy_ratio": 0.0, "information_loss": 0.0}

    conditional_var /= total_count
    degeneracy_ratio = conditional_var / total_state_var

    # Reward variance (how spread are outcomes?)
    reward_var = float(np.var(all_rewards))

    # Per-controller reward variance (should be low if E collapses)
    pairwise_reward_corrs = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            r1, r2 = reward_vectors[n1], reward_vectors[n2]
            if np.std(r1) > 1e-10 and np.std(r2) > 1e-10:
                c = float(np.corrcoef(r1, r2)[0, 1])
                if not np.isnan(c):
                    pairwise_reward_corrs.append(c)

    mean_reward_corr = float(np.mean(pairwise_reward_corrs)) if pairwise_reward_corrs else 0.0

    # Information loss: how much of state variance is NOT predictable from reward
    # Use R² of reward predicting each state dimension
    explained_var = 0.0
    for d in range(all_states.shape[1]):
        col = all_states[:, d]
        if np.std(col) < 1e-10:
            continue
        if np.std(all_rewards) < 1e-10:
            continue
        corr = float(np.corrcoef(all_rewards, col)[0, 1])
        if not np.isnan(corr):
            explained_var += corr ** 2 * np.var(col)

    information_loss = 1.0 - (explained_var / total_state_var)
    information_loss = max(0.0, min(1.0, information_loss))

    return {
        "degeneracy_ratio": float(degeneracy_ratio),
        "information_loss": float(information_loss),
        "reward_variance": float(reward_var),
        "total_state_variance": float(total_state_var),
        "conditional_state_variance": float(conditional_var),
        "mean_pairwise_reward_corr": float(mean_reward_corr),
        "n_controllers": len(names),
        "n_timesteps": min_len,
    }


# ── Environment-Invariant Dimension Analysis ─────────────────────────────


def environment_invariant_dimensions(
    transitions_dict: dict[str, list[StepTransition]],
) -> dict[str, float | list]:
    """Identify which state dimensions are preserved vs destroyed by E.

    For each dimension d of the state vector:
    - Compute correlation with reward across all controllers
    - High |corr(z_d, r)| → dimension is E-invariant (preserved in outcome)
    - Low |corr(z_d, r)| → dimension is E-destroyed (not reflected in outcome)

    Returns:
    - per_dim_reward_corr: correlation of each dimension with reward
    - invariant_dims: indices of dimensions where |corr| > 0.3
    - destroyed_dims: indices of dimensions where |corr| < 0.1
    - invariant_fraction: what fraction of state space is E-invariant
    """
    min_len = min(len(t) for t in transitions_dict.values()) if transitions_dict else 0
    if min_len < 4 or not transitions_dict:
        return {
            "per_dim_reward_corr": [],
            "invariant_dims": [],
            "destroyed_dims": [],
            "invariant_fraction": 0.0,
        }

    names = sorted(transitions_dict.keys())
    all_states = np.vstack([
        extract_state_matrix(transitions_dict[n][:min_len])
        for n in names
    ])
    all_rewards = np.concatenate([
        np.array([t.reward for t in transitions_dict[n][:min_len]], dtype=np.float64)
        for n in names
    ])

    dim_names = [
        "avatar_x", "avatar_y", "heading", "target_x", "target_y",
        "move_intent", "turn_intent", "speed_mod",
        "distance", "body_speed", "loco_quality", "effort", "phase", "phase_vel",
    ]

    per_dim_corr = []
    invariant_dims = []
    destroyed_dims = []

    for d in range(all_states.shape[1]):
        col = all_states[:, d]
        if np.std(col) < 1e-10 or np.std(all_rewards) < 1e-10:
            per_dim_corr.append(0.0)
            destroyed_dims.append(d)
            continue
        corr = float(np.corrcoef(all_rewards, col)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        per_dim_corr.append(corr)
        if abs(corr) > 0.3:
            invariant_dims.append(d)
        elif abs(corr) < 0.1:
            destroyed_dims.append(d)

    active_dims = sum(1 for c in per_dim_corr if abs(c) > 0.01)
    invariant_fraction = len(invariant_dims) / max(active_dims, 1)

    return {
        "per_dim_reward_corr": per_dim_corr,
        "dim_names": dim_names[:len(per_dim_corr)],
        "invariant_dims": invariant_dims,
        "destroyed_dims": destroyed_dims,
        "invariant_fraction": float(invariant_fraction),
        "n_invariant": len(invariant_dims),
        "n_destroyed": len(destroyed_dims),
    }


# ── Equivalence Class Analysis ───────────────────────────────────────────


def equivalence_class_size(
    transitions_dict: dict[str, list[StepTransition]],
    reward_tolerance: float = 0.15,
) -> dict[str, float | list]:
    """Measure equivalence class size: how many controllers produce ~same outcome.

    Two controllers are in the same equivalence class under E if:
        |r_i(t) - r_j(t)| < tolerance  for most timesteps

    Returns:
    - equivalence_pairs: list of (name_i, name_j, overlap_fraction)
    - mean_class_size: average number of controllers in each equivalence class
    - max_class_size: largest equivalence class
    """
    min_len = min(len(t) for t in transitions_dict.values()) if transitions_dict else 0
    if min_len < 2 or len(transitions_dict) < 2:
        return {"equivalence_pairs": [], "mean_class_size": 1.0, "max_class_size": 1}

    names = sorted(transitions_dict.keys())
    rewards = {
        n: np.array([t.reward for t in transitions_dict[n][:min_len]], dtype=np.float64)
        for n in names
    }

    # Normalize rewards for tolerance comparison
    all_r = np.concatenate(list(rewards.values()))
    r_scale = float(np.std(all_r))
    if r_scale < 1e-10:
        r_scale = 1.0
    tol = reward_tolerance * r_scale

    pairs = []
    adjacency: dict[str, set[str]] = {n: {n} for n in names}

    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            diff = np.abs(rewards[n1] - rewards[n2])
            overlap = float(np.mean(diff < tol))
            pairs.append((n1, n2, overlap))
            if overlap > 0.5:  # More than half timesteps are equivalent
                adjacency[n1].add(n2)
                adjacency[n2].add(n1)

    # Compute connected components (equivalence classes)
    visited: set[str] = set()
    classes: list[set[str]] = []
    for n in names:
        if n not in visited:
            # BFS
            component: set[str] = set()
            queue = [n]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            classes.append(component)

    class_sizes = [len(c) for c in classes]

    return {
        "equivalence_pairs": [(n1, n2, round(ov, 3)) for n1, n2, ov in pairs],
        "equivalence_classes": [sorted(c) for c in classes],
        "class_sizes": class_sizes,
        "mean_class_size": float(np.mean(class_sizes)),
        "max_class_size": max(class_sizes),
        "n_classes": len(classes),
    }


# ── Counterfactual Divergence ────────────────────────────────────────────


def counterfactual_divergence(
    transitions_E1: dict[str, list[StepTransition]],
    transitions_E2: dict[str, list[StepTransition]],
) -> dict[str, float | list]:
    """Measure whether E-equivalent controllers diverge under E'.

    If two controllers produce similar outcomes in E1 but different outcomes
    in E2, this proves that their equivalence is environment-specific:
        (z_i, a_i) ~_{E1} (z_j, a_j) but (z_i, a_i) ≁_{E2} (z_j, a_j)

    This is the key test for "environment defines equivalence classes."

    Returns:
    - per_pair_divergence: for each pair, how much did reward correlation change
    - mean_divergence: average |corr_E1 - corr_E2| across pairs
    - max_divergence: strongest divergence (most environment-specific pair)
    - divergent_pairs: pairs where E1-equivalent but E2-divergent
    """
    common_names = sorted(set(transitions_E1.keys()) & set(transitions_E2.keys()))
    if len(common_names) < 2:
        return {"mean_divergence": 0.0, "max_divergence": 0.0, "divergent_pairs": []}

    min_len_E1 = min(len(transitions_E1[n]) for n in common_names)
    min_len_E2 = min(len(transitions_E2[n]) for n in common_names)

    def _reward_corr(trans_dict: dict, names: list[str], n: int) -> dict[tuple[str, str], float]:
        rewards = {
            name: np.array([t.reward for t in trans_dict[name][:n]], dtype=np.float64)
            for name in names
        }
        corrs = {}
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                r1, r2 = rewards[n1], rewards[n2]
                if np.std(r1) < 1e-10 or np.std(r2) < 1e-10:
                    corrs[(n1, n2)] = 0.0
                else:
                    c = float(np.corrcoef(r1, r2)[0, 1])
                    corrs[(n1, n2)] = 0.0 if np.isnan(c) else c
        return corrs

    def _state_r2(trans_dict: dict, names: list[str], n: int) -> dict[tuple[str, str], float]:
        """Pairwise translation R² in a given environment."""
        matrices = {
            name: extract_state_matrix(trans_dict[name][:n])
            for name in names
        }
        r2s = {}
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                Za, Zb = matrices[n1], matrices[n2]
                # Quick OLS R²
                active_a = np.std(Za, axis=0) > 1e-10
                active_b = np.std(Zb, axis=0) > 1e-10
                Za_f = Za[:, active_a]
                Zb_f = Zb[:, active_b]
                if Za_f.shape[1] < 1 or Zb_f.shape[1] < 1:
                    r2s[(n1, n2)] = 0.0
                    continue
                # Standardize
                Za_n = (Za_f - Za_f.mean(0)) / np.maximum(Za_f.std(0), 1e-10)
                Zb_n = (Zb_f - Zb_f.mean(0)) / np.maximum(Zb_f.std(0), 1e-10)
                src = np.hstack([Za_n, np.ones((Za_n.shape[0], 1))])
                try:
                    T, _, _, _ = np.linalg.lstsq(src, Zb_n, rcond=None)
                    pred = src @ T
                    ss_res = float(np.sum((Zb_n - pred) ** 2))
                    ss_tot = float(np.sum((Zb_n - Zb_n.mean(0)) ** 2))
                    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                except np.linalg.LinAlgError:
                    r2 = 0.0
                r2s[(n1, n2)] = r2
        return r2s

    corrs_E1 = _reward_corr(transitions_E1, common_names, min_len_E1)
    corrs_E2 = _reward_corr(transitions_E2, common_names, min_len_E2)
    r2_E1 = _state_r2(transitions_E1, common_names, min_len_E1)
    r2_E2 = _state_r2(transitions_E2, common_names, min_len_E2)

    per_pair = []
    divergent_pairs = []
    for pair in corrs_E1:
        rc1 = corrs_E1[pair]
        rc2 = corrs_E2.get(pair, 0.0)
        r2_1 = r2_E1.get(pair, 0.0)
        r2_2 = r2_E2.get(pair, 0.0)
        reward_div = abs(rc1 - rc2)
        state_div = abs(r2_1 - r2_2)
        per_pair.append({
            "pair": f"{pair[0]} vs {pair[1]}",
            "reward_corr_E1": round(rc1, 3),
            "reward_corr_E2": round(rc2, 3),
            "reward_divergence": round(reward_div, 3),
            "translation_r2_E1": round(r2_1, 3),
            "translation_r2_E2": round(r2_2, 3),
            "translation_divergence": round(state_div, 3),
        })
        # E1-equivalent (high corr) but E2-divergent (lower corr)
        if rc1 > 0.7 and reward_div > 0.15:
            divergent_pairs.append(f"{pair[0]} vs {pair[1]}")

    reward_divs = [p["reward_divergence"] for p in per_pair]
    state_divs = [p["translation_divergence"] for p in per_pair]

    return {
        "per_pair": per_pair,
        "mean_reward_divergence": float(np.mean(reward_divs)) if reward_divs else 0.0,
        "max_reward_divergence": float(np.max(reward_divs)) if reward_divs else 0.0,
        "mean_translation_divergence": float(np.mean(state_divs)) if state_divs else 0.0,
        "max_translation_divergence": float(np.max(state_divs)) if state_divs else 0.0,
        "divergent_pairs": divergent_pairs,
        "n_divergent": len(divergent_pairs),
        "environment_specificity": float(np.mean(reward_divs)) if reward_divs else 0.0,
    }


# ── Translation Preserves Environment (E ∘ T ≈ E) ───────────────────────


def translation_preserves_environment(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float]:
    """Test whether the learned translation T preserves environment outcomes.

    The key equation: E ∘ T_ij ≈ E

    1. Learn T: z_a → z_b (linear OLS)
    2. The "environment outcome" of z is approximated by reward r(t)
    3. Check: does predicting z_b from z_a via T preserve the reward structure?
       Specifically: corr(r_a, r_predicted_from_T) vs corr(r_a, r_b)

    If T preserves environment:
    - r(T(z_a)) ≈ r(z_a) — translated state produces same outcome
    - The translation stays within the equivalence class

    Returns:
    - environment_preservation_r2: R² of T(z_a) predicting r_b
    - naive_reward_corr: corr(r_a, r_b) for comparison
    - translation_validity: whether T preserves E (preservation > 0.5 × naive)
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 6:
        return {
            "environment_preservation_r2": 0.0,
            "naive_reward_corr": 0.0,
            "translation_validity": False,
        }

    Za = extract_state_matrix(transitions_a[:n])
    Zb = extract_state_matrix(transitions_b[:n])
    r_a = np.array([t.reward for t in transitions_a[:n]], dtype=np.float64)
    r_b = np.array([t.reward for t in transitions_b[:n]], dtype=np.float64)

    # Remove constant columns
    active_a = np.std(Za, axis=0) > 1e-10
    active_b = np.std(Zb, axis=0) > 1e-10
    Za_f = Za[:, active_a]
    Zb_f = Zb[:, active_b]

    if Za_f.shape[1] < 1 or Zb_f.shape[1] < 1:
        return {
            "environment_preservation_r2": 0.0,
            "naive_reward_corr": 0.0,
            "translation_validity": False,
        }

    # Standardize
    Za_n = (Za_f - Za_f.mean(0)) / np.maximum(Za_f.std(0), 1e-10)
    Zb_n = (Zb_f - Zb_f.mean(0)) / np.maximum(Zb_f.std(0), 1e-10)

    # Learn T: z_a → z_b
    src = np.hstack([Za_n, np.ones((Za_n.shape[0], 1))])
    try:
        T, _, _, _ = np.linalg.lstsq(src, Zb_n, rcond=None)
    except np.linalg.LinAlgError:
        return {
            "environment_preservation_r2": 0.0,
            "naive_reward_corr": 0.0,
            "translation_validity": False,
        }

    Zb_pred = src @ T  # T(z_a) — the translated representation

    # Now: does T(z_a) predict r_b?
    # Train a simple linear model from Zb_pred → r_b
    src_pred = np.hstack([Zb_pred, np.ones((Zb_pred.shape[0], 1))])
    try:
        W, _, _, _ = np.linalg.lstsq(src_pred, r_b, rcond=None)
        r_b_hat = src_pred @ W
        ss_res = float(np.sum((r_b - r_b_hat) ** 2))
        ss_tot = float(np.sum((r_b - r_b.mean()) ** 2))
        preservation_r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    except np.linalg.LinAlgError:
        preservation_r2 = 0.0

    # Baseline: how well does z_b itself predict r_b?
    src_b = np.hstack([Zb_n, np.ones((Zb_n.shape[0], 1))])
    try:
        W_b, _, _, _ = np.linalg.lstsq(src_b, r_b, rcond=None)
        r_b_hat_b = src_b @ W_b
        ss_res_b = float(np.sum((r_b - r_b_hat_b) ** 2))
        baseline_r2 = max(0.0, 1.0 - ss_res_b / ss_tot) if ss_tot > 1e-12 else 0.0
    except np.linalg.LinAlgError:
        baseline_r2 = 0.0

    # Naive reward correlation
    if np.std(r_a) > 1e-10 and np.std(r_b) > 1e-10:
        naive_corr = float(np.corrcoef(r_a, r_b)[0, 1])
        if np.isnan(naive_corr):
            naive_corr = 0.0
    else:
        naive_corr = 0.0

    # Translation is valid if it preserves at least 50% of the environment
    # structure that z_b itself captures
    translation_valid = preservation_r2 > 0.5 * baseline_r2 if baseline_r2 > 0.1 else False

    return {
        "environment_preservation_r2": float(preservation_r2),
        "baseline_prediction_r2": float(baseline_r2),
        "preservation_ratio": float(preservation_r2 / baseline_r2) if baseline_r2 > 1e-6 else 0.0,
        "naive_reward_corr": float(naive_corr),
        "translation_validity": bool(translation_valid),
    }


# ── Composite Quotient Analysis ──────────────────────────────────────────


def full_quotient_analysis(
    transitions_dict: dict[str, list[StepTransition]],
) -> dict[str, dict]:
    """Run all quotient-operator analyses on a set of controller trajectories.

    Returns a dict with keys: degeneracy, invariant_dims, equivalence_classes,
    and per_pair_translation_validity.
    """
    result: dict[str, dict] = {}

    # 1. Degeneracy
    result["degeneracy"] = degeneracy_score(transitions_dict)

    # 2. Invariant dimensions
    result["invariant_dimensions"] = environment_invariant_dimensions(transitions_dict)

    # 3. Equivalence classes
    result["equivalence_classes"] = equivalence_class_size(transitions_dict)

    # 4. Per-pair translation validity (E ∘ T ≈ E)
    names = sorted(transitions_dict.keys())
    pair_validity = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            pv = translation_preserves_environment(
                transitions_dict[n1], transitions_dict[n2],
            )
            pv["pair"] = f"{n1} vs {n2}"
            pair_validity.append(pv)

    result["translation_validity"] = {
        "per_pair": pair_validity,
        "mean_preservation_r2": float(np.mean([
            p["environment_preservation_r2"] for p in pair_validity
        ])) if pair_validity else 0.0,
        "valid_translations": sum(
            1 for p in pair_validity if p["translation_validity"]
        ),
        "total_pairs": len(pair_validity),
    }

    return result
