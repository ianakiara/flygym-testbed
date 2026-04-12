"""Stage 7 v3 — Publishable protocol metrics for controller translation.

Five experiments that elevate the translation finding from "strong signal"
to "statistically rigorous, falsifiable claim":

1. **Cross-validation**: k-fold CV on translation R² (train vs test)
2. **Cross-world transfer**: train T in world A, test in world B
3. **Nonlinear vs linear**: MLP vs OLS — detect manifold complexity
4. **Noise robustness**: inject sensor/action noise, measure R² degradation
5. **Latent dimensionality sweep**: vary state-vector size, find saturation

Also provides statistical hygiene utilities: exclude trivial pairs,
report median + IQR alongside mean, and full distribution stats.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition
from .interoperability_metrics import extract_state_matrix


# ── Helpers ──────────────────────────────────────────────────────────────


def _ols_r2(source: np.ndarray, target: np.ndarray) -> float:
    """Fit linear T: source → target via OLS, return R²."""
    if source.shape[0] < source.shape[1] + 2 or source.shape[1] < 1 or target.shape[1] < 1:
        return 0.0
    src = np.hstack([source, np.ones((source.shape[0], 1))])
    try:
        T, _, _, _ = np.linalg.lstsq(src, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0
    pred = src @ T
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean(axis=0)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _standardize(Z: np.ndarray) -> np.ndarray:
    """Column-standardize, removing constant columns."""
    std = Z.std(axis=0)
    active = std > 1e-10
    if not active.any():
        return np.zeros((Z.shape[0], 0))
    Zf = Z[:, active]
    return (Zf - Zf.mean(axis=0)) / np.maximum(Zf.std(axis=0), 1e-10)


def _is_trivial_pair(name_a: str, name_b: str) -> bool:
    """Check whether a pair is trivially identical (both produce zero actions)."""
    trivial = {"reflex_only", "raw_control"}
    return name_a in trivial and name_b in trivial


def distribution_stats(values: list[float]) -> dict[str, float]:
    """Compute mean, median, std, IQR, min, max for a list of values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    arr = np.array(values)
    q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "iqr": q75 - q25,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(values),
    }


# ── Experiment 1: Cross-Validation ───────────────────────────────────────


def cross_validated_translation_r2(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
    *,
    k_folds: int = 5,
) -> dict[str, float]:
    """K-fold cross-validation on translation R².

    Splits timesteps into k folds, trains T on k-1 folds, tests on held-out
    fold.  Reports train R², test R², and the gap (overfitting indicator).

    This is the critical experiment for confirming the R²=0.888 finding
    is not an artifact of overfitting (64 samples, 14 dimensions).
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < k_folds * 3:  # Need at least 3 samples per fold
        return {"train_r2": 0.0, "test_r2": 0.0, "gap": 0.0, "k_folds": k_folds}

    # NOTE: Standardisation uses full-dataset statistics before the fold split.
    # This is a common simplification — the test-fold mean/std leak is minor
    # for linear regression (OLS adjusts via the bias term).  A stricter
    # implementation would re-standardise per fold using only train statistics.
    Za = _standardize(extract_state_matrix(transitions_a[:n]))
    Zb = _standardize(extract_state_matrix(transitions_b[:n]))

    if Za.shape[1] < 1 or Zb.shape[1] < 1:
        return {"train_r2": 0.0, "test_r2": 0.0, "gap": 0.0, "k_folds": k_folds}

    indices = np.arange(n)
    np.random.seed(42)  # Reproducible folds
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    train_r2s = []
    test_r2s = []

    for fold_idx in range(k_folds):
        test_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_idx])

        Za_train, Za_test = Za[train_idx], Za[test_idx]
        Zb_train, Zb_test = Zb[train_idx], Zb[test_idx]

        # Train: fit T on training data
        src_train = np.hstack([Za_train, np.ones((Za_train.shape[0], 1))])
        try:
            T, _, _, _ = np.linalg.lstsq(src_train, Zb_train, rcond=None)
        except np.linalg.LinAlgError:
            train_r2s.append(0.0)
            test_r2s.append(0.0)
            continue

        # Train R²
        pred_train = src_train @ T
        ss_res_train = float(np.sum((Zb_train - pred_train) ** 2))
        ss_tot_train = float(np.sum((Zb_train - Zb_train.mean(axis=0)) ** 2))
        tr_r2 = max(0.0, 1.0 - ss_res_train / ss_tot_train) if ss_tot_train > 1e-12 else 0.0

        # Test R²
        src_test = np.hstack([Za_test, np.ones((Za_test.shape[0], 1))])
        pred_test = src_test @ T
        ss_res_test = float(np.sum((Zb_test - pred_test) ** 2))
        ss_tot_test = float(np.sum((Zb_test - Zb_test.mean(axis=0)) ** 2))
        te_r2 = max(0.0, 1.0 - ss_res_test / ss_tot_test) if ss_tot_test > 1e-12 else 0.0

        train_r2s.append(tr_r2)
        test_r2s.append(te_r2)

    mean_train = float(np.mean(train_r2s))
    mean_test = float(np.mean(test_r2s))

    return {
        "train_r2": mean_train,
        "test_r2": mean_test,
        "gap": mean_train - mean_test,
        "overfitting_ratio": (mean_train - mean_test) / mean_train if mean_train > 1e-6 else 0.0,
        "fold_test_r2s": [round(v, 4) for v in test_r2s],
        "fold_train_r2s": [round(v, 4) for v in train_r2s],
        "k_folds": k_folds,
    }


# ── Experiment 2: Cross-World Transfer ───────────────────────────────────


def cross_world_translation_r2(
    transitions_world_a: dict[str, list[StepTransition]],
    transitions_world_b: dict[str, list[StepTransition]],
) -> dict[str, float | list]:
    """Train translation T in world A, test in world B.

    If translation generalizes across worlds, the shared structure is
    deeper than environment — it lives in the controllers themselves.

    If it fails, structure is environment-mediated (quotient operator only).

    IMPORTANT: Uses consistent column indices across both worlds to avoid
    the bug where _standardize removes different constant columns per world,
    causing T to map between misaligned features.
    """
    common = sorted(set(transitions_world_a.keys()) & set(transitions_world_b.keys()))
    if len(common) < 2:
        return {"mean_within_r2": 0.0, "mean_transfer_r2": 0.0, "transfer_ratio": 0.0}

    per_pair = []
    within_r2s = []
    transfer_r2s = []

    for i, n1 in enumerate(common):
        for n2 in common[i + 1:]:
            # Extract RAW state matrices (no standardization yet)
            n_a = min(len(transitions_world_a[n1]), len(transitions_world_a[n2]))
            n_b = min(len(transitions_world_b[n1]), len(transitions_world_b[n2]))

            Za_raw_a = extract_state_matrix(transitions_world_a[n1][:n_a])
            Zb_raw_a = extract_state_matrix(transitions_world_a[n2][:n_a])
            Za_raw_b = extract_state_matrix(transitions_world_b[n1][:n_b])
            Zb_raw_b = extract_state_matrix(transitions_world_b[n2][:n_b])

            # Find columns that are non-constant in BOTH worlds for BOTH controllers
            # This ensures T maps between the SAME features in both worlds
            active_cols = (
                (np.std(Za_raw_a, axis=0) > 1e-10) &
                (np.std(Zb_raw_a, axis=0) > 1e-10) &
                (np.std(Za_raw_b, axis=0) > 1e-10) &
                (np.std(Zb_raw_b, axis=0) > 1e-10)
            )

            if active_cols.sum() < 1:
                continue

            # Filter to consistent columns, then standardize each world separately
            Za_a = Za_raw_a[:, active_cols]
            Zb_a = Zb_raw_a[:, active_cols]
            Za_b = Za_raw_b[:, active_cols]
            Zb_b = Zb_raw_b[:, active_cols]

            # Standardize world A (train)
            Za_a_mean, Za_a_std = Za_a.mean(0), np.maximum(Za_a.std(0), 1e-10)
            Zb_a_mean, Zb_a_std = Zb_a.mean(0), np.maximum(Zb_a.std(0), 1e-10)
            Za_a_n = (Za_a - Za_a_mean) / Za_a_std
            Zb_a_n = (Zb_a - Zb_a_mean) / Zb_a_std

            # Standardize world B (test) using its OWN statistics
            Za_b_mean, Za_b_std = Za_b.mean(0), np.maximum(Za_b.std(0), 1e-10)
            Zb_b_mean, Zb_b_std = Zb_b.mean(0), np.maximum(Zb_b.std(0), 1e-10)
            Za_b_n = (Za_b - Za_b_mean) / Za_b_std
            Zb_b_n = (Zb_b - Zb_b_mean) / Zb_b_std

            # Within-world R² (baseline — world A only)
            within_r2 = _ols_r2(Za_a_n, Zb_a_n)

            # Train T on world A
            src_a = np.hstack([Za_a_n, np.ones((Za_a_n.shape[0], 1))])
            try:
                T, _, _, _ = np.linalg.lstsq(src_a, Zb_a_n, rcond=None)
            except np.linalg.LinAlgError:
                continue

            # Apply T to world B data (same column structure!)
            src_b = np.hstack([Za_b_n, np.ones((Za_b_n.shape[0], 1))])
            pred_b = src_b @ T
            ss_res = float(np.sum((Zb_b_n - pred_b) ** 2))
            ss_tot = float(np.sum((Zb_b_n - Zb_b_n.mean(axis=0)) ** 2))
            transfer_r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

            # Also compute world B within-world R² for comparison
            within_b_r2 = _ols_r2(Za_b_n, Zb_b_n)

            trivial = _is_trivial_pair(n1, n2)
            per_pair.append({
                "pair": f"{n1} vs {n2}",
                "within_world_a_r2": round(within_r2, 4),
                "within_world_b_r2": round(within_b_r2, 4),
                "within_world_r2": round(within_r2, 4),
                "transfer_r2": round(transfer_r2, 4),
                "transfer_ratio": round(transfer_r2 / within_r2, 4) if within_r2 > 1e-6 else 0.0,
                "n_shared_dims": int(active_cols.sum()),
                "trivial": trivial,
            })
            within_r2s.append(within_r2)
            transfer_r2s.append(transfer_r2)

    # Separate stats for non-trivial pairs
    nt_within = [p["within_world_r2"] for p in per_pair if not p["trivial"]]
    nt_transfer = [p["transfer_r2"] for p in per_pair if not p["trivial"]]

    return {
        "per_pair": per_pair,
        "mean_within_r2": float(np.mean(within_r2s)) if within_r2s else 0.0,
        "mean_transfer_r2": float(np.mean(transfer_r2s)) if transfer_r2s else 0.0,
        "transfer_ratio": float(np.mean(transfer_r2s)) / float(np.mean(within_r2s)) if within_r2s and np.mean(within_r2s) > 1e-6 else 0.0,
        "nontrivial_within_r2": float(np.mean(nt_within)) if nt_within else 0.0,
        "nontrivial_transfer_r2": float(np.mean(nt_transfer)) if nt_transfer else 0.0,
        "nontrivial_transfer_ratio": float(np.mean(nt_transfer)) / float(np.mean(nt_within)) if nt_within and np.mean(nt_within) > 1e-6 else 0.0,
        "transfer_stats": distribution_stats(transfer_r2s),
    }


# ── Experiment 3: Nonlinear vs Linear ────────────────────────────────────


def nonlinear_vs_linear_translation(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
    *,
    hidden_size: int = 32,
    n_epochs: int = 200,
    learning_rate: float = 0.01,
) -> dict[str, float]:
    """Compare linear OLS vs nonlinear MLP translation.

    If nonlinear >> linear: the shared manifold has complex curvature.
    If nonlinear ≈ linear: structure is genuinely linear (stronger finding).

    Uses a simple 1-hidden-layer MLP trained with gradient descent (no
    external dependencies beyond numpy).
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 10:
        return {"linear_r2": 0.0, "nonlinear_r2": 0.0, "complexity_gap": 0.0}

    Za = _standardize(extract_state_matrix(transitions_a[:n]))
    Zb = _standardize(extract_state_matrix(transitions_b[:n]))

    if Za.shape[1] < 1 or Zb.shape[1] < 1:
        return {"linear_r2": 0.0, "nonlinear_r2": 0.0, "complexity_gap": 0.0}

    # Linear baseline
    linear_r2 = _ols_r2(Za, Zb)

    # Simple MLP: input → hidden (ReLU) → output
    d_in, d_out = Za.shape[1], Zb.shape[1]
    h = min(hidden_size, d_in * 2)

    np.random.seed(42)
    W1 = np.random.randn(d_in, h) * 0.1
    b1 = np.zeros(h)
    W2 = np.random.randn(h, d_out) * 0.1
    b2 = np.zeros(d_out)

    for epoch in range(n_epochs):
        # Forward pass
        hidden = Za @ W1 + b1
        hidden_relu = np.maximum(hidden, 0)  # ReLU
        pred = hidden_relu @ W2 + b2

        # Loss (MSE)
        residual = pred - Zb

        # Backward pass
        d_out_layer = residual / n  # (n, d_out)
        dW2 = hidden_relu.T @ d_out_layer
        db2 = d_out_layer.sum(axis=0)

        d_hidden = d_out_layer @ W2.T
        d_hidden[hidden <= 0] = 0  # ReLU gradient
        dW1 = Za.T @ d_hidden
        db1 = d_hidden.sum(axis=0)

        # Gradient descent with simple clipping
        grad_norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in [dW1, db1, dW2, db2]))
        if grad_norm > 10.0:
            scale = 10.0 / grad_norm
            dW1, db1, dW2, db2 = dW1 * scale, db1 * scale, dW2 * scale, db2 * scale

        lr = learning_rate * (1.0 - epoch / n_epochs)  # Linear decay
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Final prediction R²
    hidden = Za @ W1 + b1
    hidden_relu = np.maximum(hidden, 0)
    pred = hidden_relu @ W2 + b2
    ss_res = float(np.sum((Zb - pred) ** 2))
    ss_tot = float(np.sum((Zb - Zb.mean(axis=0)) ** 2))
    nonlinear_r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "linear_r2": linear_r2,
        "nonlinear_r2": nonlinear_r2,
        "complexity_gap": nonlinear_r2 - linear_r2,
        "structure_is_linear": nonlinear_r2 - linear_r2 < 0.05,
        "hidden_size": h,
        "n_epochs": n_epochs,
    }


# ── Experiment 4: Noise Robustness ───────────────────────────────────────


def noise_robustness_sweep(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
    *,
    noise_levels: tuple[float, ...] = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0),
) -> dict[str, float | list]:
    """Measure translation R² degradation under increasing noise.

    Adds Gaussian noise to state vectors before fitting translation.
    A robust finding should degrade gracefully (not cliff-drop).

    noise_level = ratio of noise std to data std.
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 10:
        return {"noise_levels": list(noise_levels), "r2_values": [0.0] * len(noise_levels)}

    Za_raw = extract_state_matrix(transitions_a[:n])
    Zb_raw = extract_state_matrix(transitions_b[:n])

    Za = _standardize(Za_raw)
    Zb = _standardize(Zb_raw)

    if Za.shape[1] < 1 or Zb.shape[1] < 1:
        return {"noise_levels": list(noise_levels), "r2_values": [0.0] * len(noise_levels)}

    rng = np.random.RandomState(42)
    r2_values = []

    for noise_level in noise_levels:
        if noise_level == 0.0:
            r2 = _ols_r2(Za, Zb)
        else:
            # Add noise proportional to data std (which is ~1 after standardization)
            Za_noisy = Za + rng.randn(*Za.shape) * noise_level
            Zb_noisy = Zb + rng.randn(*Zb.shape) * noise_level
            r2 = _ols_r2(Za_noisy, Zb_noisy)
        r2_values.append(r2)

    # Compute degradation rate (slope of R² vs noise)
    if len(r2_values) >= 2 and r2_values[0] > 1e-6:
        # Normalized degradation: how much R² drops per unit noise
        degradation_rate = (r2_values[0] - r2_values[-1]) / (noise_levels[-1] - noise_levels[0])
        half_life = None
        for i, r2 in enumerate(r2_values):
            if r2 < r2_values[0] * 0.5:
                half_life = noise_levels[i]
                break
    else:
        degradation_rate = 0.0
        half_life = None

    # Find R² at moderate noise (1.0× — equal noise to signal)
    moderate_r2 = 0.0
    for nl, r2 in zip(noise_levels, r2_values):
        if abs(nl - 1.0) < 0.01:
            moderate_r2 = r2
            break

    # Graceful degradation: no cliff-drop between adjacent noise levels
    cliff_drop = False
    for i in range(1, len(r2_values)):
        if r2_values[i - 1] - r2_values[i] > 0.4:  # >40% drop in one step = cliff
            cliff_drop = True
            break

    return {
        "noise_levels": list(noise_levels),
        "r2_values": [round(v, 4) for v in r2_values],
        "clean_r2": r2_values[0] if r2_values else 0.0,
        "moderate_r2": moderate_r2,  # R² at noise=1.0 (equal noise to signal)
        "noisiest_r2": r2_values[-1] if r2_values else 0.0,
        "degradation_rate": round(degradation_rate, 4),
        "half_life_noise": half_life,
        "robust": r2_values[-1] > 0.3 if r2_values else False,  # Still translatable at max noise
        "moderate_robust": moderate_r2 > 0.3,  # Translatable at 1× noise
        "graceful": not cliff_drop,  # No cliff-drop in degradation
    }


# ── Experiment 5: Latent Dimensionality Sweep ────────────────────────────


_DIM_SUBSETS: dict[int, list[int]] = {
    # 5D: position + heading + distance + move_intent
    5: [0, 1, 2, 5, 8],
    # 8D: above + target_vector + turn_intent
    8: [0, 1, 2, 3, 4, 5, 6, 8],
    # 10D: above + speed_mod + body_speed
    10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # 14D: all dimensions (full state vector)
    14: list(range(14)),
}


def dimensionality_sweep(
    transitions_a: list[StepTransition],
    transitions_b: list[StepTransition],
) -> dict[str, float | list]:
    """Measure translation R² at different state-vector dimensionalities.

    Tests 5D, 8D, 10D, 14D subsets to find where translation saturates.
    If R² saturates early → only a few dimensions carry the shared structure.
    If R² keeps climbing → high-dimensional structure is genuinely rich.
    """
    n = min(len(transitions_a), len(transitions_b))
    if n < 10:
        return {"dims": [], "r2_values": []}

    Za_full = extract_state_matrix(transitions_a[:n])
    Zb_full = extract_state_matrix(transitions_b[:n])

    results = []
    for n_dims, col_indices in sorted(_DIM_SUBSETS.items()):
        # Filter to valid indices
        valid_idx = [i for i in col_indices if i < Za_full.shape[1]]
        if not valid_idx:
            results.append({"dims": n_dims, "r2": 0.0})
            continue

        Za_sub = _standardize(Za_full[:, valid_idx])
        Zb_sub = _standardize(Zb_full[:, valid_idx])

        if Za_sub.shape[1] < 1 or Zb_sub.shape[1] < 1:
            results.append({"dims": n_dims, "r2": 0.0})
            continue

        r2 = _ols_r2(Za_sub, Zb_sub)
        results.append({"dims": n_dims, "r2": round(r2, 4)})

    dims = [r["dims"] for r in results]
    r2s = [r["r2"] for r in results]

    # Saturation analysis
    saturation_dim = None
    if len(r2s) >= 2:
        for i in range(1, len(r2s)):
            if r2s[i] - r2s[i - 1] < 0.02:  # <2% improvement = saturated
                saturation_dim = dims[i]
                break

    return {
        "per_dim": results,
        "dims": dims,
        "r2_values": r2s,
        "saturation_dim": saturation_dim,
        "max_r2": max(r2s) if r2s else 0.0,
        "min_r2": min(r2s) if r2s else 0.0,
        "marginal_gains": [round(r2s[i] - r2s[i - 1], 4) for i in range(1, len(r2s))] if len(r2s) >= 2 else [],
    }


# ── Composite: Full Publishable Protocol ─────────────────────────────────


def full_publishable_analysis(
    transitions_dict: dict[str, list[StepTransition]],
    transitions_world_b: dict[str, list[StepTransition]] | None = None,
) -> dict[str, dict]:
    """Run all 5 publishable-protocol experiments.

    Args:
        transitions_dict: controller_name → transitions in primary world (A)
        transitions_world_b: optional controller_name → transitions in secondary world (B)
                            for cross-world transfer test

    Returns dict with keys: cross_validation, cross_world, nonlinear_vs_linear,
    noise_robustness, dimensionality_sweep, aggregate_stats.
    """
    names = sorted(transitions_dict.keys())
    result: dict[str, dict] = {}

    # ── Per-pair experiments ──────────────────────────────────────────
    cv_results = []
    nl_results = []
    noise_results = []
    dim_results = []

    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            trivial = _is_trivial_pair(n1, n2)
            pair_label = f"{n1} vs {n2}"

            # 1. Cross-validation
            cv = cross_validated_translation_r2(
                transitions_dict[n1], transitions_dict[n2],
            )
            cv["pair"] = pair_label
            cv["trivial"] = trivial
            cv_results.append(cv)

            # 3. Nonlinear vs linear
            nl = nonlinear_vs_linear_translation(
                transitions_dict[n1], transitions_dict[n2],
            )
            nl["pair"] = pair_label
            nl["trivial"] = trivial
            nl_results.append(nl)

            # 4. Noise robustness
            noise = noise_robustness_sweep(
                transitions_dict[n1], transitions_dict[n2],
            )
            noise["pair"] = pair_label
            noise["trivial"] = trivial
            noise_results.append(noise)

            # 5. Dimensionality sweep
            dim = dimensionality_sweep(
                transitions_dict[n1], transitions_dict[n2],
            )
            dim["pair"] = pair_label
            dim["trivial"] = trivial
            dim_results.append(dim)

    # ── Aggregate stats (excluding trivial pairs) ────────────────────
    nt_cv_test = [r["test_r2"] for r in cv_results if not r.get("trivial")]
    nt_cv_train = [r["train_r2"] for r in cv_results if not r.get("trivial")]
    nt_nl_linear = [r["linear_r2"] for r in nl_results if not r.get("trivial")]
    nt_nl_nonlinear = [r["nonlinear_r2"] for r in nl_results if not r.get("trivial")]

    result["cross_validation"] = {
        "per_pair": cv_results,
        "aggregate_train": distribution_stats(nt_cv_train),
        "aggregate_test": distribution_stats(nt_cv_test),
        "mean_overfitting_gap": float(np.mean(nt_cv_train)) - float(np.mean(nt_cv_test)) if nt_cv_train else 0.0,
    }

    # 2. Cross-world transfer (if world B provided)
    if transitions_world_b is not None:
        result["cross_world"] = cross_world_translation_r2(
            transitions_dict, transitions_world_b,
        )
    else:
        result["cross_world"] = {"status": "skipped — no second world provided"}

    result["nonlinear_vs_linear"] = {
        "per_pair": nl_results,
        "aggregate_linear": distribution_stats(nt_nl_linear),
        "aggregate_nonlinear": distribution_stats(nt_nl_nonlinear),
        "mean_complexity_gap": float(np.mean(nt_nl_nonlinear)) - float(np.mean(nt_nl_linear)) if nt_nl_linear else 0.0,
        "structure_is_linear": all(r.get("structure_is_linear", True) for r in nl_results if not r.get("trivial")),
    }

    result["noise_robustness"] = {
        "per_pair": noise_results,
        "all_robust": all(r.get("robust", False) for r in noise_results if not r.get("trivial")),
        "all_moderate_robust": all(r.get("moderate_robust", False) for r in noise_results if not r.get("trivial")),
        "all_graceful": all(r.get("graceful", True) for r in noise_results if not r.get("trivial")),
        "mean_degradation_rate": float(np.mean([
            r["degradation_rate"] for r in noise_results if not r.get("trivial")
        ])) if noise_results else 0.0,
    }

    result["dimensionality_sweep"] = {
        "per_pair": dim_results,
        "saturation_dims": [r.get("saturation_dim") for r in dim_results if not r.get("trivial")],
    }

    # ── Overall aggregate (non-trivial only) ─────────────────────────
    all_test_r2 = nt_cv_test
    result["aggregate"] = {
        "n_pairs_total": len(cv_results),
        "n_pairs_nontrivial": len(nt_cv_test),
        "cv_test_r2": distribution_stats(all_test_r2),
        "linear_r2": distribution_stats(nt_nl_linear),
        "nonlinear_r2": distribution_stats(nt_nl_nonlinear),
    }

    return result
