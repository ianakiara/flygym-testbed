"""Advanced persistence metrics — extends the basic lag-1 autocorrelation
with cross-time mutual information estimates, state decay curves,
and predictive utility of past hidden state.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition


def cross_time_mutual_information(
    transitions: list[StepTransition],
    *,
    key: str = "phase",
    n_bins: int = 10,
    max_lag: int = 5,
) -> dict[str, float]:
    """Estimate mutual information between feature values at different lags.

    Uses histogram-based MI estimation.  Returns MI at each lag up to
    *max_lag*, plus the average MI across lags.
    """
    values = np.array(
        [t.observation.summary.features.get(key, 0.0) for t in transitions],
        dtype=np.float64,
    )
    if len(values) < max_lag + 2:
        result: dict[str, float] = {"mi_mean": 0.0}
        for lag in range(1, max_lag + 1):
            result[f"mi_lag_{lag}"] = 0.0
        return result

    mi_values: list[float] = []
    result = {}
    for lag in range(1, max_lag + 1):
        x = values[:-lag]
        y = values[lag:]
        mi = _histogram_mi(x, y, n_bins)
        result[f"mi_lag_{lag}"] = mi
        mi_values.append(mi)
    result["mi_mean"] = float(np.mean(mi_values)) if mi_values else 0.0
    return result


def _histogram_mi(x: np.ndarray, y: np.ndarray, n_bins: int) -> float:
    """Estimate mutual information via 2D histogram."""
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    try:
        hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
        pxy = hist_2d / hist_2d.sum()
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
        mi = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
        return max(float(mi), 0.0)
    except (ValueError, FloatingPointError):
        return 0.0


def state_decay_curve(
    transitions: list[StepTransition],
    *,
    key: str = "phase",
    max_lag: int = 10,
) -> dict[str, float]:
    """Compute autocorrelation at increasing lags to produce a decay curve.

    Returns autocorrelation at each lag and the decay half-life estimate
    (lag at which autocorrelation first drops below 0.5).
    """
    values = np.array(
        [t.observation.summary.features.get(key, 0.0) for t in transitions],
        dtype=np.float64,
    )
    if len(values) < max_lag + 2:
        result: dict[str, float] = {"decay_half_life": 0.0}
        for lag in range(1, max_lag + 1):
            result[f"autocorr_lag_{lag}"] = 0.0
        return result

    mean_val = values.mean()
    var_val = values.var()
    result = {}
    half_life = float(max_lag)  # Default: no decay within window.

    for lag in range(1, max_lag + 1):
        if var_val < 1e-12:
            ac = 0.0
        else:
            cov = np.mean((values[:-lag] - mean_val) * (values[lag:] - mean_val))
            ac = float(cov / var_val)
            if np.isnan(ac):
                ac = 0.0
        result[f"autocorr_lag_{lag}"] = ac
        if ac < 0.5 and half_life == float(max_lag):
            half_life = float(lag)

    result["decay_half_life"] = half_life
    return result


def predictive_utility(
    transitions: list[StepTransition],
    *,
    state_key: str = "phase",
    target_key: str = "stability",
    horizon: int = 3,
) -> dict[str, float]:
    """Measure how well past state predicts future observations.

    Uses simple linear correlation between past *state_key* and future
    *target_key* at the given horizon.  Higher correlation means the
    internal state carries predictive information.
    """
    values_state = np.array(
        [t.observation.summary.features.get(state_key, 0.0) for t in transitions],
        dtype=np.float64,
    )
    values_target = np.array(
        [t.observation.summary.features.get(target_key, 0.0) for t in transitions],
        dtype=np.float64,
    )
    if len(values_state) < horizon + 2:
        return {"predictive_utility": 0.0, "predictive_horizon": float(horizon)}

    past = values_state[: -horizon]
    future = values_target[horizon:]
    n = min(len(past), len(future))
    past = past[:n]
    future = future[:n]

    if np.allclose(past, past[0]) or np.allclose(future, future[0]):
        return {"predictive_utility": 0.0, "predictive_horizon": float(horizon)}

    corr = float(np.corrcoef(past, future)[0, 1])
    return {
        "predictive_utility": 0.0 if np.isnan(corr) else abs(corr),
        "predictive_horizon": float(horizon),
    }


def hysteresis_metric(
    transitions: list[StepTransition],
    *,
    state_key: str = "stability",
    action_key: str = "move_intent",
    n_bins: int = 5,
) -> dict[str, float]:
    """Measure hysteresis — whether the same state leads to different actions
    depending on the trajectory direction (rising vs falling).

    Bins the state variable and computes mean action for rising vs falling
    transitions through each bin.
    """
    states: list[float] = []
    actions: list[float] = []
    for t in transitions:
        states.append(t.observation.summary.features.get(state_key, 0.0))
        if hasattr(t.action, action_key):
            actions.append(float(getattr(t.action, action_key)))
        else:
            actions.append(0.0)

    if len(states) < 3:
        return {"hysteresis_score": 0.0}

    s = np.array(states, dtype=np.float64)
    a = np.array(actions, dtype=np.float64)

    # Classify each transition as rising or falling.
    rising_mask = np.diff(s) > 0
    falling_mask = np.diff(s) < 0

    # Bin the states.
    if s.max() - s.min() < 1e-8:
        return {"hysteresis_score": 0.0}

    bin_edges = np.linspace(s.min(), s.max(), n_bins + 1)
    bin_indices = np.digitize(s[:-1], bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    diffs: list[float] = []
    for b in range(n_bins):
        rising_in_bin = (bin_indices == b) & rising_mask
        falling_in_bin = (bin_indices == b) & falling_mask
        if rising_in_bin.sum() > 0 and falling_in_bin.sum() > 0:
            mean_rising = float(a[:-1][rising_in_bin].mean())
            mean_falling = float(a[:-1][falling_in_bin].mean())
            diffs.append(abs(mean_rising - mean_falling))

    return {
        "hysteresis_score": float(np.mean(diffs)) if diffs else 0.0,
    }
