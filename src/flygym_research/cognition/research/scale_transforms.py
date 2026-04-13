"""Scale transforms for PR #5 — extended suite + fake metric generation.

Extends existing transforms (coarse-grain, downsample, pool) and adds
explicitly scale-sensitive "fake" metrics for contrastive benchmarking.
"""

from __future__ import annotations

import numpy as np

from ..interfaces import StepTransition


# ---------------------------------------------------------------------------
# Scale transforms (real features should survive these)
# ---------------------------------------------------------------------------

def coarse_grain(
    transitions: list[StepTransition],
    factor: int = 2,
) -> list[StepTransition]:
    """Average consecutive transitions in groups of ``factor``."""
    if factor < 2 or not transitions:
        return transitions
    result = []
    for i in range(0, len(transitions) - factor + 1, factor):
        group = transitions[i : i + factor]
        # Use the last transition as representative but average rewards
        last = group[-1]
        avg_reward = float(np.mean([t.reward for t in group]))
        result.append(StepTransition(
            observation=last.observation,
            action=last.action,
            reward=avg_reward,
            terminated=last.terminated,
            truncated=last.truncated,
            info=last.info,
        ))
    return result


def downsample(
    transitions: list[StepTransition],
    factor: int = 2,
) -> list[StepTransition]:
    """Keep every n-th transition."""
    if factor < 2 or not transitions:
        return transitions
    return transitions[::factor]


def feature_pool(
    transitions: list[StepTransition],
    window: int = 2,
) -> list[StepTransition]:
    """Max-pool features over sliding window."""
    if window < 2 or not transitions:
        return transitions
    result = []
    for i in range(len(transitions) - window + 1):
        group = transitions[i : i + window]
        # Pool features by taking max over window
        pooled_feats = {}
        all_keys = set()
        for t in group:
            all_keys.update(t.observation.summary.features.keys())
        for k in all_keys:
            vals = [t.observation.summary.features.get(k, 0.0) for t in group]
            pooled_feats[k] = float(np.max(vals))

        last = group[-1]
        obs = last.observation
        new_summary = type(obs.summary)(
            features=pooled_feats,
            active_channels=obs.summary.active_channels,
            disabled_channels=obs.summary.disabled_channels,
        )
        new_obs = type(obs)(
            raw_body=obs.raw_body,
            summary=new_summary,
            world=obs.world,
            history=obs.history,
        )
        result.append(StepTransition(
            observation=new_obs,
            action=last.action,
            reward=last.reward,
            terminated=last.terminated,
            truncated=last.truncated,
            info=last.info,
        ))
    return result


def temporal_aggregate(
    transitions: list[StepTransition],
    window: int = 3,
) -> list[StepTransition]:
    """Mean-aggregate features over sliding window."""
    if window < 2 or not transitions:
        return transitions
    result = []
    for i in range(len(transitions) - window + 1):
        group = transitions[i : i + window]
        agg_feats = {}
        all_keys = set()
        for t in group:
            all_keys.update(t.observation.summary.features.keys())
        for k in all_keys:
            vals = [t.observation.summary.features.get(k, 0.0) for t in group]
            agg_feats[k] = float(np.mean(vals))

        last = group[-1]
        obs = last.observation
        new_summary = type(obs.summary)(
            features=agg_feats,
            active_channels=obs.summary.active_channels,
            disabled_channels=obs.summary.disabled_channels,
        )
        new_obs = type(obs)(
            raw_body=obs.raw_body,
            summary=new_summary,
            world=obs.world,
            history=obs.history,
        )
        result.append(StepTransition(
            observation=new_obs,
            action=last.action,
            reward=last.reward,
            terminated=last.terminated,
            truncated=last.truncated,
            info=last.info,
        ))
    return result


def window_resize(
    transitions: list[StepTransition],
    target_length: int = 50,
) -> list[StepTransition]:
    """Resize transition sequence to target length via uniform sampling."""
    if not transitions or target_length <= 0:
        return transitions
    n = len(transitions)
    indices = np.linspace(0, n - 1, target_length, dtype=int)
    return [transitions[i] for i in indices]


def compression_proxy(
    transitions: list[StepTransition],
    keep_fraction: float = 0.5,
) -> list[StepTransition]:
    """Keep only a fraction of transitions (simulates lossy compression)."""
    if not transitions:
        return transitions
    n_keep = max(1, int(len(transitions) * keep_fraction))
    indices = np.linspace(0, len(transitions) - 1, n_keep, dtype=int)
    return [transitions[i] for i in indices]


def histogram_fold(
    transitions: list[StepTransition],
    bins: int = 4,
) -> list[StepTransition]:
    """Quantize action/feature observations into coarse bins."""
    if bins < 2 or not transitions:
        return transitions
    result = []
    for transition in transitions:
        obs = transition.observation
        feats = {
            key: float(np.round(value * bins) / bins)
            for key, value in obs.summary.features.items()
        }
        new_summary = type(obs.summary)(
            features=feats,
            active_channels=obs.summary.active_channels,
            disabled_channels=obs.summary.disabled_channels,
        )
        new_obs = type(obs)(
            raw_body=obs.raw_body,
            summary=new_summary,
            world=obs.world,
            history=obs.history,
        )
        result.append(StepTransition(
            observation=new_obs,
            action=transition.action,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            info={**transition.info, "histogram_fold": bins},
        ))
    return result


# ---------------------------------------------------------------------------
# Scale transform registry
# ---------------------------------------------------------------------------

SCALE_TRANSFORMS: dict[str, object] = {
    "original": lambda t: t,
    "coarse_2": lambda t: coarse_grain(t, 2),
    "coarse_4": lambda t: coarse_grain(t, 4),
    "downsample_2": lambda t: downsample(t, 2),
    "downsample_3": lambda t: downsample(t, 3),
    "pool_2": lambda t: feature_pool(t, 2),
    "pool_3": lambda t: feature_pool(t, 3),
    "temporal_agg_3": lambda t: temporal_aggregate(t, 3),
    "window_resize_50": lambda t: window_resize(t, 50),
    "compress_50": lambda t: compression_proxy(t, 0.5),
    "downsample_5": lambda t: downsample(t, 5),
    "coarse_8": lambda t: coarse_grain(t, 8),
    "compress_25": lambda t: compression_proxy(t, 0.25),
    "histogram_fold_4": lambda t: histogram_fold(t, 4),
}


def apply_scale_transform(
    transitions: list[StepTransition],
    name: str,
) -> list[StepTransition]:
    """Apply a named scale transform."""
    fn = SCALE_TRANSFORMS.get(name)
    if fn is None:
        return transitions
    return fn(transitions)  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Fake metrics (should be scale-sensitive / collapse under transforms)
# ---------------------------------------------------------------------------

def compute_fake_metrics(transitions: list[StepTransition]) -> dict[str, float]:
    """Compute deliberately scale-sensitive metrics.

    These should COLLAPSE or change significantly under scale transforms,
    unlike real structural features which should remain stable.
    """
    if not transitions:
        return {
            "raw_action_count": 0.0,
            "stepwise_noise_magnitude": 0.0,
            "chunk_specific_count": 0.0,
            "unnormalized_residual_tally": 0.0,
            "local_oscillation_count": 0.0,
            "resolution_support_count": 0.0,
            "small_window_variance": 0.0,
        }

    n = len(transitions)
    actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in transitions],
        dtype=np.float64,
    )
    rewards = np.array([t.reward for t in transitions], dtype=np.float64)
    step_weights = np.arange(1, n + 1, dtype=np.float64)

    # 1. Raw action count — directly proportional to episode length
    raw_action_count = float(n)

    # 2. Stepwise noise magnitude — sum of absolute action changes
    if n > 1:
        stepwise_noise = float(np.sum(np.abs(np.diff(actions))) * max(n / 8.0, 1.0))
    else:
        stepwise_noise = 0.0

    # 3. Chunk-size-specific count — number of length-3 chunks
    chunk_count = float(n // 3)

    # 4. Unnormalized residual tally — sum of squared rewards
    unnorm_residual = float(np.sum((rewards ** 2) * step_weights))

    # 5. Local oscillation count — number of sign changes in action diff
    if n > 2:
        diffs = np.diff(actions)
        sign_changes = np.sum(np.abs(np.diff(np.sign(diffs))) > 0)
        oscillation_count = float(sign_changes)
    else:
        oscillation_count = 0.0

    # 6. Resolution-specific support count — unique action values
    resolution_support = float(len(np.unique(np.round(actions * max(n / 10.0, 1.0), 1))))

    # 7. Small-window variance — variance computed on first 5 steps only
    small_window = actions[:min(5, n)]
    small_window_var = (
        float(np.var(small_window) * max(n / 5.0, 1.0))
        if len(small_window) > 1
        else 0.0
    )

    return {
        "raw_action_count": raw_action_count,
        "stepwise_noise_magnitude": stepwise_noise,
        "chunk_specific_count": chunk_count,
        "unnormalized_residual_tally": unnorm_residual,
        "local_oscillation_count": oscillation_count,
        "resolution_support_count": resolution_support,
        "small_window_variance": small_window_var,
    }


# ---------------------------------------------------------------------------
# Real metrics (should survive scale transforms)
# ---------------------------------------------------------------------------

def compute_real_metrics(transitions: list[StepTransition]) -> dict[str, float]:
    """Compute scale-robust structural metrics.

    These should remain STABLE across scale transforms.
    """
    if not transitions:
        return {
            "mean_return": 0.0,
            "success_rate": 0.0,
            "stability_mean": 0.0,
            "path_consistency": 0.0,
            "normalized_action_entropy": 0.0,
        }

    rewards = np.array([t.reward for t in transitions], dtype=np.float64)
    mean_return = float(np.mean(rewards))
    success_rate = float(np.mean(rewards > 0))

    stabilities = []
    for t in transitions:
        stabilities.append(t.observation.summary.features.get("stability", 0.5))
    stability_mean = float(np.mean(stabilities))

    # Path consistency — autocorrelation of actions at lag 1
    actions = np.array(
        [float(getattr(t.action, "move_intent", 0.0)) for t in transitions],
        dtype=np.float64,
    )
    action_prev = actions[:-1]
    action_next = actions[1:]
    if (
        len(actions) > 2
        and np.std(action_prev) > 1e-10
        and np.std(action_next) > 1e-10
    ):
        path_consistency = float(np.corrcoef(action_prev, action_next)[0, 1])
        if not np.isfinite(path_consistency):
            path_consistency = 0.0
    else:
        path_consistency = 0.0

    # Normalized action entropy
    if len(actions) > 1:
        hist, _ = np.histogram(actions, bins=min(10, len(actions) // 2 + 1))
        hist = hist[hist > 0].astype(np.float64)
        p = hist / hist.sum()
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        max_entropy = np.log(len(hist) + 1e-12)
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        norm_entropy = 0.0

    return {
        "mean_return": mean_return,
        "success_rate": success_rate,
        "stability_mean": stability_mean,
        "path_consistency": path_consistency,
        "normalized_action_entropy": norm_entropy,
    }
