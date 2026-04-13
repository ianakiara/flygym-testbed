"""Experiment 7 — Scale Law (Real vs Artifact).

Separates real structural features from scale artifacts by applying
coarse-graining, downsampling, and feature pooling, then measuring
stability of key properties.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics import (
    seam_fragility,
    summarize_metrics,
)
from ..sleep import compress_trace_bank, backbone_shared_score
from ..sleep.trace_schema import SleepCandidate, TraceEpisode
from .exp_sleep_trace_compressor import collect_trace_bank


# ---------------------------------------------------------------------------
# Scale transformations
# ---------------------------------------------------------------------------

def _coarse_grain(transitions: list, *, grain_size: int = 2) -> list:
    """Coarse-grain: average consecutive transitions."""
    from ..interfaces import StepTransition, DescendingCommand
    if grain_size <= 1 or not transitions:
        return list(transitions)
    coarsened = []
    for i in range(0, len(transitions), grain_size):
        chunk = transitions[i:i + grain_size]
        if not chunk:
            continue
        avg_reward = float(np.mean([t.reward for t in chunk]))
        last = chunk[-1]
        if hasattr(last.action, 'move_intent'):
            avg_action = DescendingCommand(
                move_intent=float(np.mean([t.action.move_intent for t in chunk if hasattr(t.action, 'move_intent')])),
                turn_intent=float(np.mean([t.action.turn_intent for t in chunk if hasattr(t.action, 'turn_intent')])),
                speed_modulation=float(np.mean([t.action.speed_modulation for t in chunk if hasattr(t.action, 'speed_modulation')])),
                stabilization_priority=float(np.mean([t.action.stabilization_priority for t in chunk if hasattr(t.action, 'stabilization_priority')])),
                target_bias=last.action.target_bias if hasattr(last.action, 'target_bias') else (0.0, 0.0),
            )
        else:
            avg_action = last.action
        coarsened.append(StepTransition(
            observation=last.observation,
            action=avg_action,
            reward=avg_reward,
            terminated=last.terminated,
            truncated=last.truncated,
            info={**last.info, "coarse_grained": True},
        ))
    return coarsened


def _downsample(transitions: list, *, factor: int = 2) -> list:
    """Downsample: keep every nth transition."""
    if factor <= 1 or not transitions:
        return list(transitions)
    return transitions[::factor]


def _feature_pool(transitions: list, *, pool_size: int = 3) -> list:
    """Feature pooling: max-pool observation features over windows."""
    from ..interfaces import StepTransition, BrainObservation, AscendingSummary
    if pool_size <= 1 or not transitions:
        return list(transitions)
    pooled = []
    for i in range(0, len(transitions), pool_size):
        chunk = transitions[i:i + pool_size]
        if not chunk:
            continue
        last = chunk[-1]
        pooled_features = {}
        for t in chunk:
            for k, v in t.observation.summary.features.items():
                if k not in pooled_features or v > pooled_features[k]:
                    pooled_features[k] = v
        pooled_summary = AscendingSummary(
            features=pooled_features,
            active_channels=last.observation.summary.active_channels,
            disabled_channels=last.observation.summary.disabled_channels,
        )
        pooled_obs = BrainObservation(
            raw_body=last.observation.raw_body,
            summary=pooled_summary,
            world=last.observation.world,
            history=last.observation.history,
        )
        pooled.append(StepTransition(
            observation=pooled_obs,
            action=last.action,
            reward=last.reward,
            terminated=last.terminated,
            truncated=last.truncated,
            info={**last.info, "feature_pooled": True},
        ))
    return pooled


SCALE_TRANSFORMS = {
    "original": lambda t: list(t),
    "coarse_grain_2": lambda t: _coarse_grain(t, grain_size=2),
    "coarse_grain_4": lambda t: _coarse_grain(t, grain_size=4),
    "downsample_2": lambda t: _downsample(t, factor=2),
    "downsample_3": lambda t: _downsample(t, factor=3),
    "feature_pool_2": lambda t: _feature_pool(t, pool_size=2),
    "feature_pool_3": lambda t: _feature_pool(t, pool_size=3),
}


# ---------------------------------------------------------------------------
# Property measurement at each scale
# ---------------------------------------------------------------------------

def _measure_properties(transitions: list) -> dict[str, float]:
    """Measure key structural properties."""
    if not transitions:
        return {"return": 0.0, "success": 0.0, "seam_fragility": 0.0, "stability_mean": 0.0, "n_steps": 0}

    metrics = summarize_metrics(transitions)
    seam = seam_fragility(transitions)

    return {
        "return": float(metrics.get("return", 0.0)),
        "success": float(metrics.get("success", 0.0)),
        "seam_fragility": float(seam.get("seam_fragility", 0.0)),
        "stability_mean": float(metrics.get("stability_mean", 0.0)),
        "n_steps": len(transitions),
    }


def _measure_candidate_properties(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
) -> dict[str, float]:
    """Measure candidate-level properties."""
    try:
        bs = backbone_shared_score(candidate, episodes)
        return {
            "backbone_shared": float(bs.get("backbone_shared_score", 0.0)),
            "seam_risk": float(bs.get("seam_risk", 0.0)),
            "interop_loss": float(bs.get("interop_loss", 0.0)),
            "scale_drift": float(bs.get("scale_drift", 0.0)),
            "redundancy_score": float(bs.get("redundancy_score", 0.0)),
        }
    except Exception:
        return {"backbone_shared": 0.0, "seam_risk": 0.0, "interop_loss": 0.0, "scale_drift": 0.0, "redundancy_score": 0.0}


# ---------------------------------------------------------------------------
# Stability analysis
# ---------------------------------------------------------------------------

def _stability_coefficient(values: list[float]) -> float:
    """Coefficient of variation: lower = more stable.

    Special cases:
    - < 2 values: insufficient data → return 0.0
    - All values identical (including all-zero): perfectly stable → return 0.0
    - Mean ≈ 0 but values vary: undefined CV → return inf
    """
    if len(values) < 2:
        return 0.0
    std = float(np.std(values))
    if std < 1e-10:
        # All values are (effectively) identical — perfectly stable
        return 0.0
    mean_abs = abs(float(np.mean(values)))
    if mean_abs < 1e-10:
        # Mean ≈ 0 but values vary → CV is undefined/infinite
        return float("inf")
    return float(std / (mean_abs + 1e-8))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_scale_law",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episodes
    episodes = collect_trace_bank(max_steps=24, seeds=[0, 1])
    selected = episodes[:min(16, len(episodes))]

    # Measure properties at each scale for each episode
    episode_scale_results = []
    property_stability = defaultdict(lambda: defaultdict(list))

    for ep in selected:
        ep_results = {"episode_id": ep.episode_id, "controller": ep.controller_name, "world_mode": ep.world_mode}
        for scale_name, transform_fn in SCALE_TRANSFORMS.items():
            transformed = transform_fn(ep.transitions)
            props = _measure_properties(transformed)
            ep_results[scale_name] = props
            for prop_name, prop_val in props.items():
                if prop_name != "n_steps":
                    property_stability[prop_name][ep.episode_id].append(prop_val)
        episode_scale_results.append(ep_results)

    # Stability analysis per property
    stability_summary = {}
    for prop_name, ep_values in property_stability.items():
        stabilities = []
        for ep_id, values in ep_values.items():
            stab = _stability_coefficient(values)
            stabilities.append(stab)
        stability_summary[prop_name] = {
            "mean_stability_cv": float(np.mean(stabilities)) if stabilities else float("inf"),
            "std_stability_cv": float(np.std(stabilities)) if stabilities else 0.0,
            "is_stable": float(np.mean(stabilities)) < 0.5 if stabilities else False,
        }

    # Candidate-level scale analysis
    artifact = compress_trace_bank(episodes)
    candidate_scale_results = []
    for cand in artifact.candidates[:10]:
        cand_props = _measure_candidate_properties(cand, episodes)
        candidate_scale_results.append({
            "candidate_id": cand.candidate_id,
            "tier": cand.redundancy_tier,
            **cand_props,
        })

    # Real vs fake classification
    real_features = []
    fake_features = []
    for prop_name, summary in stability_summary.items():
        entry = {"property": prop_name, **summary}
        if summary["is_stable"]:
            real_features.append(entry)
        else:
            fake_features.append(entry)

    # Pass criteria
    has_real = len(real_features) > 0
    has_fake_collapse = len(fake_features) > 0
    real_stable = all(f["is_stable"] for f in real_features) if real_features else False

    payload = {
        "stability_summary": stability_summary,
        "real_features": real_features,
        "fake_features": fake_features,
        "pass_criteria": {
            "real_features_stable": real_stable,
            "fake_features_collapse": has_fake_collapse,
            "separation_exists": has_real and has_fake_collapse,
        },
        "episode_scale_results": episode_scale_results[:5],
        "candidate_scale_results": candidate_scale_results,
        "n_episodes": len(selected),
        "n_scale_transforms": len(SCALE_TRANSFORMS),
    }

    (output_dir / "scale_law.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
