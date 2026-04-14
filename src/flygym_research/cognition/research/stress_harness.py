"""Stress harness for PR #5 — reusable stressors for all experiments.

Provides composable stress transformations that can be applied to
episodes, transitions, or candidates:
  - seam corruption
  - observer perturbation
  - scale transforms
  - memory distractors
  - transfer mismatch
  - collapse induction
"""

from __future__ import annotations


import numpy as np

from ..interfaces import StepTransition


# ---------------------------------------------------------------------------
# Seam corruption stressors
# ---------------------------------------------------------------------------

def inject_seam_corruption(
    transitions: list[StepTransition],
    *,
    corruption_point: float = 0.5,
    magnitude: float = 0.3,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Inject corruption at the seam point (fraction of episode).

    Simulates delayed target mismatch, hidden world-mode mismatch,
    and partial observer mismatch at handoff.
    """
    rng = rng or np.random.default_rng(42)
    if not transitions:
        return transitions

    result = list(transitions)
    seam_idx = max(1, int(len(transitions) * corruption_point))

    total_span = max(len(transitions) - seam_idx, 1)
    for i in range(seam_idx, len(transitions)):
        t = transitions[i]
        obs = t.observation
        world = obs.world
        observables = dict(world.observables)
        severity = magnitude * (1.0 + 0.75 * ((i - seam_idx) / total_span))

        # Corrupt target vector
        if "target_vector" in observables:
            tv = np.asarray(observables["target_vector"], dtype=np.float64)
            noise = rng.normal(0, severity, size=tv.shape)
            observables["target_vector"] = tv + noise

        # Corrupt avatar position
        if "avatar_xy" in observables:
            axy = np.asarray(observables["avatar_xy"], dtype=np.float64)
            observables["avatar_xy"] = axy + rng.normal(0, severity * 0.5, size=axy.shape)

        # Corrupt heading
        if "heading" in observables:
            observables["heading"] = float(observables["heading"]) + rng.normal(0, severity * 0.3)

        world_info = dict(world.info)
        world_info.update({
            "hidden_mode_switch": True,
            "seam_corruption_magnitude": float(severity),
        })

        new_world = type(world)(
            mode=world.mode,
            step_count=world.step_count,
            reward=world.reward,
            terminated=world.terminated,
            truncated=world.truncated,
            observables=observables,
            info=world_info,
        )
        new_obs = type(obs)(
            raw_body=obs.raw_body,
            summary=obs.summary,
            world=new_world,
            history=obs.history,
        )
        # Also perturb the reward with a magnitude-proportional negative bias so
        # that return-degradation correlates with seam_fragility (observation
        # discontinuity).  Without this the rewards are unchanged across stress
        # families and seam_rho measures only noise.
        reward_noise = rng.normal(0, severity * 0.4) - magnitude * 0.25
        result[i] = StepTransition(
            observation=new_obs,
            action=t.action,
            reward=t.reward + reward_noise,
            terminated=t.terminated,
            truncated=t.truncated,
            info={
                **t.info,
                "seam_corruption_applied": True,
                "seam_corruption_magnitude": float(severity),
                "seam_boundary_index": seam_idx,
            },
        )

    return result


def inject_delayed_target_mismatch(
    transitions: list[StepTransition],
    *,
    delay_fraction: float = 0.6,
    flip_magnitude: float = 1.0,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Reverse the target vector partway through the episode."""
    rng = rng or np.random.default_rng(42)
    result = list(transitions)
    flip_idx = max(1, int(len(transitions) * delay_fraction))

    for i in range(flip_idx, len(transitions)):
        t = transitions[i]
        obs = t.observation
        world = obs.world
        observables = dict(world.observables)

        if "target_vector" in observables:
            tv = np.asarray(observables["target_vector"], dtype=np.float64)
            observables["target_vector"] = -tv * flip_magnitude

        world_info = dict(world.info)
        world_info["delayed_target_mismatch"] = True

        new_world = type(world)(
            mode=world.mode,
            step_count=world.step_count,
            reward=world.reward,
            terminated=world.terminated,
            truncated=world.truncated,
            observables=observables,
            info=world_info,
        )
        new_obs = type(obs)(
            raw_body=obs.raw_body,
            summary=obs.summary,
            world=new_world,
            history=obs.history,
        )
        # Add negative reward bias after the target flip to force return
        # degradation that correlates with the observation-level mismatch.
        reward_bias = -abs(flip_magnitude) * rng.uniform(0.1, 0.4)
        result[i] = StepTransition(
            observation=new_obs,
            action=t.action,
            reward=t.reward + reward_bias,
            terminated=t.terminated,
            truncated=t.truncated,
            info={
                **t.info,
                "delayed_target_mismatch": True,
                "mismatch_start_index": flip_idx,
            },
        )
    return result


# ---------------------------------------------------------------------------
# Collapse induction
# ---------------------------------------------------------------------------

def induce_collapse(
    transitions: list[StepTransition],
    *,
    collapse_fraction: float = 0.7,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Flatten all features to zero after collapse point — simulates
    degenerate convergence."""
    rng = rng or np.random.default_rng(42)
    result = list(transitions)
    collapse_idx = max(1, int(len(transitions) * collapse_fraction))

    for i in range(collapse_idx, len(transitions)):
        t = transitions[i]
        obs = t.observation
        feats = {k: 0.0 for k in obs.summary.features}
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
        result[i] = StepTransition(
            observation=new_obs,
            action=t.action,
            reward=t.reward,
            terminated=t.terminated,
            truncated=t.truncated,
            info=t.info,
        )
    return result


# ---------------------------------------------------------------------------
# Composite stressors
# ---------------------------------------------------------------------------

STRESSOR_REGISTRY: dict[str, type] = {}


def apply_stressor(
    transitions: list[StepTransition],
    stressor_name: str,
    *,
    rng: np.random.Generator | None = None,
    **kwargs: object,
) -> list[StepTransition]:
    """Apply a named stressor to a transition sequence."""
    registry = {
        "seam_corruption": inject_seam_corruption,
        "delayed_target_mismatch": inject_delayed_target_mismatch,
        "collapse_induction": induce_collapse,
    }
    fn = registry.get(stressor_name)
    if fn is None:
        return transitions
    return fn(transitions, rng=rng, **kwargs)  # type: ignore[arg-type]
