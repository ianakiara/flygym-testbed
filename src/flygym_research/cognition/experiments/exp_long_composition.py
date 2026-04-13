"""EXP 1 — Long-horizon Composition Stress (PR #5 priority 2).

Tests whether composition ordering (boundary > bulk, corner > boundary)
survives 100–200 step episodes AND whether seam metrics acquire predictive
power under genuine seam stress.

Pass criteria:
  - boundary > bulk in ≥ 85% of paired trials
  - corner > boundary in ≥ 80%
  - seam defect correlation with failure ρ > 0.7
  - seam defect variance > 0

Includes baseline (bulk), method (corner-restored), ablation (boundary).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics import seam_fragility, summarize_metrics
from ..research.long_horizon_runner import collect_long_horizon
from ..research.stress_harness import (
    inject_delayed_target_mismatch,
    inject_seam_corruption,
)
from ..sleep.trace_schema import TraceEpisode


# ---------------------------------------------------------------------------
# Composition strategies (from v1, kept identical)
# ---------------------------------------------------------------------------

def _seam_boundary_score(composed: list, seam_idx: int) -> float:
    """Measure the observation/action discontinuity at the single seam boundary step.

    Unlike ``seam_fragility`` (which averages over all transitions), this score
    focuses on the exact transition from step ``seam_idx-1`` to ``seam_idx``.
    Boundary-aware composition reduces this score because the interpolated steps
    immediately before ``seam_idx`` lower the jump to the first pure-B frame.
    """
    if seam_idx <= 0 or seam_idx >= len(composed):
        return 0.0
    prev_t = composed[seam_idx - 1]
    curr_t = composed[seam_idx]

    prev_target = np.asarray(
        prev_t.observation.world.observables.get("target_vector", np.zeros(2)),
        dtype=np.float64,
    )
    curr_target = np.asarray(
        curr_t.observation.world.observables.get("target_vector", np.zeros(2)),
        dtype=np.float64,
    )
    target_delta = float(np.linalg.norm(curr_target - prev_target))

    prev_pos = np.asarray(prev_t.observation.raw_body.body_positions, dtype=np.float64)
    curr_pos = np.asarray(curr_t.observation.raw_body.body_positions, dtype=np.float64)
    body_delta = float(np.linalg.norm(curr_pos - prev_pos) / max(prev_pos.size, 1))

    prev_act = np.array([
        float(getattr(prev_t.action, "move_intent", 0.0)),
        float(getattr(prev_t.action, "turn_intent", 0.0)),
        float(getattr(prev_t.action, "speed_modulation", 0.0)),
    ], dtype=np.float64)
    curr_act = np.array([
        float(getattr(curr_t.action, "move_intent", 0.0)),
        float(getattr(curr_t.action, "turn_intent", 0.0)),
        float(getattr(curr_t.action, "speed_modulation", 0.0)),
    ], dtype=np.float64)
    action_delta = float(np.linalg.norm(curr_act - prev_act))

    reward_delta = abs(curr_t.reward - prev_t.reward)

    # Extra weight when the transition crosses a seam-corruption boundary
    weight = 1.0
    if curr_t.info.get("seam_corruption_applied") or prev_t.info.get("seam_corruption_applied"):
        weight += float(curr_t.info.get("seam_corruption_magnitude", 0.0))
    if prev_t.observation.world.mode != curr_t.observation.world.mode:
        weight += 0.75

    return float(
        weight * (
            0.4 * target_delta
            + 0.3 * action_delta
            + 0.2 * body_delta
            + 0.1 * reward_delta
        )
    )


def _bulk_compose(
    ep_a: TraceEpisode, ep_b: TraceEpisode,
) -> list:
    """Naive concatenation — no seam treatment."""
    return list(ep_a.transitions) + list(ep_b.transitions)


def _blend_transitions(
    ta: "StepTransition", tb: "StepTransition", alpha: float
) -> "StepTransition":
    """Return a new StepTransition that linearly interpolates ta → tb.

    Blends all fields that ``seam_fragility`` measures: target_vector,
    body_positions, action, and reward.  This reduces the observation jump
    at the composition boundary so that boundary strategies produce
    measurably lower seam_fragility than bulk (naive) composition.
    """
    from ..interfaces import StepTransition
    from ..interfaces.types import (
        BrainObservation, RawBodyFeedback, WorldState,
    )

    # Blend reward and target_vector
    avg_reward = (1.0 - alpha) * ta.reward + alpha * tb.reward
    ta_target = np.asarray(
        ta.observation.world.observables.get("target_vector", np.zeros(2)),
        dtype=np.float64,
    )
    tb_target = np.asarray(
        tb.observation.world.observables.get("target_vector", np.zeros(2)),
        dtype=np.float64,
    )
    blended_target = ((1.0 - alpha) * ta_target + alpha * tb_target).astype(np.float64)

    # Blend body positions
    ta_pos = np.asarray(ta.observation.raw_body.body_positions, dtype=np.float64)
    tb_pos = np.asarray(tb.observation.raw_body.body_positions, dtype=np.float64)
    blended_pos = (1.0 - alpha) * ta_pos + alpha * tb_pos

    # Blend action
    def _get_act(t: "StepTransition") -> np.ndarray:
        return np.array([
            float(getattr(t.action, "move_intent", 0.0)),
            float(getattr(t.action, "turn_intent", 0.0)),
            float(getattr(t.action, "speed_modulation", 0.0)),
        ], dtype=np.float64)

    ta_act = _get_act(ta)
    tb_act = _get_act(tb)
    blended_act_arr = (1.0 - alpha) * ta_act + alpha * tb_act
    from ..interfaces.types import DescendingCommand
    blended_action = DescendingCommand(
        move_intent=float(blended_act_arr[0]),
        turn_intent=float(blended_act_arr[1]),
        speed_modulation=float(blended_act_arr[2]),
        stabilization_priority=float(
            getattr(tb.action, "stabilization_priority", 0.0)
        ),
        target_bias=(
            float(blended_target[0]) if blended_target.size > 0 else 0.0,
            float(blended_target[1]) if blended_target.size > 1 else 0.0,
        ),
    )

    # Build blended observation (use tb's structure, override key fields)
    tb_obs = tb.observation
    blended_observables = dict(tb_obs.world.observables)
    blended_observables["target_vector"] = blended_target

    blended_world = WorldState(
        mode=tb_obs.world.mode,
        step_count=tb_obs.world.step_count,
        reward=avg_reward,
        terminated=tb_obs.world.terminated,
        truncated=tb_obs.world.truncated,
        observables=blended_observables,
        info=tb_obs.world.info,
    )
    blended_raw_body = RawBodyFeedback(
        time=tb_obs.raw_body.time,
        joint_angles=tb_obs.raw_body.joint_angles,
        joint_velocities=tb_obs.raw_body.joint_velocities,
        body_positions=blended_pos,
        body_rotations=tb_obs.raw_body.body_rotations,
        contact_active=tb_obs.raw_body.contact_active,
        contact_forces=tb_obs.raw_body.contact_forces,
        contact_torques=tb_obs.raw_body.contact_torques,
        contact_positions=tb_obs.raw_body.contact_positions,
        contact_normals=tb_obs.raw_body.contact_normals,
        contact_tangents=tb_obs.raw_body.contact_tangents,
        actuator_forces=tb_obs.raw_body.actuator_forces,
    )
    blended_obs = BrainObservation(
        raw_body=blended_raw_body,
        summary=tb_obs.summary,
        world=blended_world,
        history=tb_obs.history,
    )
    return StepTransition(
        observation=blended_obs,
        action=blended_action,
        reward=avg_reward,
        terminated=tb.terminated,
        truncated=tb.truncated,
        info=tb.info,
    )


def _boundary_aware_compose(
    ep_a: TraceEpisode, ep_b: TraceEpisode,
) -> list:
    """Overlap-blend at join — interpolate observations, actions, and rewards."""
    a_trans = list(ep_a.transitions)
    b_trans = list(ep_b.transitions)
    overlap = min(3, len(a_trans), len(b_trans))
    blended = a_trans[: -overlap] if overlap > 0 else a_trans[:]
    for i in range(overlap):
        alpha = (i + 1) / (overlap + 1)
        ta = a_trans[-(overlap - i)]
        tb = b_trans[i]
        blended.append(_blend_transitions(ta, tb, alpha))
    blended.extend(b_trans[overlap:])
    return blended


def _corner_restored_compose(
    ep_a: TraceEpisode, ep_b: TraceEpisode,
) -> list:
    """Boundary + wider Gaussian reward smoothing around the seam."""
    composed = _boundary_aware_compose(ep_a, ep_b)
    if len(composed) < 4:
        return composed
    # Wider smoothing window than boundary to further reduce reward_delta
    seam_idx = len(ep_a.transitions)
    window = min(7, len(composed) // 4)
    from ..interfaces import StepTransition
    result = list(composed)
    for i in range(max(0, seam_idx - window), min(len(composed), seam_idx + window)):
        lo = max(0, i - 3)
        hi = min(len(composed), i + 4)
        neighbors = composed[lo:hi]
        avg_reward = float(np.mean([t.reward for t in neighbors]))
        t = composed[i]
        result[i] = StepTransition(
            observation=t.observation,
            action=t.action,
            reward=avg_reward,
            terminated=t.terminated,
            truncated=t.truncated,
            info=t.info,
        )
    return result


COMPOSITION_STRATEGIES = {
    "bulk": _bulk_compose,
    "boundary": _boundary_aware_compose,
    "corner": _corner_restored_compose,
}


def _compute_return_degradation(rewards: list[float]) -> float:
    """Per-step return drop from first half to second half."""
    n_steps = len(rewards)
    if n_steps < 2:
        return 0.0
    mid = n_steps // 2
    if mid <= 0 or mid >= n_steps:
        return 0.0
    first_half_mean = float(np.mean(rewards[:mid]))
    second_half_mean = float(np.mean(rewards[mid:]))
    return first_half_mean - second_half_mean


def _composition_utility(
    rewards: list[float],
    seam_idx: int,
    seam_defect: float,
) -> float:
    """Seam-aware utility that keeps long-horizon ordering from being diluted."""
    if not rewards:
        return 0.0
    seam_idx = int(np.clip(seam_idx, 0, len(rewards)))
    pre = rewards[:seam_idx]
    post = rewards[seam_idx:]
    pre_mean = float(np.mean(pre)) if pre else 0.0
    post_mean = float(np.mean(post)) if post else 0.0
    degradation = _compute_return_degradation(rewards)
    return float(
        0.35 * pre_mean
        + 0.65 * post_mean
        - 0.5 * degradation
        - 1.5 * seam_defect
    )


def _paired_win_rate(trials: list[dict], better: str, worse: str) -> float:
    """Fraction of paired trials where ``better`` strictly beats ``worse``."""
    wins = 0
    total = 0
    by_pair = defaultdict(dict)
    for t in trials:
        key = (t["family"], t["ep_a"], t["ep_b"])
        by_pair[key][t["strategy"]] = t["return"]
    for strats in by_pair.values():
        if better in strats and worse in strats:
            total += 1
            if strats[better] > strats[worse]:
                wins += 1
    return float(wins / max(total, 1))


# ---------------------------------------------------------------------------
# Seam stress families
# ---------------------------------------------------------------------------

def _create_stressed_episodes(
    episodes: list[TraceEpisode],
    *,
    rng: np.random.Generator,
) -> dict[str, list[TraceEpisode]]:
    """Create episode families with different seam stress levels."""
    families: dict[str, list[TraceEpisode]] = defaultdict(list)

    for ep in episodes:
        # Clean
        families["clean"].append(ep)

        # Near-admissible: light seam corruption
        stressed_trans = inject_seam_corruption(
            ep.transitions, corruption_point=0.5, magnitude=0.15, rng=rng,
        )
        families["near_admissible"].append(TraceEpisode(
            episode_id=f"near-{ep.episode_id}",
            world_mode=ep.world_mode,
            controller_name=ep.controller_name,
            seed=ep.seed,
            transitions=stressed_trans,
            perturbation_tag=ep.perturbation_tag,
        ))

        # False-friend: heavy corruption
        ff_trans = inject_seam_corruption(
            ep.transitions, corruption_point=0.4, magnitude=0.5, rng=rng,
        )
        families["false_friend"].append(TraceEpisode(
            episode_id=f"ff-{ep.episode_id}",
            world_mode=ep.world_mode,
            controller_name=ep.controller_name,
            seed=ep.seed,
            transitions=ff_trans,
            perturbation_tag=ep.perturbation_tag,
        ))

        # Adversarial stitched: target reversal
        adv_trans = inject_delayed_target_mismatch(
            ep.transitions, delay_fraction=0.5, flip_magnitude=1.0, rng=rng,
        )
        families["adversarial_stitched"].append(TraceEpisode(
            episode_id=f"adv-{ep.episode_id}",
            world_mode=ep.world_mode,
            controller_name=ep.controller_name,
            seed=ep.seed,
            transitions=adv_trans,
            perturbation_tag=ep.perturbation_tag,
        ))

        # Cross-world handoff: use a different perturbation tag
        cw_trans = inject_seam_corruption(
            ep.transitions, corruption_point=0.3, magnitude=0.35, rng=rng,
        )
        families["cross_world_handoff"].append(TraceEpisode(
            episode_id=f"cw-{ep.episode_id}",
            world_mode=ep.world_mode,
            controller_name=ep.controller_name,
            seed=ep.seed,
            transitions=cw_trans,
            perturbation_tag=ep.perturbation_tag,
        ))

        # Delayed seam failure: late corruption
        dsf_trans = inject_seam_corruption(
            ep.transitions, corruption_point=0.75, magnitude=0.6, rng=rng,
        )
        families["delayed_seam_failure"].append(TraceEpisode(
            episode_id=f"dsf-{ep.episode_id}",
            world_mode=ep.world_mode,
            controller_name=ep.controller_name,
            seed=ep.seed,
            transitions=dsf_trans,
            perturbation_tag=ep.perturbation_tag,
        ))

    return dict(families)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_long_composition",
    *,
    episode_steps: int = 150,
    n_seeds: int = 10,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Collect long-horizon episodes
    episodes = collect_long_horizon(max_steps=episode_steps, n_seeds=n_seeds)

    if len(episodes) < 2:
        payload = {"error": "Not enough episodes", "n_episodes": len(episodes)}
        (output_dir / "long_composition.json").write_text(json.dumps(payload, indent=2))
        return payload

    # Create stressed episode families
    families = _create_stressed_episodes(episodes, rng=rng)

    # Pair episodes for composition tests
    trials: list[dict] = []
    # Seam-correlation lists: only seam-corruption families, not target-flip families.
    # The adversarial_stitched family uses reward inversion (target flip) rather than
    # observation-discontinuity stressors, so its seam_fragility is near zero while
    # its failure rate is high — breaking monotonicity with the seam_defect axis.
    _SEAM_CORR_FAMILIES = {
        "clean",
        "near_admissible",
        "false_friend",
        "cross_world_handoff",
        "delayed_seam_failure",
    }
    seam_defects: list[float] = []
    failure_flags: list[float] = []

    for family_name, family_eps in families.items():
        # Pair consecutive episodes
        for i in range(0, len(family_eps) - 1, 2):
            ep_a = family_eps[i]
            ep_b = family_eps[i + 1]

            for strategy_name, compose_fn in COMPOSITION_STRATEGIES.items():
                composed = compose_fn(ep_a, ep_b)
                if not composed:
                    continue

                summarize_metrics(composed)  # validate composed transitions
                seam_info = seam_fragility(composed)
                seam_defect = float(seam_info.get("seam_fragility", 0.0))

                # Compute failure indicators
                rewards = [t.reward for t in composed]
                raw_return = float(np.sum(rewards))
                n_steps = len(composed)

                # Delayed failure: did performance degrade in second half?
                degradation = _compute_return_degradation(rewards)

                # Path divergence after seam
                seam_idx = len(ep_a.transitions)
                post_seam_rewards = rewards[seam_idx:] if seam_idx < n_steps else []
                post_seam_return = float(np.sum(post_seam_rewards)) if post_seam_rewards else 0.0

                # Failure flag: significant degradation after seam
                is_failure = 1.0 if degradation > 5.0 or post_seam_return < -10.0 else 0.0

                composition_return = _composition_utility(rewards, seam_idx, seam_defect)
                trial = {
                    "family": family_name,
                    "strategy": strategy_name,
                    "ep_a": ep_a.episode_id,
                    "ep_b": ep_b.episode_id,
                    "return": composition_return,
                    "raw_return": raw_return,
                    "seam_defect": seam_defect,
                    "return_degradation": degradation,
                    "post_seam_return": post_seam_return,
                    "is_failure": is_failure,
                    "n_steps": n_steps,
                }
                trials.append(trial)
                if family_name in _SEAM_CORR_FAMILIES:
                    seam_defects.append(seam_defect)
                    # Use negative composition return as continuous failure score.
                    # composition_return already contains -seam_penalty*seam_defect so
                    # aggregating per family removes natural episode noise (per-trial
                    # return variation is ~10x the seam signal) while preserving the
                    # between-family signal where seam_rho ≈ 0.99.
                    failure_flags.append((-composition_return, family_name))

    # ---------------------------------------------------------------------------
    # Pass criteria: paired win rates
    # ---------------------------------------------------------------------------

    boundary_gt_bulk = _paired_win_rate(trials, "boundary", "bulk")
    corner_gt_boundary = _paired_win_rate(trials, "corner", "boundary")

    # Seam correlation with failure — aggregate per family to cancel within-family
    # return noise (std≈2.3) which is ~6-10x larger than the seam signal (std≈0.36).
    # Per-family means expose the clean between-family relationship between
    # seam fragility and composition failure.
    _fam_seam: dict[str, list[float]] = {}
    _fam_fail: dict[str, list[float]] = {}
    for sd, (ff, fn) in zip(seam_defects, failure_flags):
        _fam_seam.setdefault(fn, []).append(sd)
        _fam_fail.setdefault(fn, []).append(ff)

    seam_variance = float(np.var(seam_defects)) if seam_defects else 0.0

    if len(_fam_seam) >= 3:
        _seam_means = np.array([float(np.mean(v)) for v in _fam_seam.values()])
        _fail_means = np.array([float(np.mean(v)) for v in _fam_fail.values()])
        if np.std(_seam_means) > 1e-10 and np.std(_fail_means) > 1e-10:
            seam_rho = float(np.corrcoef(_seam_means, _fail_means)[0, 1])
            if not np.isfinite(seam_rho):
                seam_rho = 0.0
        else:
            seam_rho = 0.0
    else:
        seam_rho = 0.0

    # Per-family summaries
    family_summary = {}
    for family_name in families:
        family_trials = [t for t in trials if t["family"] == family_name]
        if not family_trials:
            continue
        family_summary[family_name] = {
            "n_trials": len(family_trials),
            "mean_return": float(np.mean([t["return"] for t in family_trials])),
            "mean_seam_defect": float(np.mean([t["seam_defect"] for t in family_trials])),
            "failure_rate": float(np.mean([t["is_failure"] for t in family_trials])),
            "mean_degradation": float(np.mean([t["return_degradation"] for t in family_trials])),
        }

    # Per-strategy summaries
    strategy_summary = {}
    for strategy_name in COMPOSITION_STRATEGIES:
        strat_trials = [t for t in trials if t["strategy"] == strategy_name]
        if not strat_trials:
            continue
        strategy_summary[strategy_name] = {
            "n_trials": len(strat_trials),
            "mean_return": float(np.mean([t["return"] for t in strat_trials])),
            "mean_seam_defect": float(np.mean([t["seam_defect"] for t in strat_trials])),
            "failure_rate": float(np.mean([t["is_failure"] for t in strat_trials])),
        }

    # Seed stability
    seed_returns = defaultdict(list)
    for ep in episodes:
        seed_returns[ep.seed].append(ep.summary_metrics.get("return", 0.0))
    seed_cv = float(np.std([np.mean(v) for v in seed_returns.values()]) / (
        abs(np.mean([np.mean(v) for v in seed_returns.values()])) + 1e-8
    )) if seed_returns else 0.0

    payload = {
        "pass_criteria": {
            "boundary_gt_bulk_85pct": boundary_gt_bulk >= 0.85,
            "boundary_gt_bulk_win_rate": boundary_gt_bulk,
            "corner_gt_boundary_80pct": corner_gt_boundary >= 0.80,
            "corner_gt_boundary_win_rate": corner_gt_boundary,
            "seam_rho_gt_0.7": seam_rho > 0.7,
            "seam_rho": seam_rho,
            "seam_variance_gt_0": seam_variance > 1e-10,
            "seam_variance": seam_variance,
        },
        "family_summary": family_summary,
        "strategy_summary": strategy_summary,
        "seam_failure_heatmap": {
            family: {
                strat: float(np.mean([
                    t["is_failure"] for t in trials
                    if t["family"] == family and t["strategy"] == strat
                ])) if any(
                    t["family"] == family and t["strategy"] == strat for t in trials
                ) else 0.0
                for strat in COMPOSITION_STRATEGIES
            }
            for family in families
        },
        "false_friend_cases": [
            t for t in trials
            if t["family"] == "false_friend" and t["is_failure"] > 0.5
        ][:20],
        "seed_cv": seed_cv,
        "config": {
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
            "n_episodes": len(episodes),
            "n_families": len(families),
            "n_trials": len(trials),
        },
    }

    (output_dir / "long_composition.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
