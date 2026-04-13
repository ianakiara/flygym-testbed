"""Experiment 1 — Canonical Composition Law.

Proves or kills:  R₁ # R₂ admissible ⟺ δ_seam ≤ ε

Three composition strategies (bulk-only, boundary-aware, corner-restored)
are tested on baseline, cross-world stitched, and adversarial trajectories.
Includes adversarial cases where local metrics look good but global
composition fails.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import MemoryController, ReducedDescendingController, ReflexOnlyController
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..metrics import (
    interoperability_score,
    seam_fragility,
    summarize_metrics,
)
from ..worlds import NativePhysicalWorld, SimplifiedEmbodiedWorld
from .benchmark_harness import run_episode


# ---------------------------------------------------------------------------
# Composition strategies
# ---------------------------------------------------------------------------

def _bulk_compose(
    transitions_a: list, transitions_b: list
) -> list:
    """Bulk-only: naive concatenation with no seam handling."""
    return list(transitions_a) + list(transitions_b)


def _boundary_aware_compose(
    transitions_a: list, transitions_b: list, *, overlap: int = 2
) -> list:
    """Boundary-aware: overlap-blend at the join."""
    a = list(transitions_a)
    b = list(transitions_b)
    if len(a) < overlap or len(b) < overlap:
        return a + b
    blended = []
    for i in range(overlap):
        alpha = (i + 1) / (overlap + 1)
        t_a = a[-(overlap - i)]
        t_b = b[i]
        blended_action = type(t_a.action)(
            move_intent=float((1 - alpha) * t_a.action.move_intent + alpha * t_b.action.move_intent),
            turn_intent=float((1 - alpha) * t_a.action.turn_intent + alpha * t_b.action.turn_intent),
            speed_modulation=float((1 - alpha) * t_a.action.speed_modulation + alpha * t_b.action.speed_modulation),
            stabilization_priority=float(
                (1 - alpha) * t_a.action.stabilization_priority + alpha * t_b.action.stabilization_priority
            ),
            target_bias=(
                float((1 - alpha) * t_a.action.target_bias[0] + alpha * t_b.action.target_bias[0]),
                float((1 - alpha) * t_a.action.target_bias[1] + alpha * t_b.action.target_bias[1]),
            ),
        )
        from ..interfaces import StepTransition
        blended.append(StepTransition(
            observation=t_b.observation,
            action=blended_action,
            reward=float((1 - alpha) * t_a.reward + alpha * t_b.reward),
            terminated=t_b.terminated,
            truncated=t_b.truncated,
            info={**t_a.info, **t_b.info, "blended": True},
        ))
    return a[:-overlap] + blended + b[overlap:]


def _corner_restored_compose(
    transitions_a: list, transitions_b: list, *, overlap: int = 2
) -> list:
    """Corner-restored: boundary-aware + seam defect correction at corners."""
    composed = _boundary_aware_compose(transitions_a, transitions_b, overlap=overlap)
    if len(composed) < 3:
        return composed
    seam_start = max(0, len(transitions_a) - overlap - 1)
    seam_end = min(len(composed), len(transitions_a) + overlap + 1)
    for i in range(seam_start, seam_end):
        if i <= 0 or i >= len(composed) - 1:
            continue
        prev_r = composed[i - 1].reward
        next_r = composed[min(i + 1, len(composed) - 1)].reward
        curr_r = composed[i].reward
        expected = 0.5 * (prev_r + next_r)
        if abs(curr_r - expected) > abs(expected) * 0.5 + 0.1:
            from ..interfaces import StepTransition
            composed[i] = StepTransition(
                observation=composed[i].observation,
                action=composed[i].action,
                reward=expected,
                terminated=composed[i].terminated,
                truncated=composed[i].truncated,
                info={**composed[i].info, "corner_restored": True},
            )
    return composed


# ---------------------------------------------------------------------------
# Adversarial episode generators
# ---------------------------------------------------------------------------

def _make_env(world_mode: str, *, noise: float = 0.02):
    cfg = EnvConfig(avatar_noise_scale=noise)
    body = BodylessBodyLayer(config=BodyLayerConfig())
    if world_mode == "avatar_remapped":
        return FlyAvatarEnv(body=body, env_config=cfg)
    if world_mode == "native_physical":
        return FlyBodyWorldEnv(body=body, world=NativePhysicalWorld(config=cfg), config=cfg)
    if world_mode == "simplified_embodied":
        return FlyBodyWorldEnv(body=body, world=SimplifiedEmbodiedWorld(config=cfg), config=cfg)
    raise ValueError(world_mode)


def _collect_baseline_episodes(*, max_steps: int = 24, seeds: list[int] | None = None):
    seeds = seeds or [0, 1, 2]
    controllers = {
        "reflex": ReflexOnlyController(),
        "reduced": ReducedDescendingController(),
        "memory": MemoryController(),
    }
    worlds = ["avatar_remapped", "native_physical", "simplified_embodied"]
    episodes = []
    for world_mode in worlds:
        for name, ctrl in controllers.items():
            for seed in seeds:
                env = _make_env(world_mode)
                transitions = run_episode(env, ctrl, seed=seed, max_steps=max_steps)
                episodes.append({
                    "controller": name,
                    "world_mode": world_mode,
                    "seed": seed,
                    "transitions": transitions,
                    "tag": "baseline",
                })
    return episodes


def _adversarial_reversed_target(episode_data: dict) -> dict:
    """Reversed target bias mid-episode: local metrics look good per-half
    but global composition fails because intent flips."""
    transitions = list(episode_data["transitions"])
    mid = len(transitions) // 2
    from ..interfaces import StepTransition, DescendingCommand
    modified = []
    for i, t in enumerate(transitions):
        if i >= mid and hasattr(t.action, 'target_bias'):
            new_action = DescendingCommand(
                move_intent=-t.action.move_intent,
                turn_intent=-t.action.turn_intent,
                speed_modulation=t.action.speed_modulation,
                stabilization_priority=t.action.stabilization_priority,
                target_bias=(-t.action.target_bias[0], -t.action.target_bias[1]),
            )
            modified.append(StepTransition(
                observation=t.observation, action=new_action,
                reward=t.reward, terminated=t.terminated,
                truncated=t.truncated, info={**t.info, "adversarial": "reversed_target"},
            ))
        else:
            modified.append(t)
    return {**episode_data, "transitions": modified, "tag": "adversarial_reversed_target"}


def _adversarial_conflicting_worlds(ep_a: dict, ep_b: dict) -> dict:
    """Splice first half from world A with second half from world B."""
    t_a = ep_a["transitions"]
    t_b = ep_b["transitions"]
    mid_a = len(t_a) // 2
    mid_b = len(t_b) // 2
    spliced = list(t_a[:mid_a]) + list(t_b[mid_b:])
    return {
        "controller": ep_a["controller"],
        "world_mode": f"{ep_a['world_mode']}+{ep_b['world_mode']}",
        "seed": ep_a["seed"],
        "transitions": spliced,
        "tag": "adversarial_conflicting_worlds",
    }


def _adversarial_partial_splice(episode_data: dict) -> dict:
    """Keep only every other transition — creates temporal incoherence."""
    transitions = list(episode_data["transitions"])
    spliced = transitions[::2]
    return {**episode_data, "transitions": spliced, "tag": "adversarial_partial_splice"}


# ---------------------------------------------------------------------------
# Composition evaluation
# ---------------------------------------------------------------------------

def _evaluate_composition(composed_transitions: list, original_a: list, original_b: list) -> dict:
    """Evaluate quality of a composed trajectory."""
    if len(composed_transitions) < 2:
        return {"L_hyb": 0.0, "seam_defect": 1.0, "failure": True,
                "return_degradation": 0.0, "trajectory_divergence": 0.0, "repair_cost": 0.0}

    metrics_composed = summarize_metrics(composed_transitions)
    metrics_a = summarize_metrics(original_a) if original_a else {}
    metrics_b = summarize_metrics(original_b) if original_b else {}

    seam_info = seam_fragility(composed_transitions)
    seam_def = float(seam_info.get("seam_fragility", 0.0))

    avg_return_orig = 0.5 * (metrics_a.get("return", 0.0) + metrics_b.get("return", 0.0))
    return_composed = metrics_composed.get("return", 0.0)
    return_degradation = avg_return_orig - return_composed if avg_return_orig != 0 else 0.0

    traj_div = 0.0
    if len(original_a) > 0 and len(original_b) > 0:
        interop = interoperability_score(original_a, composed_transitions[:len(original_a)])
        traj_div = 1.0 - float(interop.get("interoperability_score", 1.0))

    L_hyb = float(np.clip(
        metrics_composed.get("return", 0.0) / max(abs(avg_return_orig), 1.0) - 0.5 * seam_def,
        -1.0, 1.0,
    ))

    failure = seam_def > 0.3 or metrics_composed.get("success", 0.0) < 0.5 * max(
        metrics_a.get("success", 0.0), metrics_b.get("success", 0.0), 0.01
    )

    repair_cost = float(np.clip(seam_def + 0.3 * traj_div, 0.0, 1.0))

    return {
        "L_hyb": L_hyb,
        "seam_defect": seam_def,
        "failure": bool(failure),
        "return_degradation": return_degradation,
        "trajectory_divergence": traj_div,
        "repair_cost": repair_cost,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_canonical_composition_law",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_episodes = _collect_baseline_episodes(max_steps=24)

    # Build adversarial cases
    adversarial_episodes = []
    for ep in baseline_episodes[:6]:
        adversarial_episodes.append(_adversarial_reversed_target(ep))
        adversarial_episodes.append(_adversarial_partial_splice(ep))

    # Cross-world splices
    by_ctrl = defaultdict(list)
    for ep in baseline_episodes:
        by_ctrl[ep["controller"]].append(ep)
    for ctrl_eps in by_ctrl.values():
        for i in range(len(ctrl_eps)):
            for j in range(i + 1, min(i + 3, len(ctrl_eps))):
                if ctrl_eps[i]["world_mode"] != ctrl_eps[j]["world_mode"]:
                    adversarial_episodes.append(
                        _adversarial_conflicting_worlds(ctrl_eps[i], ctrl_eps[j])
                    )

    all_episodes = baseline_episodes + adversarial_episodes

    strategies = {
        "bulk_only": _bulk_compose,
        "boundary_aware": _boundary_aware_compose,
        "corner_restored": _corner_restored_compose,
    }

    results_by_strategy = {name: [] for name in strategies}
    false_friend_cases = []

    for ep in all_episodes:
        transitions = ep["transitions"]
        if len(transitions) < 4:
            continue
        mid = len(transitions) // 2
        part_a = transitions[:mid]
        part_b = transitions[mid:]

        for strategy_name, compose_fn in strategies.items():
            composed = compose_fn(part_a, part_b)
            eval_result = _evaluate_composition(composed, part_a, part_b)
            row = {
                "controller": ep["controller"],
                "world_mode": ep["world_mode"],
                "seed": ep.get("seed", 0),
                "tag": ep["tag"],
                "n_transitions": len(composed),
                **eval_result,
            }
            results_by_strategy[strategy_name].append(row)

            # Detect false friends: local looks good but global fails
            if not eval_result["failure"]:
                continue
            local_seam_a = seam_fragility(part_a).get("seam_fragility", 0.0)
            local_seam_b = seam_fragility(part_b).get("seam_fragility", 0.0)
            if local_seam_a < 0.15 and local_seam_b < 0.15 and eval_result["seam_defect"] > 0.2:
                false_friend_cases.append({
                    **row,
                    "strategy": strategy_name,
                    "local_seam_a": local_seam_a,
                    "local_seam_b": local_seam_b,
                })

    # Compute pass/fail criteria
    def _stats(rows):
        if not rows:
            return {"n": 0, "mean_L_hyb": 0.0, "mean_seam_defect": 0.0, "failure_rate": 0.0}
        return {
            "n": len(rows),
            "mean_L_hyb": float(np.mean([r["L_hyb"] for r in rows])),
            "mean_seam_defect": float(np.mean([r["seam_defect"] for r in rows])),
            "failure_rate": float(np.mean([r["failure"] for r in rows])),
            "mean_return_degradation": float(np.mean([r["return_degradation"] for r in rows])),
            "mean_trajectory_divergence": float(np.mean([r["trajectory_divergence"] for r in rows])),
            "mean_repair_cost": float(np.mean([r["repair_cost"] for r in rows])),
        }

    summary_by_strategy = {name: _stats(rows) for name, rows in results_by_strategy.items()}

    # Seam-defect vs failure correlation
    all_rows = [r for rows in results_by_strategy.values() for r in rows]
    seam_vals = np.array([r["seam_defect"] for r in all_rows])
    fail_vals = np.array([float(r["failure"]) for r in all_rows])
    if len(seam_vals) > 2 and np.std(seam_vals) > 1e-8 and np.std(fail_vals) > 1e-8:
        seam_failure_correlation = float(np.corrcoef(seam_vals, fail_vals)[0, 1])
    else:
        seam_failure_correlation = 0.0

    # Pass criteria evaluation
    bulk_fr = summary_by_strategy["bulk_only"]["failure_rate"]
    boundary_fr = summary_by_strategy["boundary_aware"]["failure_rate"]
    corner_fr = summary_by_strategy["corner_restored"]["failure_rate"]

    boundary_beats_bulk = (1.0 - boundary_fr) >= 0.85 * (1.0 - bulk_fr) if bulk_fr < 1.0 else boundary_fr < bulk_fr
    corner_beats_boundary = (1.0 - corner_fr) >= 0.80 * (1.0 - boundary_fr) if boundary_fr < 1.0 else corner_fr < boundary_fr
    seam_predicts_failure = abs(seam_failure_correlation) > 0.7

    # Phase diagram: tag × strategy
    phase_diagram = {}
    for tag in sorted({r["tag"] for r in all_rows}):
        phase_diagram[tag] = {}
        for strategy_name in strategies:
            tag_rows = [r for r in results_by_strategy[strategy_name] if r["tag"] == tag]
            phase_diagram[tag][strategy_name] = _stats(tag_rows)

    # Seam failure heatmap: world_mode × strategy
    seam_heatmap = {}
    for wm in sorted({r["world_mode"] for r in all_rows}):
        seam_heatmap[wm] = {}
        for strategy_name in strategies:
            wm_rows = [r for r in results_by_strategy[strategy_name] if r["world_mode"] == wm]
            seam_heatmap[wm][strategy_name] = float(np.mean([r["seam_defect"] for r in wm_rows])) if wm_rows else 0.0

    payload = {
        "summary_by_strategy": summary_by_strategy,
        "seam_failure_correlation": seam_failure_correlation,
        "pass_criteria": {
            "boundary_beats_bulk_85pct": boundary_beats_bulk,
            "corner_beats_boundary_80pct": corner_beats_boundary,
            "seam_predicts_failure_rho_gt_0.7": seam_predicts_failure,
        },
        "phase_diagram": phase_diagram,
        "seam_heatmap": seam_heatmap,
        "false_friend_cases": false_friend_cases[:20],
        "n_total_evaluations": len(all_rows),
        "n_baseline": len(baseline_episodes),
        "n_adversarial": len(adversarial_episodes),
    }

    (output_dir / "canonical_composition_law.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
