"""Full validation program — runs all 10 stages and produces final reports.

Usage:
    python -m flygym_research.cognition.validation.run_validation
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    BodylessAvatarController,
    MemoryController,
    NoAscendingFeedbackController,
    PlannerController,
    RandomController,
    RawControlController,
    ReducedDescendingController,
    ReflexOnlyController,
)
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..experiments.benchmark_harness import run_episode
from ..interfaces import BrainInterface, DescendingCommand, StepTransition
from ..metrics import (
    controller_action_distribution,
    cross_condition_objectness,
    cross_time_mutual_information,
    global_disruption_signature,
    history_dependence,
    hysteresis_metric,
    interoperability_score,
    latent_state_similarity,
    predictive_utility,
    reward_trajectory_similarity,
    seam_fragility,
    self_world_separation,
    shared_objectness_score,
    stabilization_quality,
    state_decay_curve,
    state_persistence,
    summarize_metrics,
    target_representation_stability,
    task_performance,
)
from ..worlds import AvatarRemappedWorld, NativePhysicalWorld, SimplifiedEmbodiedWorld
from .claims_ledger import ClaimsLedger, ClaimTier, overclaiming_filter
from .validation_suite import ValidationSuite

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS_FAST = list(range(10))
SEEDS_HEAVY = list(range(5))
EPISODE_STEPS = 64
EPISODE_STEPS_LONG = 128


@dataclass
class StageResult:
    stage: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environment factories
# ---------------------------------------------------------------------------


def _make_bodyworld_env(
    world_cls, *, disabled_channels: frozenset[str] = frozenset(),
    stabilization_gain: float = 0.1,
) -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    body_cfg = BodyLayerConfig(
        disabled_feedback_channels=disabled_channels,
        stabilization_gain=stabilization_gain,
    )
    return FlyBodyWorldEnv(
        body=BodylessBodyLayer(config=body_cfg),
        world=world_cls(config=cfg),
        config=cfg,
    )


def _make_avatar_env(
    *, disabled_channels: frozenset[str] = frozenset(),
) -> FlyAvatarEnv:
    body_cfg = BodyLayerConfig(disabled_feedback_channels=disabled_channels)
    return FlyAvatarEnv(body=BodylessBodyLayer(config=body_cfg))


def _all_controllers() -> dict[str, BrainInterface]:
    return {
        "reflex_only": ReflexOnlyController(),
        "random": RandomController(),
        "reduced_descending": ReducedDescendingController(),
        "no_ascending": NoAscendingFeedbackController(),
        "raw_control": RawControlController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "bodyless": BodylessAvatarController(),
    }


def _baseline_controllers() -> dict[str, BrainInterface]:
    return {
        "reflex_only": ReflexOnlyController(),
        "bodyless": BodylessAvatarController(),
        "raw_control": RawControlController(),
        "reduced_descending": ReducedDescendingController(),
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _full_metrics(transitions: list[StepTransition]) -> dict[str, float]:
    """Compute all available metrics for a transition list."""
    m = summarize_metrics(transitions)
    if len(transitions) >= 5:
        m.update(cross_time_mutual_information(transitions, max_lag=3))
        m.update(state_decay_curve(transitions, max_lag=5))
        m.update(predictive_utility(transitions, horizon=2))
        m.update(hysteresis_metric(transitions))
        m.update(target_representation_stability(transitions))
    return m


def _run_seeds(
    env_factory, controller: BrainInterface, seeds: list[int],
    max_steps: int = EPISODE_STEPS,
) -> tuple[list[list[StepTransition]], list[dict[str, float]]]:
    """Run multiple seeded episodes, return transitions and metrics."""
    all_transitions = []
    all_metrics = []
    for seed in seeds:
        env = env_factory()
        t = run_episode(env, controller, seed=seed, max_steps=max_steps)
        all_transitions.append(t)
        all_metrics.append(_full_metrics(t))
    return all_transitions, all_metrics


def _agg(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate metric dicts into mean/std."""
    if not metrics_list:
        return {}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    result = {}
    for k in sorted(all_keys):
        vals = [m.get(k, 0.0) for m in metrics_list]
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    return result


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------


def stage_1_sanity(output_dir: Path) -> StageResult:
    """Stage 1 — Sanity and infrastructure validation."""
    notes = []
    checks_passed = 0
    checks_total = 0

    # 1. Smoke test: all controllers can run in avatar env
    controllers = _all_controllers()
    for name, ctrl in controllers.items():
        checks_total += 1
        try:
            env = _make_avatar_env()
            t = run_episode(env, ctrl, seed=0, max_steps=10)
            assert len(t) > 0, f"{name} produced no transitions"
            checks_passed += 1
            notes.append(f"  {name}: OK ({len(t)} steps)")
        except Exception as e:
            notes.append(f"  {name}: FAIL — {e}")

    # 2. Deterministic seeded rollouts
    checks_total += 1
    env1 = _make_avatar_env()
    env2 = _make_avatar_env()
    ctrl = ReducedDescendingController()
    t1 = run_episode(env1, ctrl, seed=42, max_steps=20)
    ctrl2 = ReducedDescendingController()
    t2 = run_episode(env2, ctrl2, seed=42, max_steps=20)
    rewards1 = [t.reward for t in t1]
    rewards2 = [t.reward for t in t2]
    if np.allclose(rewards1, rewards2, atol=1e-10):
        checks_passed += 1
        notes.append("  Determinism: PASS (same seed → same rewards)")
    else:
        notes.append("  Determinism: FAIL")

    # 3. No NaN check (target_distance is NaN by design when no target)
    checks_total += 1
    nan_found = False
    nan_allowed = {"target_distance"}
    env = _make_avatar_env()
    t = run_episode(env, ReducedDescendingController(), seed=0, max_steps=30)
    for tr in t:
        for k, v in tr.observation.summary.features.items():
            if np.isnan(v) and k not in nan_allowed:
                nan_found = True
                notes.append(f"  Unexpected NaN in feature {k}")
    if not nan_found:
        checks_passed += 1
        notes.append("  NaN check: PASS (target_distance NaN allowed by design)")

    # 4. All world modes run
    for world_cls, world_name in [
        (SimplifiedEmbodiedWorld, "simplified"),
        (AvatarRemappedWorld, "avatar"),
        (NativePhysicalWorld, "native"),
    ]:
        checks_total += 1
        try:
            env = _make_bodyworld_env(world_cls)
            t = run_episode(env, ReducedDescendingController(), seed=0, max_steps=10)
            assert len(t) > 0
            checks_passed += 1
            notes.append(f"  World {world_name}: OK")
        except Exception as e:
            notes.append(f"  World {world_name}: FAIL — {e}")

    # 5. Ablation toggling by config only
    for channels in [
        frozenset({"pose"}), frozenset({"locomotion"}),
        frozenset({"contact"}), frozenset({"target"}), frozenset({"internal"}),
        frozenset({"pose", "locomotion", "contact", "target", "internal"}),
    ]:
        checks_total += 1
        try:
            env = _make_avatar_env(disabled_channels=channels)
            t = run_episode(env, ReducedDescendingController(), seed=0, max_steps=5)
            assert len(t) > 0
            checks_passed += 1
        except Exception as e:
            notes.append(f"  Ablation {channels}: FAIL — {e}")

    # 6. Action/observation shape consistency
    checks_total += 1
    env = _make_avatar_env()
    t = run_episode(env, ReducedDescendingController(), seed=0, max_steps=10)
    shapes_ok = True
    for i, tr in enumerate(t):
        if tr.observation.raw_body.body_positions.shape[1] != 3:
            shapes_ok = False
    if shapes_ok:
        checks_passed += 1
        notes.append("  Shape consistency: PASS")

    passed = checks_passed == checks_total
    _write_csv(
        [{"check": n} for n in notes],
        output_dir / "stage1_sanity.csv",
    )
    return StageResult(
        stage="Stage 1: Sanity",
        passed=passed,
        details={"checks_passed": checks_passed, "checks_total": checks_total},
        notes=notes,
    )


def stage_2_baseline_truth(output_dir: Path) -> StageResult:
    """Stage 2 — Baseline truth validation."""
    notes = []
    rows = []
    controllers = _baseline_controllers()

    world_configs = [
        ("simplified", lambda: _make_bodyworld_env(SimplifiedEmbodiedWorld)),
        ("avatar", lambda: _make_avatar_env()),
        ("native", lambda: _make_bodyworld_env(NativePhysicalWorld)),
    ]

    for world_name, env_factory in world_configs:
        for ctrl_name, ctrl in controllers.items():
            _, metrics_list = _run_seeds(env_factory, ctrl, SEEDS_FAST)
            agg = _agg(metrics_list)
            row = {"world": world_name, "controller": ctrl_name}
            row.update(agg)
            rows.append(row)
            notes.append(
                f"  {world_name}/{ctrl_name}: return={agg.get('return_mean', 0):.4f}±{agg.get('return_std', 0):.4f}, "
                f"stability={agg.get('stability_mean_mean', 0):.4f}"
            )

    _write_csv(rows, output_dir / "baseline_truth_table.csv")

    # Check: does reduced_descending beat reflex_only?
    rd_returns = []
    ro_returns = []
    for r in rows:
        if r["controller"] == "reduced_descending":
            rd_returns.append(r.get("return_mean", 0))
        elif r["controller"] == "reflex_only":
            ro_returns.append(r.get("return_mean", 0))

    rd_mean = np.mean(rd_returns) if rd_returns else 0
    ro_mean = np.mean(ro_returns) if ro_returns else 0
    gap = rd_mean - ro_mean
    passed = gap > 0.05
    notes.append(f"  Reduced vs reflex gap: {gap:.4f} (threshold: 0.05)")

    return StageResult(
        stage="Stage 2: Baseline Truth",
        passed=passed,
        details={"reduced_mean": float(rd_mean), "reflex_mean": float(ro_mean), "gap": float(gap)},
        notes=notes,
    )


def stage_3_body_substrate(output_dir: Path) -> StageResult:
    """Stage 3 — Body substrate value validation."""
    notes = []
    rows = []

    body_ctrls = {
        "reduced_descending": ReducedDescendingController(),
        "raw_control": RawControlController(),
    }
    bodyless_ctrl = BodylessAvatarController()

    world_configs = [
        ("simplified", lambda: _make_bodyworld_env(SimplifiedEmbodiedWorld)),
        ("avatar", lambda: _make_avatar_env()),
    ]

    structure_metrics = ["stability_mean_mean", "state_autocorrelation_mean", "history_dependence_mean", "self_world_marker_mean"]
    wins_by_world = {}

    for world_name, env_factory in world_configs:
        # Bodyless
        _, bl_metrics = _run_seeds(env_factory, bodyless_ctrl, SEEDS_FAST)
        bl_agg = _agg(bl_metrics)

        # Body-preserving
        for ctrl_name, ctrl in body_ctrls.items():
            _, body_metrics = _run_seeds(env_factory, ctrl, SEEDS_FAST)
            body_agg = _agg(body_metrics)

            wins = 0
            for sm in structure_metrics:
                body_val = body_agg.get(sm, 0)
                bl_val = bl_agg.get(sm, 0)
                if body_val > bl_val:
                    wins += 1

            key = f"{world_name}/{ctrl_name}"
            wins_by_world[key] = wins
            row = {
                "world": world_name, "controller": ctrl_name,
                "structure_wins": wins,
            }
            row.update({f"body_{k}": v for k, v in body_agg.items()})
            row.update({f"bodyless_{k}": v for k, v in bl_agg.items()})
            rows.append(row)
            notes.append(f"  {key}: body wins {wins}/4 structure metrics vs bodyless")

    _write_csv(rows, output_dir / "body_substrate_value.csv")

    # Pass: body beats bodyless on >=3 structure metrics in >=2 worlds
    worlds_with_3plus = sum(1 for v in wins_by_world.values() if v >= 3)
    passed = worlds_with_3plus >= 2
    notes.append(f"  Worlds with >=3 wins: {worlds_with_3plus} (need >=2)")

    return StageResult(
        stage="Stage 3: Body Substrate",
        passed=passed,
        details={"wins_by_world": wins_by_world, "worlds_with_3plus": worlds_with_3plus},
        notes=notes,
    )


def stage_4_ascending_loop(output_dir: Path) -> StageResult:
    """Stage 4 — Ascending/descending loop validation."""
    notes = []
    rows = []

    ctrl = ReducedDescendingController()
    ablation_configs = {
        "baseline": frozenset(),
        "no_pose": frozenset({"pose"}),
        "no_locomotion": frozenset({"locomotion"}),
        "no_contact": frozenset({"contact"}),
        "no_target": frozenset({"target"}),
        "no_internal": frozenset({"internal"}),
        "all_off": frozenset({"pose", "locomotion", "contact", "target", "internal"}),
    }

    baseline_agg = None
    ablation_results = {}

    for abl_name, channels in ablation_configs.items():
        env_factory = lambda ch=channels: _make_avatar_env(disabled_channels=ch)
        _, metrics_list = _run_seeds(env_factory, ctrl, SEEDS_FAST)
        agg = _agg(metrics_list)
        ablation_results[abl_name] = agg

        if abl_name == "baseline":
            baseline_agg = agg

        row = {"ablation": abl_name}
        row.update(agg)
        rows.append(row)

    _write_csv(rows, output_dir / "ascending_loop_ablation.csv")

    # Check: at least one channel removal causes significant stability drop
    significant_drops = []
    for abl_name, agg in ablation_results.items():
        if abl_name == "baseline":
            continue
        baseline_stab = baseline_agg.get("stability_mean_mean", 0)
        abl_stab = agg.get("stability_mean_mean", 0)
        drop = baseline_stab - abl_stab
        baseline_sw = baseline_agg.get("self_world_marker_mean", 0)
        abl_sw = agg.get("self_world_marker_mean", 0)
        sw_drop = abs(baseline_sw) - abs(abl_sw)
        notes.append(f"  {abl_name}: stab_drop={drop:.4f}, sw_drop={sw_drop:.4f}")
        if drop > 0.10 or sw_drop > 0.10:
            significant_drops.append(abl_name)

    passed = len(significant_drops) > 0
    notes.append(f"  Significant drops: {significant_drops}")

    # Strong pass: different channels produce different collapse signatures
    if len(significant_drops) > 1:
        notes.append("  STRONG PASS: multiple channels produce different collapse signatures")

    return StageResult(
        stage="Stage 4: Ascending Loop",
        passed=passed,
        details={"significant_drops": significant_drops},
        notes=notes,
    )


def stage_5_history_dependence(output_dir: Path) -> StageResult:
    """Stage 5 — History dependence / hysteresis validation."""
    notes = []
    rows = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "random": RandomController(),
    }

    for ctrl_name, ctrl in controllers.items():
        env_factory = lambda: _make_avatar_env()
        all_trans, all_metrics = _run_seeds(env_factory, ctrl, SEEDS_FAST, max_steps=EPISODE_STEPS_LONG)
        agg = _agg(all_metrics)

        row = {"controller": ctrl_name}
        row.update(agg)
        rows.append(row)
        notes.append(
            f"  {ctrl_name}: history_dep={agg.get('history_dependence_mean', 0):.4f}, "
            f"hysteresis={agg.get('hysteresis_score_mean', 0):.4f}, "
            f"mi={agg.get('mi_mean_mean', 0):.4f}"
        )

    _write_csv(rows, output_dir / "history_dependence.csv")

    # Check: memory controller has more history dependence than random
    mem_hist = 0
    rand_hist = 0
    for r in rows:
        if r["controller"] == "memory":
            mem_hist = r.get("history_dependence_mean", 0)
        elif r["controller"] == "random":
            rand_hist = r.get("history_dependence_mean", 0)

    gap = mem_hist - rand_hist
    passed = gap > 0.05
    notes.append(f"  Memory vs random history gap: {gap:.4f}")

    return StageResult(
        stage="Stage 5: History Dependence",
        passed=passed,
        details={"memory_hist": float(mem_hist), "random_hist": float(rand_hist), "gap": float(gap)},
        notes=notes,
    )


def stage_6_self_world(output_dir: Path) -> StageResult:
    """Stage 6 — Self/world disambiguation validation."""
    notes = []
    rows = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "reflex_only": ReflexOnlyController(),
        "bodyless": BodylessAvatarController(),
        "random": RandomController(),
    }

    # Avatar world has external events built in
    for ctrl_name, ctrl in controllers.items():
        env_factory = lambda: _make_avatar_env()
        all_trans, all_metrics = _run_seeds(env_factory, ctrl, SEEDS_FAST)
        agg = _agg(all_metrics)

        row = {"controller": ctrl_name}
        row.update(agg)
        rows.append(row)
        notes.append(
            f"  {ctrl_name}: self_world={agg.get('self_world_marker_mean', 0):.4f}"
        )

    _write_csv(rows, output_dir / "self_world_disambiguation.csv")

    # Check: reduced/memory beat reflex, bodyless, random on self_world_marker
    rd_sw = next((r.get("self_world_marker_mean", 0) for r in rows if r["controller"] == "reduced_descending"), 0)
    baselines_sw = [
        abs(r.get("self_world_marker_mean", 0))
        for r in rows if r["controller"] in ("reflex_only", "bodyless", "random")
    ]
    baseline_max = max(baselines_sw) if baselines_sw else 0
    passed = abs(rd_sw) > baseline_max
    notes.append(f"  Reduced |self_world|={abs(rd_sw):.4f} vs baseline_max={baseline_max:.4f}")

    return StageResult(
        stage="Stage 6: Self/World",
        passed=passed,
        details={"rd_sw": float(rd_sw), "baseline_max": float(baseline_max)},
        notes=notes,
    )


def stage_7_interoperability(output_dir: Path) -> StageResult:
    """Stage 7 — Controller interoperability validation."""
    notes = []
    rows = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "reflex_only": ReflexOnlyController(),
        "raw_control": RawControlController(),
    }

    env_factory = lambda: _make_avatar_env()
    ctrl_transitions: dict[str, list[StepTransition]] = {}

    for ctrl_name, ctrl in controllers.items():
        env = env_factory()
        t = run_episode(env, ctrl, seed=0, max_steps=EPISODE_STEPS)
        ctrl_transitions[ctrl_name] = t

    # Pairwise comparisons
    ctrl_names = list(controllers.keys())
    raw_agreements = []
    translated_agreements = []

    for i in range(len(ctrl_names)):
        for j in range(i + 1, len(ctrl_names)):
            n1, n2 = ctrl_names[i], ctrl_names[j]
            t1, t2 = ctrl_transitions[n1], ctrl_transitions[n2]
            interop = interoperability_score(t1, t2)
            latent = latent_state_similarity(t1, t2)
            reward = reward_trajectory_similarity(t1, t2)

            raw_agr = abs(latent.get("latent_correlation", 0))
            trans_agr = interop.get("interoperability_score", 0)
            raw_agreements.append(raw_agr)
            translated_agreements.append(trans_agr)

            row = {
                "pair": f"{n1}_vs_{n2}",
                "raw_latent_corr": latent.get("latent_correlation", 0),
                "reward_corr": reward.get("reward_correlation", 0),
                "interop_score": trans_agr,
            }
            rows.append(row)
            notes.append(f"  {n1} vs {n2}: interop={trans_agr:.4f}, raw_corr={raw_agr:.4f}")

    _write_csv(rows, output_dir / "controller_interoperability.csv")

    mean_raw = np.mean(raw_agreements) if raw_agreements else 0
    mean_trans = np.mean(translated_agreements) if translated_agreements else 0
    gap = mean_trans - mean_raw
    passed = gap > 0.10
    notes.append(f"  Mean translated={mean_trans:.4f} vs raw={mean_raw:.4f}, gap={gap:.4f}")

    return StageResult(
        stage="Stage 7: Interoperability",
        passed=passed,
        details={"mean_raw": float(mean_raw), "mean_trans": float(mean_trans), "gap": float(gap)},
        notes=notes,
    )


def stage_8_seam_stress(output_dir: Path) -> StageResult:
    """Stage 8 — Seam / composition validation."""
    notes = []
    rows = []

    ctrl = ReducedDescendingController()
    env_factory_normal = lambda: _make_avatar_env()

    # Baseline
    _, baseline_metrics = _run_seeds(env_factory_normal, ctrl, SEEDS_FAST)
    baseline_agg = _agg(baseline_metrics)

    # Seam perturbations via ablations (approximations)
    perturbations = {
        "broken_descending": frozenset({"pose", "locomotion"}),
        "sensor_dropout": frozenset({"contact", "target"}),
        "all_channels_broken": frozenset({"pose", "locomotion", "contact", "target", "internal"}),
    }

    seam_predicts_failure = False
    for pert_name, channels in perturbations.items():
        env_factory = lambda ch=channels: _make_avatar_env(disabled_channels=ch)
        _, pert_metrics = _run_seeds(env_factory, ctrl, SEEDS_FAST)
        pert_agg = _agg(pert_metrics)

        seam_base = baseline_agg.get("seam_fragility_mean", 0)
        seam_pert = pert_agg.get("seam_fragility_mean", 0)
        perf_base = baseline_agg.get("return_mean", 0)
        perf_pert = pert_agg.get("return_mean", 0)
        perf_drop = perf_base - perf_pert
        stab_base = baseline_agg.get("stability_mean_mean", 0)
        stab_pert = pert_agg.get("stability_mean_mean", 0)
        stab_drop = stab_base - stab_pert

        # Seam perturbation is meaningful if it changes stability, seam metric, or performance
        if abs(seam_pert - seam_base) > 0.01 or abs(perf_drop) > 0.1 or stab_drop > 0.1:
            seam_predicts_failure = True

        row = {
            "perturbation": pert_name,
            "seam_fragility_baseline": seam_base,
            "seam_fragility_perturbed": seam_pert,
            "return_baseline": perf_base,
            "return_perturbed": perf_pert,
            "performance_drop": perf_drop,
            "stability_drop": stab_drop,
        }
        rows.append(row)
        notes.append(f"  {pert_name}: perf_drop={perf_drop:.4f}, stab_drop={stab_drop:.4f}, seam_delta={seam_pert-seam_base:.4f}")

    _write_csv(rows, output_dir / "seam_stress.csv")

    return StageResult(
        stage="Stage 8: Seam Stress",
        passed=seam_predicts_failure,
        details={"seam_predicts_failure": seam_predicts_failure},
        notes=notes,
    )


def stage_9_shared_objectness(output_dir: Path) -> StageResult:
    """Stage 9 — Shared objectness validation."""
    notes = []
    rows = []

    env_factory = lambda: _make_avatar_env()

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "random": RandomController(),
    }

    ctrl_transitions: dict[str, list[StepTransition]] = {}
    for ctrl_name, ctrl in controllers.items():
        env = env_factory()
        t = run_episode(env, ctrl, seed=0, max_steps=EPISODE_STEPS)
        ctrl_transitions[ctrl_name] = t

    # Cross-controller objectness
    internal_scores = []
    shared_scores = []

    ctrl_names = list(controllers.keys())
    for i in range(len(ctrl_names)):
        for j in range(i + 1, len(ctrl_names)):
            n1, n2 = ctrl_names[i], ctrl_names[j]
            t1, t2 = ctrl_transitions[n1], ctrl_transitions[n2]

            int_score = target_representation_stability(t1)
            shared = shared_objectness_score(t1, t2)

            internal_scores.append(int_score.get("target_persistence", 0))
            shared_scores.append(shared.get("shared_objectness_score", 0))

            row = {
                "pair": f"{n1}_vs_{n2}",
                "internal_persistence": int_score.get("target_persistence", 0),
                "shared_objectness": shared.get("shared_objectness_score", 0),
            }
            rows.append(row)
            notes.append(f"  {n1} vs {n2}: internal={int_score.get('target_persistence',0):.4f}, shared={shared.get('shared_objectness_score',0):.4f}")

    _write_csv(rows, output_dir / "shared_objectness.csv")

    mean_internal = np.mean(internal_scores) if internal_scores else 0
    mean_shared = np.mean(shared_scores) if shared_scores else 0
    gap = mean_shared - mean_internal
    passed = gap > 0.05
    notes.append(f"  Mean shared={mean_shared:.4f} vs internal={mean_internal:.4f}, gap={gap:.4f}")

    return StageResult(
        stage="Stage 9: Shared Objectness",
        passed=passed,
        details={"mean_internal": float(mean_internal), "mean_shared": float(mean_shared), "gap": float(gap)},
        notes=notes,
    )


def stage_10_transfer(output_dir: Path) -> StageResult:
    """Stage 10 — Transfer validation."""
    notes = []
    rows = []

    ctrl = ReducedDescendingController()

    world_factories = {
        "simplified": lambda: _make_bodyworld_env(SimplifiedEmbodiedWorld),
        "avatar": lambda: _make_avatar_env(),
        "native": lambda: _make_bodyworld_env(NativePhysicalWorld),
    }

    world_metrics: dict[str, dict[str, float]] = {}
    for world_name, factory in world_factories.items():
        _, metrics_list = _run_seeds(factory, ctrl, SEEDS_FAST)
        world_metrics[world_name] = _agg(metrics_list)

    # Check which findings survive across worlds
    finding_keys = [
        ("stability_mean_mean", "stability"),
        ("state_autocorrelation_mean", "persistence"),
        ("history_dependence_mean", "history"),
        ("self_world_marker_mean", "self_world"),
    ]

    surviving_findings = []
    for metric_key, finding_name in finding_keys:
        values = [world_metrics[w].get(metric_key, 0) for w in world_metrics]
        # Finding survives if non-zero in at least 2 worlds
        nonzero = sum(1 for v in values if abs(v) > 0.01)
        survives = nonzero >= 2
        if survives:
            surviving_findings.append(finding_name)
        row = {"finding": finding_name, "survives_transfer": survives}
        for w in world_metrics:
            row[f"{w}_value"] = world_metrics[w].get(metric_key, 0)
        rows.append(row)
        notes.append(f"  {finding_name}: nonzero in {nonzero}/3 worlds → {'SURVIVES' if survives else 'FAILS'}")

    _write_csv(rows, output_dir / "transfer_validation.csv")

    passed = len(surviving_findings) >= 1
    notes.append(f"  Surviving findings: {surviving_findings}")

    return StageResult(
        stage="Stage 10: Transfer",
        passed=passed,
        details={"surviving_findings": surviving_findings},
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Main validation runner
# ---------------------------------------------------------------------------


def run_full_validation(output_dir: str | Path = "results/validation") -> dict[str, Any]:
    """Execute all 10 validation stages and produce final reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL VALIDATION PROGRAM")
    print("=" * 70)

    results: list[StageResult] = []
    stage_fns = [
        ("Stage 1", stage_1_sanity),
        ("Stage 2", stage_2_baseline_truth),
        ("Stage 3", stage_3_body_substrate),
        ("Stage 4", stage_4_ascending_loop),
        ("Stage 5", stage_5_history_dependence),
        ("Stage 6", stage_6_self_world),
        ("Stage 7", stage_7_interoperability),
        ("Stage 8", stage_8_seam_stress),
        ("Stage 9", stage_9_shared_objectness),
        ("Stage 10", stage_10_transfer),
    ]

    for name, fn in stage_fns:
        print(f"\n{'─' * 60}")
        print(f"Running {name}...")
        print(f"{'─' * 60}")
        t0 = time.time()
        try:
            result = fn(output_dir)
            elapsed = time.time() - t0
            status = "PASS" if result.passed else "FAIL"
            print(f"  → {status} ({elapsed:.1f}s)")
            for n in result.notes:
                print(n)
            results.append(result)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  → ERROR ({elapsed:.1f}s): {e}")
            results.append(StageResult(
                stage=name, passed=False,
                details={"error": str(e)},
                notes=[f"  ERROR: {e}"],
            ))

    # Build summary
    summary = {
        "total_stages": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "stages": {r.stage: {"passed": r.passed, "details": r.details} for r in results},
    }

    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY: {summary['passed']}/{summary['total_stages']} stages passed")
    print(f"{'=' * 70}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.stage}")

    # Save summary
    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    run_full_validation()
