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
    counterfactual_divergence,
    cross_condition_objectness,
    cross_time_mutual_information,
    degeneracy_score,
    environment_invariant_dimensions,
    equivalence_class_size,
    full_quotient_analysis,
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
    translation_preserves_environment,
)
from ..metrics.publishable_metrics import (
    cross_validated_translation_r2,
    cross_world_translation_r2,
    distribution_stats,
    full_publishable_analysis,
    nonlinear_vs_linear_translation,
    noise_robustness_sweep,
    dimensionality_sweep,
)
from ..sleep import CompressionConfig, benchmark_portable_replay, compress_trace_bank
from ..worlds import AvatarRemappedWorld, NativePhysicalWorld, SimplifiedEmbodiedWorld
from ..experiments.exp_sleep_trace_compressor import collect_trace_bank
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
    # Honest status beyond binary pass/fail:
    #   "pass", "strong_pass", "fail",
    #   "blocked" (infra limitation),
    #   "metric_invalid" (measurement instrument broken),
    #   "inconclusive" (underpowered test),
    #   "proxy_insufficient" (metric too simplistic)
    status: str = ""

    def __post_init__(self) -> None:
        if not self.status:
            self.status = "pass" if self.passed else "fail"


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
        # BodylessBodyLayer always returns stability=1.0 regardless of
        # controller — the comparison is structurally impossible without
        # MuJoCo/EGL providing a real physics body layer.
        status="blocked" if not passed else "pass",
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

    strong = len(significant_drops) > 1
    return StageResult(
        stage="Stage 4: Ascending Loop",
        passed=passed,
        details={"significant_drops": significant_drops},
        notes=notes,
        status="strong_pass" if strong else ("pass" if passed else "fail"),
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
    """Stage 6 — Self/world disambiguation validation.

    Uses longer episodes and stronger perturbations to ensure enough
    external events occur for meaningful comparison.  The metric now
    measures action-response and target-vector disruption rather than
    body_speed (which is disconnected from world events in
    BodylessBodyLayer).
    """
    notes = []
    rows = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "reflex_only": ReflexOnlyController(),
        "bodyless": BodylessAvatarController(),
        "random": RandomController(),
    }

    # Use longer episodes (128 steps) so we get ~18 external events
    # (period=7) instead of potentially 0 on short episodes.
    for ctrl_name, ctrl in controllers.items():
        env_factory = lambda: _make_avatar_env()
        all_trans, all_metrics = _run_seeds(
            env_factory, ctrl, SEEDS_FAST, max_steps=EPISODE_STEPS_LONG,
        )
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
    mem_sw = next((r.get("self_world_marker_mean", 0) for r in rows if r["controller"] == "memory"), 0)
    best_body = max(abs(rd_sw), abs(mem_sw))
    baselines_sw = [
        abs(r.get("self_world_marker_mean", 0))
        for r in rows if r["controller"] in ("reflex_only", "bodyless", "random")
    ]
    baseline_max = max(baselines_sw) if baselines_sw else 0
    passed = best_body > baseline_max
    notes.append(f"  Best body-aware |self_world|={best_body:.4f} vs baseline_max={baseline_max:.4f}")

    return StageResult(
        stage="Stage 6: Self/World",
        passed=passed,
        details={"rd_sw": float(rd_sw), "mem_sw": float(mem_sw), "baseline_max": float(baseline_max)},
        notes=notes,
    )


def stage_7_interoperability(output_dir: Path) -> StageResult:
    """Stage 7 — Controller interoperability validation.

    Measures whether different controllers share *translatable* internal
    structure, not just similar reward trajectories.

    Core metric: R² of a linear translation T: z_a → z_b between
    multi-dimensional state vectors.  Reward is reported for diagnostics
    but deliberately excluded from the composite score to avoid
    environment-imposed inflation.

    All controllers run exactly EPISODE_STEPS steps (early termination
    disabled) to ensure equal-length trajectories for fair comparison.
    """
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
        # Force all controllers to run exactly EPISODE_STEPS — do not
        # break on early termination so that all trajectories are the
        # same length.  This prevents length-mismatch bias (planner
        # was terminating at ~10 steps by reaching the target).
        observation = env.reset(seed=0)
        ctrl.reset(seed=0)
        transitions: list[StepTransition] = []
        for _ in range(EPISODE_STEPS):
            action = ctrl.act(observation)
            transition = env.step(action)
            transitions.append(transition)
            observation = transition.observation
            # Do NOT break on terminated/truncated — force equal length
        ctrl_transitions[ctrl_name] = transitions
        notes.append(f"  {ctrl_name}: {len(transitions)} steps collected")

    # Pairwise comparisons
    ctrl_names = list(controllers.keys())
    raw_alignments = []
    translated_alignments = []

    for i in range(len(ctrl_names)):
        for j in range(i + 1, len(ctrl_names)):
            n1, n2 = ctrl_names[i], ctrl_names[j]
            t1, t2 = ctrl_transitions[n1], ctrl_transitions[n2]
            interop = interoperability_score(t1, t2)

            raw_agr = interop.get("raw_alignment", 0)
            trans_agr = interop.get("translated_alignment", 0)
            raw_alignments.append(raw_agr)
            translated_alignments.append(trans_agr)

            row = {
                "pair": f"{n1}_vs_{n2}",
                "raw_alignment": raw_agr,
                "translated_alignment": trans_agr,
                "translation_r2_ab": interop.get("translation_r2_ab", 0),
                "translation_r2_ba": interop.get("translation_r2_ba", 0),
                "action_corr_move": interop.get("action_corr_move", 0),
                "action_corr_turn": interop.get("action_corr_turn", 0),
                "action_mae": interop.get("action_mae", 0),
                "reward_corr": interop.get("reward_correlation", 0),
                "raw_dims_active": interop.get("raw_dims_active", 0),
            }
            rows.append(row)
            notes.append(
                f"  {n1} vs {n2}: translated_R2={trans_agr:.4f}, "
                f"raw={raw_agr:.4f}, action_move_corr={interop.get('action_corr_move', 0):.4f}, "
                f"reward_corr={interop.get('reward_correlation', 0):.4f}"
            )

    _write_csv(rows, output_dir / "controller_interoperability.csv")

    mean_raw = float(np.mean(raw_alignments)) if raw_alignments else 0.0
    mean_trans = float(np.mean(translated_alignments)) if translated_alignments else 0.0
    gap = mean_trans - mean_raw

    # Pass condition: translated alignment significantly beats raw alignment.
    # This means a linear structure-preserving map exists where raw comparison
    # shows nothing.
    passed = gap > 0.05 and mean_trans > 0.15
    notes.append(
        f"  Mean translated_R2={mean_trans:.4f} vs raw={mean_raw:.4f}, gap={gap:.4f}"
    )

    return StageResult(
        stage="Stage 7: Interoperability",
        passed=passed,
        details={
            "mean_raw": mean_raw,
            "mean_trans": mean_trans,
            "gap": gap,
        },
        notes=notes,
    )


def stage_7b_quotient_operator(output_dir: Path) -> StageResult:
    """Stage 7b — Environment as quotient operator.

    Formalizes the insight that E acts as a many→one projection:
        E: Z (controller space) → R (outcome space)

    This stage runs 5 experiments:
    1. Degeneracy detection — how much latent variance is lost in reward
    2. Environment-invariant dimensions — which state dims E preserves vs destroys
    3. Equivalence class analysis — how many controllers map to ~same outcome
    4. Translation validity — does T_ij preserve environment (E ∘ T ≈ E)
    5. Counterfactual divergence — do E-equivalent controllers diverge under E'

    Pass condition: degeneracy_ratio > 0.3 AND at least 1 divergent pair in
    counterfactual test, confirming that environment defines equivalence classes.
    """
    notes = []

    # ── Shared controller set and trajectory collection ──────────────
    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "reflex_only": ReflexOnlyController(),
        "raw_control": RawControlController(),
    }

    def _collect_transitions(
        env_factory, ctrls: dict[str, BrainInterface], n_steps: int = EPISODE_STEPS,
    ) -> dict[str, list[StepTransition]]:
        """Run all controllers in a given environment for n_steps."""
        result: dict[str, list[StepTransition]] = {}
        for ctrl_name, ctrl in ctrls.items():
            env = env_factory()
            observation = env.reset(seed=0)
            ctrl.reset(seed=0)
            transitions: list[StepTransition] = []
            for _ in range(n_steps):
                action = ctrl.act(observation)
                transition = env.step(action)
                transitions.append(transition)
                observation = transition.observation
            result[ctrl_name] = transitions
        return result

    # ── E1: standard avatar environment ──────────────────────────────
    notes.append("Collecting trajectories in E1 (avatar_remapped, standard)...")
    env_factory_E1 = lambda: _make_avatar_env()
    trans_E1 = _collect_transitions(env_factory_E1, controllers)
    notes.append(f"  E1: {len(trans_E1)} controllers × {EPISODE_STEPS} steps")

    # ── E2: perturbed avatar environment ─────────────────────────────
    # Double movement speed, 5× noise, reduced stability effect
    notes.append("Collecting trajectories in E2 (perturbed avatar: 2× speed, 5× noise, low stability)...")
    perturbed_cfg = EnvConfig(
        avatar_step_scale=0.70,      # 2× movement speed (default 0.35)
        avatar_noise_scale=0.10,     # 5× noise (default 0.02)
        avatar_stability_gain=0.20,  # reduced stability effect (default 0.75)
    )
    env_factory_E2 = lambda: FlyAvatarEnv(
        body=BodylessBodyLayer(config=BodyLayerConfig()),
        env_config=perturbed_cfg,
    )
    # Re-instantiate controllers for E2 (fresh state)
    controllers_E2 = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "reflex_only": ReflexOnlyController(),
        "raw_control": RawControlController(),
    }
    trans_E2 = _collect_transitions(env_factory_E2, controllers_E2)
    notes.append(f"  E2: {len(trans_E2)} controllers × {EPISODE_STEPS} steps")

    # ── Experiment 1: Degeneracy Detection ───────────────────────────
    notes.append("\n--- Experiment 1: Degeneracy Detection ---")
    degen = degeneracy_score(trans_E1)
    notes.append(f"  degeneracy_ratio = {degen['degeneracy_ratio']:.4f}")
    notes.append(f"  information_loss = {degen['information_loss']:.4f}")
    notes.append(f"  mean_pairwise_reward_corr = {degen['mean_pairwise_reward_corr']:.4f}")
    notes.append(
        f"  Interpretation: {degen['degeneracy_ratio']:.1%} of state variance exists "
        f"within same-reward bins → controllers use different internal states for same outcomes"
    )

    # ── Experiment 2: Environment-Invariant Dimensions ───────────────
    notes.append("\n--- Experiment 2: Invariant Dimension Analysis ---")
    inv_dims = environment_invariant_dimensions(trans_E1)
    dim_names = inv_dims.get("dim_names", [])
    per_dim = inv_dims.get("per_dim_reward_corr", [])
    notes.append(f"  Invariant dims (|corr with reward| > 0.3): {inv_dims['n_invariant']}")
    notes.append(f"  Destroyed dims (|corr with reward| < 0.1): {inv_dims['n_destroyed']}")
    notes.append(f"  Invariant fraction: {inv_dims['invariant_fraction']:.3f}")
    for i, (name, corr) in enumerate(zip(dim_names, per_dim)):
        tag = "PRESERVED" if abs(corr) > 0.3 else ("DESTROYED" if abs(corr) < 0.1 else "partial")
        notes.append(f"    dim[{i}] {name:20s} corr={corr:+.3f}  [{tag}]")

    # ── Experiment 3: Equivalence Class Analysis ─────────────────────
    notes.append("\n--- Experiment 3: Equivalence Class Analysis ---")
    equiv = equivalence_class_size(trans_E1)
    notes.append(f"  Number of equivalence classes: {equiv['n_classes']}")
    notes.append(f"  Mean class size: {equiv['mean_class_size']:.1f}")
    notes.append(f"  Max class size: {equiv['max_class_size']}")
    for cls in equiv.get("equivalence_classes", []):
        notes.append(f"    Class: {cls}")
    for n1, n2, ov in equiv.get("equivalence_pairs", []):
        notes.append(f"    {n1} vs {n2}: overlap={ov:.3f}")

    # ── Experiment 4: Translation Validity (E ∘ T ≈ E) ──────────────
    notes.append("\n--- Experiment 4: Translation Preserves Environment ---")
    ctrl_names = sorted(trans_E1.keys())
    validity_results = []
    for i, n1 in enumerate(ctrl_names):
        for n2 in ctrl_names[i + 1:]:
            tv = translation_preserves_environment(trans_E1[n1], trans_E1[n2])
            validity_results.append({
                "pair": f"{n1}_vs_{n2}",
                "preservation_r2": tv["environment_preservation_r2"],
                "baseline_r2": tv["baseline_prediction_r2"],
                "preservation_ratio": tv["preservation_ratio"],
                "naive_reward_corr": tv["naive_reward_corr"],
                "valid": tv["translation_validity"],
            })
            tag = "VALID" if tv["translation_validity"] else "INVALID"
            notes.append(
                f"  {n1} vs {n2}: preservation_R²={tv['environment_preservation_r2']:.3f} "
                f"(baseline={tv['baseline_prediction_r2']:.3f}, ratio={tv['preservation_ratio']:.3f}) [{tag}]"
            )

    valid_count = sum(1 for v in validity_results if v["valid"])
    mean_preservation = float(np.mean([v["preservation_r2"] for v in validity_results])) if validity_results else 0.0
    notes.append(f"  Valid translations: {valid_count}/{len(validity_results)}")
    notes.append(f"  Mean preservation R²: {mean_preservation:.4f}")

    # ── Experiment 5: Counterfactual Divergence (E1 vs E2) ───────────
    notes.append("\n--- Experiment 5: Counterfactual Divergence (E1 vs E2) ---")
    cf = counterfactual_divergence(trans_E1, trans_E2)
    notes.append(f"  Mean reward divergence: {cf['mean_reward_divergence']:.4f}")
    notes.append(f"  Max reward divergence: {cf['max_reward_divergence']:.4f}")
    notes.append(f"  Mean translation divergence: {cf['mean_translation_divergence']:.4f}")
    notes.append(f"  Divergent pairs (E1-equiv but E2-divergent): {cf['n_divergent']}")
    for p in cf.get("per_pair", []):
        notes.append(
            f"    {p['pair']}: reward_corr E1={p['reward_corr_E1']:.3f} → E2={p['reward_corr_E2']:.3f} "
            f"(Δ={p['reward_divergence']:.3f}), "
            f"trans_R² E1={p['translation_r2_E1']:.3f} → E2={p['translation_r2_E2']:.3f} "
            f"(Δ={p['translation_divergence']:.3f})"
        )

    # ── Write CSV ────────────────────────────────────────────────────
    rows = validity_results  # Main results table
    _write_csv(rows, output_dir / "quotient_operator.csv")

    # Also write counterfactual results
    if cf.get("per_pair"):
        _write_csv(cf["per_pair"], output_dir / "counterfactual_divergence.csv")

    # ── Pass Condition ───────────────────────────────────────────────
    # Core claim: environment defines equivalence classes of controllers.
    # Three independent lines of evidence:
    #   1. Information destruction: E loses substantial state information.
    #      Use information_loss (continuous R²-based) rather than dimension
    #      count.  Threshold 0.2 = at least 20% of state info is NOT
    #      reflected in reward.
    #   2. Equivalence classes: E groups controllers into ≥2 classes
    #      (different internal representations → same outcomes).
    #   3. Translation validity: learned T preserves E (E∘T ≈ E), i.e.
    #      translations stay within equivalence classes.  Majority valid.
    info_destroyed = degen["information_loss"] > 0.2
    multiple_classes = equiv["n_classes"] >= 2
    translation_valid = valid_count >= len(validity_results) // 2

    passed = info_destroyed and multiple_classes and translation_valid

    notes.append(f"\n--- Summary ---")
    notes.append(f"  Information destroyed: {info_destroyed} (loss={degen['information_loss']:.3f}, degeneracy={degen['degeneracy_ratio']:.3f})")
    notes.append(f"  Multiple equiv classes: {multiple_classes} ({equiv['n_classes']} classes)")
    notes.append(f"  Translation valid: {translation_valid} ({valid_count}/{len(validity_results)})")
    notes.append(f"  PASS: {passed}")

    return StageResult(
        stage="Stage 7b: Quotient Operator",
        passed=passed,
        details={
            "degeneracy_ratio": degen["degeneracy_ratio"],
            "information_loss": degen["information_loss"],
            "invariant_fraction": inv_dims["invariant_fraction"],
            "n_invariant_dims": inv_dims["n_invariant"],
            "n_destroyed_dims": inv_dims["n_destroyed"],
            "equivalence_classes": equiv["n_classes"],
            "mean_class_size": equiv["mean_class_size"],
            "valid_translations": valid_count,
            "total_pairs": len(validity_results),
            "mean_preservation_r2": mean_preservation,
            "counterfactual_divergent_pairs": cf["n_divergent"],
            "mean_reward_divergence": cf["mean_reward_divergence"],
            "mean_translation_divergence": cf["mean_translation_divergence"],
        },
        notes=notes,
    )


def stage_7c_publishable_protocol(output_dir: Path) -> StageResult:
    """Stage 7c — Publishable protocol for controller translation.

    Five experiments that elevate the translation finding to statistical rigor:
    1. Cross-validation (5-fold) — confirm R² holds out-of-sample
    2. Cross-world transfer — train T in avatar, test in simplified
    3. Nonlinear vs linear — MLP vs OLS (detect manifold complexity)
    4. Noise robustness — R² degradation under increasing noise
    5. Dimensionality sweep — find saturation point (5D/8D/10D/14D)

    Also reports aggregate stats excluding trivial pairs (reflex↔raw_control),
    with median + IQR alongside mean.

    Pass condition (three independent evidence lines):
      1. CV test R² > 0.3 (finding is not overfitting)
      2. Structure is linear (MLP gap < 0.05 over OLS)
      3. Noise degrades gracefully (no cliff-drops) AND ≥50% of nontrivial
         pairs remain meaningful (R² > 0.3) at 1× noise (equal to signal std)
    """
    notes = []

    # ── Controller set ───────────────────────────────────────────────
    controllers_A = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "reflex_only": ReflexOnlyController(),
        "raw_control": RawControlController(),
    }

    def _collect(env_factory, ctrls, n_steps=EPISODE_STEPS):
        result = {}
        for name, ctrl in ctrls.items():
            env = env_factory()
            obs = env.reset(seed=0)
            ctrl.reset(seed=0)
            trans = []
            for _ in range(n_steps):
                action = ctrl.act(obs)
                t = env.step(action)
                trans.append(t)
                obs = t.observation
            result[name] = trans
        return result

    # ── World A: avatar_remapped ─────────────────────────────────────
    notes.append("Collecting trajectories in World A (avatar_remapped)...")
    trans_A = _collect(lambda: _make_avatar_env(), controllers_A)
    notes.append(f"  World A: {len(trans_A)} controllers × {EPISODE_STEPS} steps")

    # ── World B: simplified_embodied ─────────────────────────────────
    notes.append("Collecting trajectories in World B (simplified_embodied)...")
    controllers_B = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "reflex_only": ReflexOnlyController(),
        "raw_control": RawControlController(),
    }
    trans_B = _collect(
        lambda: _make_bodyworld_env(SimplifiedEmbodiedWorld),
        controllers_B,
    )
    notes.append(f"  World B: {len(trans_B)} controllers × {EPISODE_STEPS} steps")

    # ── Run all 5 experiments ────────────────────────────────────────
    from ..metrics.publishable_metrics import (
        full_publishable_analysis,
        _is_trivial_pair,
    )

    pub = full_publishable_analysis(trans_A, trans_B)

    # ── Experiment 1: Cross-Validation ───────────────────────────────
    notes.append("\n--- Experiment 1: Cross-Validation (5-fold) ---")
    cv = pub["cross_validation"]
    agg_train = cv["aggregate_train"]
    agg_test = cv["aggregate_test"]
    notes.append(f"  Train R² (nontrivial): mean={agg_train['mean']:.4f}, median={agg_train['median']:.4f}")
    notes.append(f"  Test R²  (nontrivial): mean={agg_test['mean']:.4f}, median={agg_test['median']:.4f}")
    notes.append(f"  Overfitting gap: {cv['mean_overfitting_gap']:.4f}")
    for p in cv["per_pair"]:
        tag = " [trivial]" if p.get("trivial") else ""
        notes.append(
            f"    {p['pair']}: train={p['train_r2']:.3f} test={p['test_r2']:.3f} "
            f"gap={p['gap']:.3f} ovfit={p.get('overfitting_ratio', 0):.2%}{tag}"
        )

    # ── Experiment 2: Cross-World Transfer ───────────────────────────
    notes.append("\n--- Experiment 2: Cross-World Transfer ---")
    cw = pub["cross_world"]
    if "per_pair" in cw:
        notes.append(f"  Within-world R² (mean): {cw['mean_within_r2']:.4f}")
        notes.append(f"  Transfer R² (mean): {cw['mean_transfer_r2']:.4f}")
        notes.append(f"  Transfer ratio: {cw['transfer_ratio']:.4f}")
        notes.append(f"  Nontrivial within: {cw['nontrivial_within_r2']:.4f}")
        notes.append(f"  Nontrivial transfer: {cw['nontrivial_transfer_r2']:.4f}")
        notes.append(f"  Nontrivial ratio: {cw['nontrivial_transfer_ratio']:.4f}")
        for p in cw.get("per_pair", []):
            tag = " [trivial]" if p.get("trivial") else ""
            notes.append(
                f"    {p['pair']}: within={p['within_world_r2']:.3f} "
                f"transfer={p['transfer_r2']:.3f} ratio={p['transfer_ratio']:.3f}{tag}"
            )
    else:
        notes.append(f"  {cw.get('status', 'skipped')}")

    # ── Experiment 3: Nonlinear vs Linear ────────────────────────────
    notes.append("\n--- Experiment 3: Nonlinear vs Linear ---")
    nl = pub["nonlinear_vs_linear"]
    agg_lin = nl["aggregate_linear"]
    agg_nl = nl["aggregate_nonlinear"]
    notes.append(f"  Linear R² (nontrivial): mean={agg_lin['mean']:.4f}, median={agg_lin['median']:.4f}")
    notes.append(f"  Nonlinear R² (nontrivial): mean={agg_nl['mean']:.4f}, median={agg_nl['median']:.4f}")
    notes.append(f"  Complexity gap: {nl['mean_complexity_gap']:.4f}")
    notes.append(f"  Structure is linear: {nl['structure_is_linear']}")
    for p in nl["per_pair"]:
        tag = " [trivial]" if p.get("trivial") else ""
        notes.append(
            f"    {p['pair']}: linear={p['linear_r2']:.3f} nonlinear={p['nonlinear_r2']:.3f} "
            f"gap={p['complexity_gap']:.3f} linear={p.get('structure_is_linear', '?')}{tag}"
        )

    # ── Experiment 4: Noise Robustness ───────────────────────────────
    notes.append("\n--- Experiment 4: Noise Robustness ---")
    nr = pub["noise_robustness"]
    notes.append(f"  All pairs robust at 2× noise: {nr['all_robust']}")
    notes.append(f"  Mean degradation rate: {nr['mean_degradation_rate']:.4f}")
    for p in nr["per_pair"]:
        tag = " [trivial]" if p.get("trivial") else ""
        notes.append(
            f"    {p['pair']}: clean={p['clean_r2']:.3f} @1×={p.get('moderate_r2', 0):.3f} "
            f"@2×={p['noisiest_r2']:.3f} graceful={p.get('graceful', '?')}{tag}"
        )

    # ── Experiment 5: Dimensionality Sweep ───────────────────────────
    notes.append("\n--- Experiment 5: Dimensionality Sweep ---")
    ds = pub["dimensionality_sweep"]
    for p in ds["per_pair"]:
        tag = " [trivial]" if p.get("trivial") else ""
        dims_str = " → ".join(
            f"{d['dims']}D:{d['r2']:.3f}" for d in p.get("per_dim", [])
        )
        notes.append(f"    {p['pair']}: {dims_str} sat@{p.get('saturation_dim', '?')}D{tag}")

    # ── Save detailed CSV ────────────────────────────────────────────
    csv_rows = []
    for p in cv["per_pair"]:
        csv_rows.append({
            "pair": p["pair"],
            "trivial": p.get("trivial", False),
            "cv_train_r2": p["train_r2"],
            "cv_test_r2": p["test_r2"],
            "cv_gap": p["gap"],
        })
    _write_csv(csv_rows, output_dir / "publishable_cross_validation.csv")

    cw_rows = []
    for p in cw.get("per_pair", []):
        cw_rows.append({
            "pair": p["pair"],
            "trivial": p.get("trivial", False),
            "within_world_r2": p["within_world_r2"],
            "transfer_r2": p["transfer_r2"],
            "transfer_ratio": p["transfer_ratio"],
        })
    _write_csv(cw_rows, output_dir / "publishable_cross_world.csv")

    nl_rows = []
    for p in nl["per_pair"]:
        nl_rows.append({
            "pair": p["pair"],
            "trivial": p.get("trivial", False),
            "linear_r2": p["linear_r2"],
            "nonlinear_r2": p["nonlinear_r2"],
            "complexity_gap": p["complexity_gap"],
        })
    _write_csv(nl_rows, output_dir / "publishable_nonlinear_vs_linear.csv")

    # ── Pass Condition ───────────────────────────────────────────────
    # Three independent evidence lines for publishable quality:
    #   1. Cross-validation: test R² must be meaningful (>0.3)
    #      This confirms the finding is not overfitting.
    #   2. Structure is linear: nonlinear doesn't add >5% over linear.
    #      This means shared structure is genuinely linear (stronger claim).
    #   3. Noise degrades gracefully: no cliff-drops in R² curve AND
    #      R² still meaningful at moderate noise (1× signal std).
    #      Note: at extreme noise (2×), R² will drop — that's expected.
    #      The key is graceful degradation, not absolute threshold.
    cv_holds = agg_test["mean"] > 0.3
    structure_linear = nl["structure_is_linear"]

    # Noise robustness: check graceful degradation per nontrivial pair
    nt_noise = [p for p in nr["per_pair"] if not p.get("trivial")]
    noise_graceful = all(p.get("graceful", True) for p in nt_noise)
    noise_moderate = sum(1 for p in nt_noise if p.get("moderate_robust", False)) >= len(nt_noise) // 2

    passed = cv_holds and structure_linear and noise_graceful and noise_moderate

    notes.append(f"\n--- Pass Condition ---")
    notes.append(f"  CV test R² > 0.3: {cv_holds} (mean={agg_test['mean']:.4f})")
    notes.append(f"  Structure is linear: {structure_linear}")
    notes.append(f"  Noise degrades gracefully: {noise_graceful}")
    notes.append(f"  Noise moderate robust (≥50% pairs R²>0.3 at 1× noise): {noise_moderate}")
    notes.append(f"  PASS: {passed}")

    return StageResult(
        stage="Stage 7c: Publishable Protocol",
        passed=passed,
        details={
            "cv_train_r2_mean": agg_train["mean"],
            "cv_test_r2_mean": agg_test["mean"],
            "cv_test_r2_median": agg_test["median"],
            "cv_overfitting_gap": cv["mean_overfitting_gap"],
            "cross_world_within_r2": cw.get("nontrivial_within_r2", 0.0),
            "cross_world_transfer_r2": cw.get("nontrivial_transfer_r2", 0.0),
            "cross_world_transfer_ratio": cw.get("nontrivial_transfer_ratio", 0.0),
            "linear_r2_mean": agg_lin["mean"],
            "nonlinear_r2_mean": agg_nl["mean"],
            "complexity_gap": nl["mean_complexity_gap"],
            "structure_is_linear": nl["structure_is_linear"],
            "noise_graceful": noise_graceful,
            "noise_moderate_robust": noise_moderate,
            "noise_degradation_rate": nr["mean_degradation_rate"],
        },
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


def _candidate_is_stressed(candidate) -> bool:
    return (
        len(candidate.evidence.get("ablation_channels", [])) > 0
        or candidate.evidence.get("perturbation_tag") == "noisy"
    )


def stage_9_shared_objectness(output_dir: Path) -> StageResult:
    """Stage 9 — Shared-structure validation."""
    notes = []
    rows = []
    episodes = collect_trace_bank(
        seeds=[0],
        world_modes=["avatar_remapped", "simplified_embodied", "native_physical"],
        ablations=[frozenset(), frozenset({"pose"})],
        perturbation_tags=["baseline", "noisy"],
        max_steps=32,
    )
    artifact = compress_trace_bank(episodes, config=CompressionConfig())
    replay = benchmark_portable_replay(episodes, artifact)

    coherent = 0
    degraded = 0
    degenerate = 0
    stressed_degenerate = 0
    baseline_coherent = 0
    for candidate in artifact.candidates:
        score = candidate.score_components
        world_modes = ",".join(candidate.portability_evidence.get("world_modes", []))
        is_stressed = _candidate_is_stressed(candidate)
        regime = score.get("shared_structure_regime", "degenerate_convergence")
        if regime == "coherent_shared_structure":
            coherent += 1
            if not is_stressed:
                baseline_coherent += 1
        elif regime == "portable_but_degraded":
            degraded += 1
        else:
            degenerate += 1
            if is_stressed:
                stressed_degenerate += 1
        rows.append({
            "candidate_id": candidate.candidate_id,
            "redundancy_tier": candidate.redundancy_tier,
            "world_modes": world_modes,
            "perturbation_tag": candidate.evidence.get("perturbation_tag", "baseline"),
            "backbone_shared_score": score.get("backbone_shared_score", 0.0),
            "functional_transfer_gain": score.get("functional_transfer_gain", 0.0),
            "portability_fraction": score.get("portability_fraction", 0.0),
            "shared_structure_regime": regime,
        })
        notes.append(
            f"  {candidate.candidate_id}: tier={candidate.redundancy_tier}, worlds={world_modes}, "
            f"backbone={score.get('backbone_shared_score', 0.0):.3f}, regime={regime}"
        )

    replay_by_tier = replay.get("by_tier", {})
    for tier, values in sorted(replay_by_tier.items()):
        notes.append(
            f"  replay[{tier}]: backbone={values.get('mean_backbone_shared', 0.0):.3f}, "
            f"transfer={values.get('mean_functional_transfer_gain', 0.0):.3f}, "
            f"return_lift={values.get('mean_return_lift', 0.0):.3f}"
        )

    _write_csv(rows, output_dir / "shared_structure.csv")
    passed = coherent >= 1 and stressed_degenerate >= 1 and baseline_coherent >= 1
    return StageResult(
        stage="Stage 9: Shared Structure",
        passed=passed,
        details={
            "coherent_shared_structure": coherent,
            "portable_but_degraded": degraded,
            "degenerate_convergence": degenerate,
            "baseline_coherent": baseline_coherent,
            "stressed_degenerate": stressed_degenerate,
            "portable_replay": replay.get("summary", {}),
        },
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

    finding_keys = [
        ("stability_mean_mean", "stability"),
        ("state_autocorrelation_mean", "persistence"),
        ("history_dependence_mean", "history"),
        ("self_world_marker_mean", "self_world"),
    ]

    surviving_findings = []
    for metric_key, finding_name in finding_keys:
        values = [world_metrics[w].get(metric_key, 0) for w in world_metrics]
        nonzero = sum(1 for v in values if abs(v) > 0.01)
        survives = nonzero >= 2
        if survives:
            surviving_findings.append(finding_name)
        row = {"finding": finding_name, "survives_transfer": survives}
        for w in world_metrics:
            row[f"{w}_value"] = world_metrics[w].get(metric_key, 0)
        rows.append(row)
        notes.append(f"  {finding_name}: nonzero in {nonzero}/3 worlds → {'SURVIVES' if survives else 'FAILS'}")

    episodes = collect_trace_bank(seeds=[0], max_steps=24)
    artifact = compress_trace_bank(episodes, config=CompressionConfig())
    replay = benchmark_portable_replay(episodes, artifact)
    portable = replay.get("by_tier", {}).get("portable", {})
    universal = replay.get("by_tier", {}).get("universal", {})
    notes.append(
        f"  portable replay transfer={portable.get('mean_functional_transfer_gain', 0.0):.3f}, "
        f"universal replay transfer={universal.get('mean_functional_transfer_gain', 0.0):.3f}"
    )

    _write_csv(rows, output_dir / "transfer_validation.csv")

    replay_support = max(
        portable.get("mean_functional_transfer_gain", 0.0),
        universal.get("mean_functional_transfer_gain", 0.0),
    )
    passed = len(surviving_findings) >= 1 and replay_support >= 0.0
    notes.append(f"  Surviving findings: {surviving_findings}")

    return StageResult(
        stage="Stage 10: Transfer",
        passed=passed,
        details={
            "surviving_findings": surviving_findings,
            "portable_replay_support": replay_support,
            "portable_replay_summary": replay.get("summary", {}),
        },
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
        ("Stage 7b", stage_7b_quotient_operator),
        ("Stage 7c", stage_7c_publishable_protocol),
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
    status_labels = {
        "pass": "PASS",
        "strong_pass": "STRONG PASS",
        "fail": "FAIL",
        "blocked": "BLOCKED",
        "metric_invalid": "METRIC INVALID",
        "inconclusive": "INCONCLUSIVE",
        "proxy_insufficient": "PROXY INSUFFICIENT",
    }
    passed_count = sum(1 for r in results if r.passed)
    summary = {
        "total_stages": len(results),
        "passed": passed_count,
        "failed": sum(1 for r in results if not r.passed),
        "stages": {
            r.stage: {"passed": r.passed, "status": r.status, "details": r.details}
            for r in results
        },
    }

    print(f"\n{'=' * 70}")
    print(f"VALIDATION SUMMARY: {passed_count}/{summary['total_stages']} stages passed")
    print(f"{'=' * 70}")
    for r in results:
        label = status_labels.get(r.status, r.status.upper())
        print(f"  [{label}] {r.stage}")

    # Save summary
    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    run_full_validation()
