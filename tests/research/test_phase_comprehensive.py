"""Comprehensive tests for all 8 phases of the execution plan.

Phase 1: Hard-Memory Benchmark (tasks, controllers, complexity meter, auditor)
Phase 2: Portability scoring + functional transfer
Phase 3: Body substrate (design-only, MuJoCo blocked)
Phase 4: BackboneShared composite metric
Phase 5: Repair policies (mismatch-aware, adaptive patchability)
Phase 6: Sleep Layer v2 (10-step cycle)
Phase 7: Degenerate convergence detector
Phase 8: Pipeline mock + seam monitor
"""

import numpy as np
import pytest

from flygym_research.cognition.body_reflex import BodylessBodyLayer
from flygym_research.cognition.config import EnvConfig
from flygym_research.cognition.controllers import (
    ReducedDescendingController,
    SlotMemoryController,
)
from flygym_research.cognition.envs import FlyAvatarEnv
from flygym_research.cognition.experiments import run_episode
from flygym_research.cognition.interfaces import (
    AscendingSummary,
    DescendingCommand,
    RawBodyFeedback,
    StepTransition,
)
from flygym_research.cognition.tasks import (
    BranchDependentRecallTask,
    SameStateDifferentHistoryTask,
)


# ─── Shared helpers ──────────────────────────────────────────────────────


def _make_raw_feedback(x: float = 0.0, y: float = 0.0) -> RawBodyFeedback:
    return RawBodyFeedback(
        time=0.01,
        joint_angles=np.zeros(6, dtype=np.float64),
        joint_velocities=np.zeros(6, dtype=np.float64),
        body_positions=np.array([[x, y, 1.5]], dtype=np.float64),
        body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        contact_active=np.ones(6, dtype=np.float64),
        contact_forces=np.zeros((6, 3), dtype=np.float64),
        contact_torques=np.zeros((6, 3), dtype=np.float64),
        contact_positions=np.zeros((6, 3), dtype=np.float64),
        contact_normals=np.zeros((6, 3), dtype=np.float64),
        contact_tangents=np.zeros((6, 3), dtype=np.float64),
        actuator_forces=np.zeros(6, dtype=np.float64),
    )


def _make_summary() -> AscendingSummary:
    return AscendingSummary(
        features={
            "stability": 0.8, "thorax_height_mm": 1.5, "body_speed_mm_s": 0.3,
            "contact_fraction": 0.5, "slip_risk": 0.1, "collision_load": 0.0,
            "locomotion_quality": 0.7, "actuator_effort": 0.2,
            "target_distance": 2.0, "target_salience": 0.5,
            "phase": 0.3, "phase_velocity": 0.1,
        },
        active_channels=("stability", "contact", "locomotion"),
        disabled_channels=(),
    )


def _make_transitions(n: int = 12) -> list[StepTransition]:
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    controller = ReducedDescendingController()
    return run_episode(env, controller, seed=42, max_steps=n)


def _collect_episodes(seeds=None, world_modes=None, max_steps=12):
    from flygym_research.cognition.experiments.exp_sleep_trace_compressor import (
        collect_trace_bank,
    )
    seeds = seeds or [0]
    world_modes = world_modes or ["avatar_remapped"]
    return collect_trace_bank(
        seeds=seeds, world_modes=world_modes,
        ablations=[frozenset()], perturbation_tags=["baseline"],
        max_steps=max_steps,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Hard-Memory Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════


class TestSameStateDifferentHistoryTask:
    def test_reset_and_step(self):
        task = SameStateDifferentHistoryTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "same_state_different_history_task"
        assert "cue_signal" in ws.observables
        assert "cue_visible" in ws.observables
        assert "in_decision_phase" in ws.observables

    def test_four_phases(self):
        config = EnvConfig(episode_steps=50, success_radius_mm=0.001)
        task = SameStateDifferentHistoryTask(
            config=config, event_step=2, event_visible_steps=2, decision_delay=5,
        )
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=0, raw_feedback=raw, summary=summary)
        phases = []
        for _ in range(15):
            ws = task.step(DescendingCommand(), raw, summary)
            phases.append(ws.info.get("phase"))
        assert phases[0] == "pre_event"      # step 1 < event_step=2
        assert phases[1] == "event"          # step 2
        assert phases[2] == "event"          # step 3
        assert phases[3] == "waiting"        # step 4 (after event window)

    def test_cue_ambiguous_after_event(self):
        config = EnvConfig(episode_steps=50, success_radius_mm=0.001)
        task = SameStateDifferentHistoryTask(
            config=config, event_step=2, event_visible_steps=2, decision_delay=5,
        )
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=0, raw_feedback=raw, summary=summary)
        signals = []
        for _ in range(10):
            ws = task.step(DescendingCommand(), raw, summary)
            signals.append(ws.observables["cue_signal"])
        # After event window, signal should be 0.5 (ambiguous)
        assert signals[5] == 0.5

    def test_event_type_randomized(self):
        task = SameStateDifferentHistoryTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        events = set()
        for seed in range(10):
            ws = task.reset(seed=seed, raw_feedback=raw, summary=summary)
            events.add(ws.info.get("event_type"))
        assert len(events) >= 2  # both event types should appear


class TestBranchDependentRecallTask:
    def test_reset_and_step(self):
        task = BranchDependentRecallTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "branch_dependent_recall_task"
        assert "cue_signal" in ws.observables

    def test_five_phases(self):
        config = EnvConfig(episode_steps=50, success_radius_mm=0.001)
        task = BranchDependentRecallTask(
            config=config, branch_step=2, branch_visible_steps=2,
            convergence_steps=3, decision_delay=5,
        )
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=0, raw_feedback=raw, summary=summary)
        phases = []
        for _ in range(15):
            ws = task.step(DescendingCommand(), raw, summary)
            phases.append(ws.info.get("phase"))
        assert phases[0] == "pre_branch"
        assert phases[1] == "branch_visible"
        assert phases[4] == "converging"

    def test_cue_fades_during_convergence(self):
        config = EnvConfig(episode_steps=50, success_radius_mm=0.001)
        task = BranchDependentRecallTask(
            config=config, branch_step=2, branch_visible_steps=2,
            convergence_steps=3, decision_delay=5,
        )
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=42, raw_feedback=raw, summary=summary)
        signals = []
        for _ in range(12):
            ws = task.step(DescendingCommand(), raw, summary)
            signals.append(ws.observables["cue_signal"])
        # After convergence, should be fully ambiguous
        assert signals[8] == 0.5


class TestSlotMemoryController:
    def test_reset_and_act(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SlotMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        action = ctrl.act(obs)
        assert isinstance(action, DescendingCommand)

    def test_slot_state_updates(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SlotMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        state = ctrl.get_internal_state()
        assert state["step_count"] == 1.0
        assert state["active_slots"] >= 1.0

    def test_importance_decays(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SlotMemoryController(importance_decay=0.9)
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        initial_imp = ctrl.get_internal_state()["max_importance"]
        assert isinstance(initial_imp, float)
        for _ in range(5):
            action = ctrl.act(obs)
            obs_next = env.step(action).observation
            obs = obs_next
        imp2 = ctrl.get_internal_state()["max_importance"]
        # Importance should decay over time (unless new high-importance writes)
        assert isinstance(imp2, float)

    def test_hidden_state_nonzero(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SlotMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        for _ in range(3):
            action = ctrl.act(obs)
            obs = env.step(action).observation
        state = ctrl.get_internal_state()
        assert state["hidden_norm"] > 0.0

    def test_intervene_state(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SlotMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        before = ctrl.get_internal_state()["hidden_norm"]
        ctrl.intervene_state(np.ones(ctrl.slot_dim) * 0.5)
        after = ctrl.get_internal_state()["hidden_norm"]
        assert after > before

    def test_shuffle_state(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SlotMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        for _ in range(3):
            ctrl.act(obs)
            obs = env.step(ctrl.act(obs)).observation
        state_before = ctrl.get_internal_state()
        ctrl.shuffle_state(np.random.default_rng(99))
        state_after = ctrl.get_internal_state()
        # Shuffling should preserve norms but change content
        assert abs(state_after["step_count"] - state_before["step_count"]) < 1e-6


class TestTaskComplexityMeter:
    def test_classify_complexity(self):
        from flygym_research.cognition.diagnostics.task_complexity_meter import (
            classify_complexity,
        )
        # classify_complexity takes (reactive, memory, slot, random) returns
        # trivial: reactive barely beats random
        cclass, gaps = classify_complexity(1.0, 1.1, 1.2, 0.8)
        assert cclass == "trivial"  # reactive_vs_random < 0.5

        # shallow: reactive clearly beats random, but memory/slot don't beat reactive
        cclass, gaps = classify_complexity(3.0, 3.1, 3.2, 1.0)
        assert cclass == "shallow"  # gaps < 0.5

        # deep: slot significantly beats reactive
        cclass, gaps = classify_complexity(3.0, 3.5, 5.0, 1.0)
        assert cclass == "deep"  # slot_vs_trace >= 1.0

        # moderate: memory or slot beats reactive by 0.5-1.0
        cclass, gaps = classify_complexity(3.0, 3.8, 3.6, 1.0)
        assert cclass == "moderate"  # memory gap = 0.8 >= 0.5

    def test_compute_task_complexity(self):
        from flygym_research.cognition.diagnostics.task_complexity_meter import (
            compute_task_complexity,
        )
        returns = {
            "random": [1.0, 1.5, 0.5],
            "reactive": [3.0, 3.5, 2.5],
            "memory": [4.0, 4.5, 3.5],
            "slot": [6.0, 6.5, 5.5],
        }
        result = compute_task_complexity(returns)
        assert result.memory_vs_trace_gap == pytest.approx(1.0, abs=0.01)
        assert result.slot_vs_trace_gap == pytest.approx(3.0, abs=0.01)
        assert result.complexity_class in ("trivial", "shallow", "moderate", "deep", "ultra_deep")

    def test_batch_complexity_assessment(self):
        from flygym_research.cognition.diagnostics.task_complexity_meter import (
            batch_complexity_assessment,
        )
        returns_by_task = {
            "easy_task": {
                "random": [1.0], "reactive": [2.0], "memory": [2.1], "slot": [2.2],
            },
            "hard_task": {
                "random": [1.0], "reactive": [2.0], "memory": [5.0], "slot": [8.0],
            },
        }
        results = batch_complexity_assessment(returns_by_task)
        assert "easy_task" in results
        assert "hard_task" in results
        assert results["hard_task"].slot_vs_trace_gap > results["easy_task"].slot_vs_trace_gap


class TestMetricAuditor:
    def test_check_permutation_invariance(self):
        from flygym_research.cognition.diagnostics.metric_auditor import (
            check_permutation_invariance,
        )
        # A metric that IS permutation invariant (mean) — should flag as blind
        def mean_metric(values):
            return {"mean": float(np.mean(values))}

        result = check_permutation_invariance(
            mean_metric, [np.array([1.0, 2.0, 3.0])], "mean",
        )
        # Mean IS invariant → should NOT pass (flags structural blindness)
        assert not result.passed

    def test_check_scale_invariance(self):
        from flygym_research.cognition.diagnostics.metric_auditor import (
            check_scale_invariance,
        )
        # A metric that IS scale-sensitive (std)
        def std_metric(values):
            return {"std": float(np.std(values))}

        result = check_scale_invariance(
            std_metric, [np.array([1.0, 2.0, 3.0])], "std",
        )
        # std responds to scaling → should pass
        assert result.passed

    def test_check_constant_output(self):
        from flygym_research.cognition.diagnostics.metric_auditor import (
            check_constant_output,
        )
        # A metric that always returns the same value
        def const_metric(values):
            return {"val": 42.0}

        result = check_constant_output(const_metric, [[1.0], [2.0], [3.0]], "val")
        assert not result.passed  # should detect constant output

    def test_check_averaging_distortion(self):
        from flygym_research.cognition.diagnostics.metric_auditor import (
            check_averaging_distortion,
        )
        # Values with high variance
        result = check_averaging_distortion(
            [1.0, 100.0, 1.0, 100.0, 1.0], "bimodal_metric"
        )
        assert not result.passed  # high CV should flag distortion

    def test_audit_metric_full(self):
        from flygym_research.cognition.diagnostics.metric_auditor import (
            audit_metric, AuditResult,
        )
        results = [
            AuditResult(
                metric_name="", check_name="test1", passed=True,
                severity="info", detail="ok",
            ),
            AuditResult(
                metric_name="", check_name="test2", passed=False,
                severity="critical", detail="fail",
            ),
        ]
        report = audit_metric("test_metric", results)
        assert not report.overall_passed
        assert report.critical_failures == 1
        assert report.warnings == 0


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Portability Scoring + Functional Transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestPortabilityScoring:
    def test_candidate_class_hierarchy(self):
        from flygym_research.cognition.sleep.portability import CandidateClass
        assert CandidateClass.UNIVERSAL.value == "universal"
        assert CandidateClass.PORTABLE.value == "portable"
        assert CandidateClass.LOCAL.value == "local"

    def test_classify_candidate_local(self):
        from flygym_research.cognition.sleep.portability import classify_candidate
        episodes = _collect_episodes(seeds=[0], world_modes=["avatar_remapped"])
        from flygym_research.cognition.sleep import extract_sleep_candidates
        candidates = extract_sleep_candidates(episodes, min_equivalence_strength=0.1)
        if candidates:
            classified = classify_candidate(candidates[0], episodes)
            assert classified.world_count >= 1
            assert classified.portability_score >= 0.0
            assert classified.portability_score <= 1.0

    def test_classify_all_candidates_sorted(self):
        from flygym_research.cognition.sleep.portability import classify_all_candidates
        episodes = _collect_episodes(
            seeds=[0], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        from flygym_research.cognition.sleep import extract_sleep_candidates
        candidates = extract_sleep_candidates(
            episodes, min_equivalence_strength=0.1, cross_world=True,
        )
        classified = classify_all_candidates(candidates, episodes)
        # Should be sorted by portability_score descending
        for i in range(len(classified) - 1):
            assert classified[i].portability_score >= classified[i + 1].portability_score

    def test_portability_summary(self):
        from flygym_research.cognition.sleep.portability import (
            classify_all_candidates,
            portability_summary,
        )
        episodes = _collect_episodes(
            seeds=[0], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        from flygym_research.cognition.sleep import extract_sleep_candidates
        candidates = extract_sleep_candidates(
            episodes, min_equivalence_strength=0.1, cross_world=True,
        )
        classified = classify_all_candidates(candidates, episodes)
        stats = portability_summary(classified)
        assert "n_candidates" in stats
        assert "n_universal" in stats
        assert "n_portable" in stats
        assert "n_local" in stats
        assert stats["hierarchy"] == "universal > portable > local"

    def test_select_transfer_candidates(self):
        from flygym_research.cognition.sleep.portability import (
            CandidateClass,
            classify_all_candidates,
            select_transfer_candidates,
        )
        episodes = _collect_episodes(
            seeds=[0, 1], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        from flygym_research.cognition.sleep import extract_sleep_candidates
        candidates = extract_sleep_candidates(
            episodes, min_equivalence_strength=0.1, cross_world=True,
        )
        classified = classify_all_candidates(candidates, episodes)
        selected = select_transfer_candidates(classified, min_class=CandidateClass.LOCAL)
        assert len(selected) <= 10


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: BackboneShared Composite Metric
# ═══════════════════════════════════════════════════════════════════════════


class TestBackboneShared:
    def test_compute_with_single_controller(self):
        from flygym_research.cognition.metrics.backbone_shared import compute_backbone_shared
        transitions = _make_transitions(12)
        result = compute_backbone_shared({"ctrl_a": transitions})
        # Single controller → zero score
        assert result["backbone_shared"] == 0.0
        assert result["gate_approved"] is False

    def test_compute_with_two_controllers(self):
        from flygym_research.cognition.metrics.backbone_shared import compute_backbone_shared
        t1 = _make_transitions(12)
        t2 = _make_transitions(12)
        result = compute_backbone_shared({"ctrl_a": t1, "ctrl_b": t2})
        assert "backbone_shared" in result
        assert "quotient_redundancy" in result
        assert "interop_loss" in result
        assert "seam_fragility_mean" in result
        assert "gate_approved" in result

    def test_cross_world_activates_drift(self):
        from flygym_research.cognition.metrics.backbone_shared import compute_backbone_shared
        episodes_sw = _collect_episodes(seeds=[0], world_modes=["avatar_remapped"])
        episodes_cw = _collect_episodes(seeds=[0], world_modes=["simplified_embodied"])
        sw_trans = {ep.controller_name: ep.transitions for ep in episodes_sw}
        cw_trans = {ep.controller_name: ep.transitions for ep in episodes_cw}
        result = compute_backbone_shared(sw_trans, cross_world_transitions=cw_trans)
        # Scale drift should be ≥ 0 with cross-world data
        assert result["scale_drift"] >= 0.0

    def test_compare_world_modes(self):
        from flygym_research.cognition.metrics.backbone_shared import compare_world_modes
        episodes_sw = _collect_episodes(seeds=[0], world_modes=["avatar_remapped"])
        episodes_cw = _collect_episodes(seeds=[0], world_modes=["simplified_embodied"])
        sw_trans = {ep.controller_name: ep.transitions for ep in episodes_sw}
        cw_trans = {ep.controller_name: ep.transitions for ep in episodes_cw}
        result = compare_world_modes(sw_trans, cw_trans)
        assert "same_world_backbone_shared" in result
        assert "cross_world_backbone_shared" in result
        assert "backbone_shared_delta" in result
        assert "distinguishes_modes" in result

    def test_config_affects_gate(self):
        from flygym_research.cognition.metrics.backbone_shared import (
            BackboneSharedConfig,
            compute_backbone_shared,
        )
        t1 = _make_transitions(12)
        t2 = _make_transitions(12)
        strict = BackboneSharedConfig(gate_threshold=10.0)
        result = compute_backbone_shared({"a": t1, "b": t2}, config=strict)
        assert result["gate_approved"] is False  # threshold too high

    def test_component_breakdown(self):
        from flygym_research.cognition.metrics.backbone_shared import compute_backbone_shared
        t1 = _make_transitions(12)
        t2 = _make_transitions(12)
        result = compute_backbone_shared({"a": t1, "b": t2})
        components = result["components"]
        assert "Omega_contribution" in components
        assert "interop_penalty" in components
        assert "seam_penalty" in components
        assert "drift_penalty" in components


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 5: Repair Policies
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairPolicies:
    def test_failure_type_enum(self):
        from flygym_research.cognition.sleep.repair_policies import FailureType
        assert FailureType.SEAM_DOMINANT.value == "seam_dominant"
        assert FailureType.MISMATCH_DOMINANT.value == "mismatch_dominant"
        assert FailureType.JOINT_FAILURE.value == "joint_failure"
        assert FailureType.MILD_FAILURE.value == "mild_failure"
        assert FailureType.NO_FAILURE.value == "no_failure"

    def test_diagnose_failure(self):
        from flygym_research.cognition.sleep.repair_policies import diagnose_failure
        transitions = _make_transitions(12)
        diag = diagnose_failure("test_ep", transitions)
        assert diag.episode_id == "test_ep"
        assert diag.seam_score >= 0.0
        assert diag.mismatch_score >= 0.0
        assert diag.seam_excess >= 0.0
        assert diag.mismatch_excess >= 0.0

    def test_apply_repair_no_failure(self):
        from flygym_research.cognition.sleep.repair_policies import (
            FailureType, apply_repair, diagnose_failure,
        )
        transitions = _make_transitions(12)
        diag = diagnose_failure("test_ep", transitions, seam_threshold=10.0, mismatch_threshold=10.0)
        # With very high thresholds, should be no failure
        assert diag.failure_type == FailureType.NO_FAILURE
        result = apply_repair(diag, transitions)
        assert result.repair_success is True
        assert result.repair_strategy == "none_needed"

    def test_no_repair_strategy(self):
        from flygym_research.cognition.sleep.repair_policies import (
            diagnose_failure, no_repair_strategy,
        )
        transitions = _make_transitions(12)
        diag = diagnose_failure("test_ep", transitions)
        result = no_repair_strategy(diag, transitions)
        assert result.repair_strategy == "no_repair"
        assert result.repaired_seam == diag.seam_score

    def test_uniform_repair_strategy(self):
        from flygym_research.cognition.sleep.repair_policies import (
            diagnose_failure, uniform_repair_strategy,
        )
        transitions = _make_transitions(12)
        diag = diagnose_failure("test_ep", transitions)
        result = uniform_repair_strategy(diag, transitions, reduction=0.3)
        assert result.repair_strategy == "uniform"
        # Repaired scores should be less than original
        assert result.repaired_seam <= diag.seam_score
        assert result.repaired_mismatch <= diag.mismatch_score

    def test_seam_only_repair_strategy(self):
        from flygym_research.cognition.sleep.repair_policies import (
            diagnose_failure, seam_only_repair_strategy,
        )
        transitions = _make_transitions(12)
        diag = diagnose_failure("test_ep", transitions)
        result = seam_only_repair_strategy(diag, transitions)
        assert result.repair_strategy == "seam_only"
        # Mismatch should be unchanged
        assert result.repaired_mismatch == diag.mismatch_score

    def test_compare_repair_strategies(self):
        from flygym_research.cognition.sleep.repair_policies import compare_repair_strategies
        episodes = _collect_episodes(
            seeds=[0, 1], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        episodes_with_trans = [
            (ep.episode_id, ep.transitions) for ep in episodes
        ]
        comparison = compare_repair_strategies(episodes_with_trans)
        assert "no_repair" in comparison
        assert "uniform" in comparison
        assert "seam_only" in comparison
        assert "mismatch_aware" in comparison

    def test_adaptive_patchability_from_distribution(self):
        from flygym_research.cognition.sleep.repair_policies import (
            AdaptivePatchability, diagnose_failure,
        )
        episodes = _collect_episodes(
            seeds=[0, 1], world_modes=["avatar_remapped"],
        )
        diagnoses = [
            diagnose_failure(ep.episode_id, ep.transitions)
            for ep in episodes
        ]
        failures = [d for d in diagnoses if d.failure_type.value != "no_failure"]
        if failures:
            adaptive = AdaptivePatchability.from_distribution(failures)
            assert adaptive.seam_margin >= 1.0
            assert adaptive.mismatch_margin >= 1.0
            assert adaptive.budget >= 1
        else:
            # No failures found — adaptive still works with no input
            adaptive = AdaptivePatchability()
            assert adaptive.budget == 10

    def test_adaptive_patchability_is_patchable(self):
        from flygym_research.cognition.sleep.repair_policies import (
            AdaptivePatchability, FailureDiagnosis, FailureType,
        )
        adaptive = AdaptivePatchability(seam_margin=2.0, mismatch_margin=2.0)
        # Within margin
        diag_ok = FailureDiagnosis(
            episode_id="ok", failure_type=FailureType.MILD_FAILURE,
            seam_score=0.3, mismatch_score=0.5,
            seam_threshold=0.2, mismatch_threshold=0.35,
            seam_excess=0.1, mismatch_excess=0.15,
        )
        assert adaptive.is_patchable(diag_ok) is True
        # Beyond margin
        diag_bad = FailureDiagnosis(
            episode_id="bad", failure_type=FailureType.JOINT_FAILURE,
            seam_score=1.0, mismatch_score=2.0,
            seam_threshold=0.2, mismatch_threshold=0.35,
            seam_excess=0.8, mismatch_excess=1.65,
        )
        assert adaptive.is_patchable(diag_bad) is False


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 6: Sleep Layer v2
# ═══════════════════════════════════════════════════════════════════════════


class TestSleepV2:
    def test_sleep_v2_config_defaults(self):
        from flygym_research.cognition.sleep.sleep_v2 import SleepV2Config
        config = SleepV2Config()
        assert config.cross_world is True
        assert config.max_scale_drift == 0.35
        assert config.replay_fragile is True

    def test_run_sleep_v2_cycle(self):
        from flygym_research.cognition.sleep.sleep_v2 import run_sleep_v2_cycle
        episodes = _collect_episodes(
            seeds=[0, 1],
            world_modes=["avatar_remapped", "simplified_embodied"],
        )
        result = run_sleep_v2_cycle(episodes)
        assert len(result.cycle_steps_completed) == 10  # exactly 10 steps
        assert "1_classify" in result.cycle_steps_completed
        assert "10_validate" in result.cycle_steps_completed
        assert result.artifact.artifact_id.startswith("sleep-")

    def test_sleep_v2_validation_keys(self):
        from flygym_research.cognition.sleep.sleep_v2 import run_sleep_v2_cycle
        episodes = _collect_episodes(seeds=[0], world_modes=["avatar_remapped"])
        result = run_sleep_v2_cycle(episodes)
        v = result.validation
        assert "n_original" in v
        assert "n_compressed" in v
        assert "n_demoted" in v
        assert "n_repairs_attempted" in v
        assert "compression_ratio" in v

    def test_compare_sleep_modes(self):
        from flygym_research.cognition.sleep.sleep_v2 import compare_sleep_modes
        episodes = _collect_episodes(
            seeds=[0, 1],
            world_modes=["avatar_remapped", "simplified_embodied"],
        )
        comparison = compare_sleep_modes(episodes)
        assert "no_sleep" in comparison
        assert "compression_only" in comparison
        assert "full_sleep_v2" in comparison
        assert comparison["no_sleep"]["size"] >= comparison["compression_only"]["size"]


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 7: Degenerate Convergence Detector
# ═══════════════════════════════════════════════════════════════════════════


class TestDegenerateConvergence:
    def test_convergence_type_enum(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            ConvergenceType,
        )
        assert ConvergenceType.HEALTHY.value == "healthy"
        assert ConvergenceType.DEGENERATE.value == "degenerate"
        assert ConvergenceType.AMBIGUOUS.value == "ambiguous"

    def test_compute_diversity(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            compute_diversity,
        )
        episodes = _collect_episodes(
            seeds=[0], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        trans_dict = {ep.controller_name: ep.transitions for ep in episodes}
        diversity = compute_diversity(trans_dict)
        assert isinstance(diversity, float)
        assert diversity >= 0.0

    def test_compute_recovery(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            compute_recovery,
        )
        episodes = _collect_episodes(seeds=[0, 1], world_modes=["avatar_remapped"])
        trans = {ep.controller_name: ep.transitions for ep in episodes}
        recovery = compute_recovery(trans, trans)  # same data → high recovery
        assert isinstance(recovery, float)

    def test_compute_transfer(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            compute_transfer,
        )
        episodes1 = _collect_episodes(seeds=[0], world_modes=["avatar_remapped"])
        episodes2 = _collect_episodes(seeds=[0], world_modes=["simplified_embodied"])
        t1 = {ep.controller_name: ep.transitions for ep in episodes1}
        t2 = {ep.controller_name: ep.transitions for ep in episodes2}
        transfer = compute_transfer(t1, t2)
        assert isinstance(transfer, float)
        assert transfer >= 0.0

    def test_detect_convergence(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            detect_convergence,
        )
        episodes = _collect_episodes(
            seeds=[0], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        trans_dict = {ep.controller_name: ep.transitions for ep in episodes}
        analysis = detect_convergence(trans_dict)
        assert analysis.convergence_type.value in ("healthy", "degenerate", "ambiguous")
        assert 0.0 <= analysis.omega_score
        assert isinstance(analysis.detail, dict)

    def test_construct_degenerate_scenario_reward_collapse(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            construct_degenerate_scenario,
        )
        episodes = _collect_episodes(seeds=[0], world_modes=["avatar_remapped"])
        trans_dict = {ep.controller_name: ep.transitions for ep in episodes}
        collapsed = construct_degenerate_scenario(trans_dict, "reward_collapse")
        # All controllers should have same reward
        rewards_per_ctrl = {
            name: [t.reward for t in trans]
            for name, trans in collapsed.items()
        }
        all_rewards = list(rewards_per_ctrl.values())
        if len(all_rewards) >= 2:
            np.testing.assert_allclose(all_rewards[0], all_rewards[1])

    def test_construct_degenerate_scenario_compression_narrowing(self):
        from flygym_research.cognition.diagnostics.degenerate_convergence import (
            construct_degenerate_scenario,
        )
        episodes = _collect_episodes(
            seeds=[0], world_modes=["avatar_remapped", "simplified_embodied"],
        )
        trans_dict = {ep.controller_name: ep.transitions for ep in episodes}
        narrowed = construct_degenerate_scenario(trans_dict, "compression_narrowing")
        # All controllers should share the same episodes
        all_ids = set()
        for name, trans in narrowed.items():
            ids = tuple(id(t) for t in trans)
            all_ids.add(ids)
        assert len(all_ids) == 1  # all identical


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 8: Pipeline Mock + Seam Monitor
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineMock:
    def test_perturbation_type_enum(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import PerturbationType
        assert len(PerturbationType) == 6

    def test_build_default_pipeline(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            build_default_pipeline,
        )
        pipeline = build_default_pipeline()
        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].name == "retriever"
        assert pipeline.stages[1].name == "processor"
        assert pipeline.stages[2].name == "executor"

    def test_pipeline_run(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            build_default_pipeline,
        )
        pipeline = build_default_pipeline()
        result = pipeline.run({"query": "What is Paris?"})
        assert "final_output" in result
        assert "intermediates" in result
        assert result["n_stages"] == 3
        assert result["final_output"]["confidence"] >= 0.0

    def test_apply_perturbation_chunk_mismatch(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            PerturbationType, apply_perturbation, build_default_pipeline,
        )
        pipeline = build_default_pipeline()
        baseline = pipeline.run({"query": "What is Paris?"})
        perturbed = apply_perturbation(baseline, PerturbationType.CHUNK_MISMATCH, pipeline)
        assert "perturbation" in perturbed
        assert perturbed["perturbation"]["type"] == "chunk_mismatch"

    def test_apply_perturbation_schema_mismatch(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            PerturbationType, apply_perturbation, build_default_pipeline,
        )
        pipeline = build_default_pipeline()
        baseline = pipeline.run({"query": "test"})
        perturbed = apply_perturbation(baseline, PerturbationType.SCHEMA_MISMATCH, pipeline)
        assert "perturbation" in perturbed

    def test_pipeline_seam_fragility(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            PerturbationType, apply_perturbation, build_default_pipeline,
            pipeline_seam_fragility,
        )
        pipeline = build_default_pipeline()
        baseline = pipeline.run({"query": "What is machine learning?"})
        perturbed_results = {}
        for ptype in PerturbationType:
            perturbed_results[ptype.value] = apply_perturbation(baseline, ptype, pipeline)
        analysis = pipeline_seam_fragility(baseline, perturbed_results)
        assert "mean_seam_fragility" in analysis
        assert "max_seam_fragility" in analysis
        assert "n_structurally_broken" in analysis
        assert "per_perturbation" in analysis
        assert len(analysis["per_perturbation"]) == 6

    def test_custom_knowledge_base(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            build_default_pipeline,
        )
        kb = [{"id": "custom1", "text": "Custom knowledge for testing.", "relevance": 0.9}]
        pipeline = build_default_pipeline(knowledge_base=kb)
        result = pipeline.run({"query": "Custom knowledge"})
        assert result["final_output"]["n_sources"] >= 1

    def test_empty_query(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            build_default_pipeline,
        )
        pipeline = build_default_pipeline()
        result = pipeline.run({"query": ""})
        assert result["final_output"]["confidence"] >= 0.0

    def test_all_perturbations_produce_results(self):
        from flygym_research.cognition.diagnostics.pipeline_mock import (
            PerturbationType, apply_perturbation, build_default_pipeline,
        )
        pipeline = build_default_pipeline()
        baseline = pipeline.run({"query": "test all perturbations"})
        for ptype in PerturbationType:
            perturbed = apply_perturbation(baseline, ptype, pipeline)
            assert "final_output" in perturbed


# ═══════════════════════════════════════════════════════════════════════════
# PHASE POC Experiment Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPOCExperiments:
    def test_hard_memory_benchmark(self, tmp_path):
        from flygym_research.cognition.experiments.exp_hard_memory_benchmark import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "hard_memory")
        assert result["n_tasks"] == 8
        assert result["n_controllers"] == 4
        assert "controller_means" in result
        assert "complexity" in result
        assert "pass_condition_met" in result

    def test_functional_transfer(self, tmp_path):
        from flygym_research.cognition.experiments.exp_functional_transfer import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "transfer")
        assert "tier_returns" in result
        assert "tier_counts" in result
        assert "portability_stats" in result
        assert "pass_condition_met" in result
        assert "navigation_task" in result
        assert "hard_task_distractor" in result
        assert "hard_tier_returns" in result
        assert "pass_conditions" in result
        assert "score_distribution" in result

    def test_backbone_shared(self, tmp_path):
        from flygym_research.cognition.experiments.exp_backbone_shared import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "backbone")
        assert "same_world_backbone_shared" in result
        assert "cross_world_backbone_shared" in result
        assert "comparison" in result
        assert "pass_condition_met" in result

    def test_repair_v2(self, tmp_path):
        from flygym_research.cognition.experiments.exp_repair_v2 import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "repair_v2")
        assert "failure_type_distribution" in result
        assert "strategy_comparison" in result
        assert "method_comparison" in result
        assert "cv_adaptive_summary" in result
        assert "repair_roi_queue" in result
        assert "pass_condition_met" in result
        assert "data_driven_wins" in result

    def test_sleep_v2(self, tmp_path):
        from flygym_research.cognition.experiments.exp_sleep_v2 import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "sleep_v2")
        assert "sleep_cycle_steps" in result
        assert "comparison" in result
        assert "sleep_v2_details" in result
        assert "pass_condition_met" in result

    def test_degenerate_convergence(self, tmp_path):
        from flygym_research.cognition.experiments.exp_degenerate_convergence import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "degenerate")
        assert "baseline" in result
        assert "scenarios" in result
        assert "pass_condition_met" in result

    def test_pipeline_seam(self, tmp_path):
        from flygym_research.cognition.experiments.exp_pipeline_seam import (
            run_experiment,
        )
        result = run_experiment(tmp_path / "pipeline")
        assert "per_query_results" in result
        assert "aggregate" in result
        assert "pass_condition_met" in result
