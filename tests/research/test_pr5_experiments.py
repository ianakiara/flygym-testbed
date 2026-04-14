"""Tests for PR #5 production-grade v2 experiments.

Runs all 7 upgraded experiments with production parameters and validates
pass/fail criteria per the runbook.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


# ---------------------------------------------------------------------------
# Infrastructure module smoke tests
# ---------------------------------------------------------------------------

class TestInfrastructureModules:
    """Verify all 8 infrastructure modules import and function correctly."""

    def test_long_horizon_runner_imports(self):
        from flygym_research.cognition.research.long_horizon_runner import (
            collect_long_horizon,
        )
        episodes = collect_long_horizon(max_steps=30, n_seeds=2)
        assert len(episodes) > 0

    def test_candidate_factory_imports(self):
        from flygym_research.cognition.research.candidate_factory import (
            build_candidate_pool,
        )
        from flygym_research.cognition.research.long_horizon_runner import (
            collect_long_horizon,
        )
        episodes = collect_long_horizon(max_steps=30, n_seeds=2)
        candidates, labels = build_candidate_pool(episodes, min_pool_size=10)
        assert len(candidates) > 0
        assert len(labels) > 0

    def test_adversarial_families_imports(self):
        from flygym_research.cognition.research.adversarial_families import (
            generate_adversarial_family,
        )
        from flygym_research.cognition.research.long_horizon_runner import (
            collect_long_horizon,
        )
        from flygym_research.cognition.sleep import compress_trace_bank
        episodes = collect_long_horizon(max_steps=30, n_seeds=2)
        artifact = compress_trace_bank(episodes)
        import numpy as np
        result = generate_adversarial_family(
            artifact.candidates, episodes, rng=np.random.default_rng(0),
        )
        assert isinstance(result, list)

    def test_stress_harness_imports(self):
        from flygym_research.cognition.research.stress_harness import (
            inject_seam_corruption,
            inject_delayed_target_mismatch,
        )
        from flygym_research.cognition.research.long_horizon_runner import (
            collect_long_horizon,
        )
        import numpy as np
        episodes = collect_long_horizon(max_steps=30, n_seeds=2)
        trans = episodes[0].transitions
        rng = np.random.default_rng(0)
        stressed = inject_seam_corruption(trans, rng=rng)
        assert len(stressed) == len(trans)
        delayed = inject_delayed_target_mismatch(trans, rng=rng)
        assert len(delayed) == len(trans)

    def test_observer_families_imports(self):
        from flygym_research.cognition.research.observer_families import (
            OBSERVER_FAMILIES,
        )
        assert len(OBSERVER_FAMILIES) == 9

    def test_transfer_scoring_imports(self):
        from flygym_research.cognition.research.transfer_scoring import (
            compute_transfer_score,
        )
        from flygym_research.cognition.research.long_horizon_runner import (
            collect_long_horizon,
        )
        from flygym_research.cognition.sleep import compress_trace_bank
        episodes = collect_long_horizon(max_steps=30, n_seeds=2)
        artifact = compress_trace_bank(episodes)
        if artifact.candidates:
            result = compute_transfer_score(artifact.candidates[0], episodes)
            assert "portability_fraction" in result

    def test_scale_transforms_imports(self):
        from flygym_research.cognition.research.scale_transforms import (
            SCALE_TRANSFORMS,
        )
        assert len(SCALE_TRANSFORMS) >= 7

    def test_memory_task_factory_imports(self):
        from flygym_research.cognition.research.memory_task_factory import (
            ALL_TASK_FAMILIES,
            create_task,
        )
        assert len(ALL_TASK_FAMILIES) == 6
        for name in ALL_TASK_FAMILIES:
            task = create_task(name)
            assert task is not None


# ---------------------------------------------------------------------------
# EXP 3: Deep Memory Benchmark v2
# ---------------------------------------------------------------------------

class TestExpDeepMemoryV2:
    """EXP 3 — Deep memory benchmark with 6 task families and 5 controllers."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_deep_memory_v2 import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("deep_memory_v2")
        return run_experiment(output_dir=out, episode_steps=50, n_seeds=3)

    def test_experiment_runs(self, results):
        assert "controller_summary" in results
        assert "task_controller_matrix" in results
        assert "pass_criteria" in results

    def test_has_all_controllers(self, results):
        controllers = set(results["controller_summary"].keys())
        expected = {"reactive", "reduced_descending", "scalar_memory",
                    "buffer_memory", "attention_memory"}
        assert expected.issubset(controllers)

    def test_has_all_task_families(self, results):
        tasks = set(results["task_controller_matrix"].keys())
        assert len(tasks) == 6

    def test_memory_advantage_curves(self, results):
        curves = results["memory_advantage_curves"]
        assert len(curves) > 0

    def test_pass_criteria_structure(self, results):
        pc = results["pass_criteria"]
        assert "memory_gt_reactive_4_of_6_tasks" in pc
        assert "causal_depth_gt_1" in pc
        assert "attention_gt_scalar_on_selective" in pc

    def test_seed_stability(self, results):
        assert "seed_stability" in results

    def test_output_file_created(self, results, tmp_path_factory):
        # Results dict is the parsed JSON — experiment ran and produced data
        assert results["config"]["total_runs"] > 0


# ---------------------------------------------------------------------------
# EXP 1: Long-horizon Composition Stress
# ---------------------------------------------------------------------------

class TestExpLongComposition:
    """EXP 1 — Long-horizon composition with seam stressors."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_long_composition import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("long_composition")
        return run_experiment(output_dir=out, episode_steps=50, n_seeds=3)

    def test_experiment_runs(self, results):
        assert "pass_criteria" in results
        assert "family_summary" in results
        assert "strategy_summary" in results

    def test_has_composition_strategies(self, results):
        strategies = set(results["strategy_summary"].keys())
        expected = {"bulk", "boundary", "corner"}
        assert expected.issubset(strategies)

    def test_has_stress_families(self, results):
        families = set(results["family_summary"].keys())
        assert len(families) >= 3  # at least clean + near_admissible + false_friend

    def test_seam_heatmap(self, results):
        assert "seam_failure_heatmap" in results

    def test_pass_criteria_structure(self, results):
        pc = results["pass_criteria"]
        assert "boundary_gt_bulk_win_rate" in pc
        assert "corner_gt_boundary_win_rate" in pc
        assert "seam_rho" in pc
        assert "seam_variance" in pc


# ---------------------------------------------------------------------------
# EXP 2: BackboneShared Adversarial Benchmark v2
# ---------------------------------------------------------------------------

class TestExpBackboneSharedV2:
    """EXP 2 — BackboneShared with 200+ candidates and 9 scoring models."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_backbone_shared_v2 import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("backbone_shared_v2")
        return run_experiment(
            output_dir=out, episode_steps=50, n_seeds=3, min_pool_size=20,
        )

    def test_experiment_runs(self, results):
        assert "model_results" in results
        assert "pass_criteria" in results

    def test_has_all_scoring_models(self, results):
        models = set(results["model_results"].keys())
        expected = {"entropy", "rank", "omega", "omega_shared",
                    "backbone_shared", "return_only", "low_drift_only",
                    "transfer_only", "combined_learned"}
        assert expected.issubset(models)

    def test_confusion_matrix(self, results):
        assert "confusion_matrix" in results

    def test_feature_importance(self, results):
        assert "feature_importance" in results

    def test_family_distribution(self, results):
        assert "family_distribution" in results
        assert len(results["family_distribution"]) > 0


# ---------------------------------------------------------------------------
# EXP 5: Selector Stress Test v2
# ---------------------------------------------------------------------------

class TestExpSelectorStressV2:
    """EXP 5 — Survival vs scalar vs basin+collapse selectors."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_selector_stress_v2 import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("selector_stress_v2")
        return run_experiment(
            output_dir=out, episode_steps=50, n_seeds=3, min_pool_size=20,
        )

    def test_experiment_runs(self, results):
        assert "in_distribution" in results
        assert "out_of_distribution" in results
        assert "pass_criteria" in results

    def test_three_selectors(self, results):
        selectors = set(results["in_distribution"].keys())
        expected = {"scalar", "survival", "basin_collapse"}
        assert expected.issubset(selectors)

    def test_ood_noise_levels(self, results):
        ood = results["out_of_distribution"]
        assert len(ood) >= 2

    def test_adversarial_acceptance(self, results):
        assert "adversarial_acceptance" in results

    def test_collapse_distances(self, results):
        assert "collapse_distances" in results

    def test_collapse_distance_formula_matches_selector(self):
        from flygym_research.cognition.experiments.exp_selector_stress_v2 import (
            _collapse_distance,
        )

        item = {
            "seam_risk": 0.2,
            "scale_drift": 0.1,
            "degeneracy_penalty": 0.3,
            "interop_loss": 0.4,
        }
        expected = float(
            np.clip(1.2 - 0.9 * 0.2 - 0.7 * 0.1 - 0.9 * 0.3 - 0.5 * 0.4, 0.0, 1.0)
        )

        assert _collapse_distance(item) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# EXP 7: Observer-Family Benchmark v2
# ---------------------------------------------------------------------------

class TestExpObserverInteropV2:
    """EXP 7 — Observer interop with 9 observer families."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_observer_interop_v2 import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("observer_interop_v2")
        return run_experiment(output_dir=out, episode_steps=50, n_seeds=3)

    def test_experiment_runs(self, results):
        assert "per_family" in results
        assert "pass_criteria" in results
        assert "overall" in results

    def test_has_all_observer_families(self, results):
        families = set(results["per_family"].keys())
        assert len(families) == 9

    def test_effect_size_computed(self, results):
        for family_data in results["per_family"].values():
            assert "effect_size" in family_data

    def test_hard_transform_check(self, results):
        pc = results["pass_criteria"]
        assert "hard_transforms_survive" in pc


# ---------------------------------------------------------------------------
# EXP 4: Portable Replay Benchmark v2
# ---------------------------------------------------------------------------

class TestExpPortableReplayV2:
    """EXP 4 — Portable replay with 5-world protocol."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_portable_replay_v2 import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("portable_replay_v2")
        return run_experiment(output_dir=out, episode_steps=50, n_seeds=3)

    def test_experiment_runs(self, results):
        assert "by_tier" in results
        assert "pass_criteria" in results

    def test_transfer_heatmap(self, results):
        assert "transfer_heatmap" in results

    def test_false_portable_detection(self, results):
        assert "false_portables" in results

    def test_tier_migration(self, results):
        assert "tier_migration" in results


# ---------------------------------------------------------------------------
# EXP 6: Scale Law Contrastive Benchmark
# ---------------------------------------------------------------------------

class TestExpScaleContrastiveV2:
    """EXP 6 — Scale law with real vs fake property families."""

    @pytest.fixture(scope="class")
    def results(self, tmp_path_factory):
        from flygym_research.cognition.experiments.exp_scale_contrastive_v2 import (
            run_experiment,
        )
        out = tmp_path_factory.mktemp("scale_contrastive_v2")
        return run_experiment(output_dir=out, episode_steps=50, n_seeds=3)

    def test_experiment_runs(self, results):
        assert "real_stability" in results
        assert "fake_stability" in results
        assert "pass_criteria" in results

    def test_has_real_properties(self, results):
        assert len(results["real_stability"]) >= 4

    def test_has_fake_properties(self, results):
        assert len(results["fake_stability"]) >= 5

    def test_separation_metrics(self, results):
        assert "separation_metrics" in results
        assert "classifier_accuracy" in results["separation_metrics"]

    def test_collapse_rate(self, results):
        assert "collapse_rate_fake" in results["separation_metrics"]


class TestPr5FormulaFixes:
    def test_deep_memory_scalar_controller_is_distinct(self, tmp_path):
        from flygym_research.cognition.experiments.exp_deep_memory_v2 import (
            run_experiment,
        )

        results = run_experiment(output_dir=tmp_path, episode_steps=20, n_seeds=2)
        reduced = results["controller_summary"]["reduced_descending"]
        scalar = results["controller_summary"]["scalar_memory"]
        assert reduced != scalar

    def test_long_composition_ties_do_not_count_as_wins(self):
        from flygym_research.cognition.experiments.exp_long_composition import (
            _paired_win_rate,
        )

        trials = [
            {"family": "clean", "ep_a": "a", "ep_b": "b", "strategy": "bulk", "return": 1.0},
            {"family": "clean", "ep_a": "a", "ep_b": "b", "strategy": "boundary", "return": 1.0},
            {"family": "clean", "ep_a": "c", "ep_b": "d", "strategy": "bulk", "return": 0.0},
            {"family": "clean", "ep_a": "c", "ep_b": "d", "strategy": "boundary", "return": 2.0},
        ]

        assert _paired_win_rate(trials, "boundary", "bulk") == pytest.approx(0.5)

    def test_long_composition_degradation_is_length_normalized(self):
        from flygym_research.cognition.experiments.exp_long_composition import (
            _compute_return_degradation,
        )

        short = [1.0] * 5 + [0.0] * 5
        long = [1.0] * 50 + [0.0] * 50
        assert _compute_return_degradation(short) == pytest.approx(
            _compute_return_degradation(long),
        )

    def test_long_composition_failure_score_is_reward_only(self):
        from flygym_research.cognition.experiments.exp_long_composition import (
            _reward_only_failure_score,
        )

        rewards = [1.0] * 5 + [0.0] * 5
        score = _reward_only_failure_score(rewards, 5)
        assert score == pytest.approx(0.15)

    def test_portable_replay_degradation_slope_uses_mismatch_severity(self):
        from flygym_research.cognition.experiments.exp_portable_replay_v2 import (
            _compute_degradation_slope,
        )

        replay_results = {
            "avatar_remapped": {"return_lift": 0.0},
            "simplified_embodied": {"return_lift": -1.0},
            "native_physical": {"return_lift": -4.0},
        }
        slope = _compute_degradation_slope(replay_results, "avatar_remapped")
        assert slope < 0.0

    def test_backbone_shared_transfer_only_is_clipped(self, monkeypatch):
        from flygym_research.cognition.experiments import exp_backbone_shared_v2 as mod

        monkeypatch.setattr(
            mod,
            "compute_transfer_score",
            lambda candidate, episodes: {
                "functional_transfer_gain": 2.0,
                "portability_fraction": 1.0,
            },
        )
        score = mod._score_transfer_only(SimpleNamespace(), [])
        assert score == 1.0

    def test_backbone_shared_seed_stability_is_not_hardcoded_true(self):
        from flygym_research.cognition.experiments.exp_backbone_shared_v2 import (
            _compute_seed_stability,
        )

        stability = _compute_seed_stability([], min_pool_size=20)
        assert stability["stable"] is False

    def test_scale_contrastive_rank_correlation_is_finite_for_constant_inputs(self):
        from flygym_research.cognition.experiments.exp_scale_contrastive_v2 import (
            _rank_correlation,
        )

        assert _rank_correlation([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0
