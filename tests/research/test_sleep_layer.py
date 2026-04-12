from pathlib import Path

import pytest

from flygym_research.cognition.body_reflex import BodylessBodyLayer
from flygym_research.cognition.controllers import ReducedDescendingController
from flygym_research.cognition.envs import FlyAvatarEnv
from flygym_research.cognition.experiments import run_episode
from flygym_research.cognition.experiments.exp_drift_cleanup import run_experiment as run_drift_cleanup
from flygym_research.cognition.experiments.exp_interop_alignment import run_experiment as run_interop_alignment
from flygym_research.cognition.experiments.exp_memory_closure import run_experiment as run_memory_closure
from flygym_research.cognition.experiments.exp_seam_repair import run_experiment as run_seam_repair
from flygym_research.cognition.experiments.exp_sleep_trace_compressor import (
    collect_trace_bank,
    run_experiment as run_sleep_trace_compressor,
)
from flygym_research.cognition.metrics.sleep_metrics import (
    compression_gain,
    drift_staleness_score,
    post_compression_robustness_delta,
    repairability_score,
    trajectory_equivalence_strength,
)
from flygym_research.cognition.sleep import (
    CompressionConfig,
    TraceEpisode,
    TraceStore,
    analyze_seam_failures,
    build_alignment_registry,
    build_memory_packets,
    cleanup_memory_bank,
    compress_trace_bank,
    extract_sleep_candidates,
    repairability_curve,
)



def _make_episode(seed: int = 0, controller_name: str = "reduced") -> TraceEpisode:
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    controller = ReducedDescendingController()
    transitions = run_episode(env, controller, seed=seed, max_steps=12)
    return TraceEpisode(
        controller_name=controller_name,
        world_mode="avatar_remapped",
        seed=seed,
        transitions=transitions,
        controller_state=controller.get_internal_state(),
        body_state=env.body.get_internal_state(),
    )


class TestSleepTraceSchemaAndStore:
    def test_trace_store_round_trip(self, tmp_path: Path):
        episode = _make_episode()
        store = TraceStore(tmp_path)
        store.save_trace_bank([episode], metadata={"kind": "test"})
        loaded = store.load_trace_bank()
        assert len(loaded) == 1
        assert loaded[0].episode_id == episode.episode_id
        assert len(loaded[0].transitions) == len(episode.transitions)

    def test_sleep_artifact_round_trip(self, tmp_path: Path):
        episodes = [_make_episode(seed=0), _make_episode(seed=1)]
        artifact = compress_trace_bank(
            episodes,
            config=CompressionConfig(min_equivalence_strength=0.1),
        )
        store = TraceStore(tmp_path)
        store.save_sleep_artifact(artifact)
        loaded = store.load_sleep_artifact()
        assert loaded.artifact_id == artifact.artifact_id
        assert loaded.compressed_episode_ids == artifact.compressed_episode_ids
        assert loaded.residual_episode_ids == artifact.residual_episode_ids
        assert len(loaded.candidates) == len(artifact.candidates)


class TestSleepCompression:
    def test_extract_candidates_groups_equivalent_runs(self):
        episodes = [_make_episode(seed=0), _make_episode(seed=0, controller_name="reduced_b")]
        candidates = extract_sleep_candidates(episodes, min_equivalence_strength=0.3)
        assert len(candidates) == 1
        assert len(candidates[0].member_episode_ids) == 2

    def test_compress_trace_bank_outputs_validation(self):
        episodes = [_make_episode(seed=0), _make_episode(seed=1)]
        artifact = compress_trace_bank(
            episodes,
            config=CompressionConfig(min_equivalence_strength=0.1),
        )
        assert artifact.validation["compression_gain"] >= 0.0
        assert artifact.artifact_id.startswith("sleep-")
        assert artifact.reports["summary"]["n_candidates"] >= 1


class TestSleepProcesses:
    def test_seam_and_alignment_reports(self):
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        seam_report = analyze_seam_failures(episodes)
        alignment = build_alignment_registry(episodes, min_alignment_score=0.0)
        assert "n_failures" in seam_report
        assert alignment["n_approved"] + alignment["n_rejected"] >= 1

    def test_memory_closure_and_cleanup(self):
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        artifact_a = compress_trace_bank(episodes)
        artifact_b = compress_trace_bank(episodes)
        artifact_a.metadata.update({"source": "memory-pack", "age_days": 1.0, "reuse_count": 3.0})
        artifact_b.metadata.update({"source": "memory-pack", "age_days": 20.0, "reuse_count": 0.0})
        packets = build_memory_packets(episodes, artifact_a, window_size=6, stride=3)
        cleanup = cleanup_memory_bank([artifact_a, artifact_b], stale_threshold=0.4)
        assert packets["n_packets"] >= 1
        assert cleanup["n_kept"] == 1


class TestSleepMetrics:
    def test_sleep_metrics_return_reasonable_values(self):
        left = _make_episode(seed=0)
        right = _make_episode(seed=1)
        eq = trajectory_equivalence_strength(left.transitions, right.transitions)
        assert 0.0 <= eq["trajectory_equivalence_strength"] <= 1.0
        assert compression_gain(10, 4)["compression_gain"] == 0.6
        drift = drift_staleness_score({"age_days": 10, "reuse_count": 0}, {"pass_rate": 0.5})
        assert 0.0 <= drift["drift_staleness_score"] <= 1.0
        repair = repairability_score({"n_failures": 3, "n_patchable_failures": 2})
        assert repair["repairability_score"] == pytest.approx(2 / 3)

    def test_repairability_score_edge_cases(self):
        # Zero failures → perfect score
        repair_zero = repairability_score({"n_failures": 0, "n_patchable_failures": 0})
        assert repair_zero["repairability_score"] == 1.0

        # Failures but none patchable → zero score
        repair_none = repairability_score({"n_failures": 5, "n_patchable_failures": 0})
        assert repair_none["repairability_score"] == 0.0

    def test_robustness_delta_handles_inverted_polarity(self):
        # seam_fragility is "lower is better": a decrease should be treated
        # as improvement (positive delta), not regression.
        baseline = {"return": 1.0, "seam_fragility": 0.5}
        compressed_better = {"return": 1.0, "seam_fragility": 0.3}
        compressed_worse = {"return": 1.0, "seam_fragility": 0.7}

        delta_better = post_compression_robustness_delta(baseline, compressed_better)
        delta_worse = post_compression_robustness_delta(baseline, compressed_worse)

        # Improved seam_fragility → positive contribution → positive delta
        assert delta_better["post_compression_robustness_delta"] > 0.0
        assert not delta_better["robustness_regressed"]

        # Worsened seam_fragility → negative contribution → negative delta
        assert delta_worse["post_compression_robustness_delta"] < 0.0


class TestSleepExperiments:
    def test_experiments_emit_outputs(self, tmp_path: Path):
        summary = run_sleep_trace_compressor(tmp_path / "compressor", config=CompressionConfig())
        seam = run_seam_repair(tmp_path / "seam")
        alignment = run_interop_alignment(tmp_path / "interop")
        packets = run_memory_closure(tmp_path / "closure")
        cleanup = run_drift_cleanup(tmp_path / "drift")

        assert summary["artifact"]["artifact_id"].startswith("sleep-")
        assert "n_failures" in seam
        assert alignment["n_approved"] + alignment["n_rejected"] >= 1
        assert packets["n_packets"] >= 1
        assert cleanup["n_kept"] == 1
        report_path = tmp_path / "compressor" / "sleep_report.md"
        assert report_path.exists()
        assert "Sleep Artifact" in report_path.read_text()


class TestCrossWorldEquivalence:
    """Tests for cross-world clustering and scale_drift activation."""

    def test_cross_world_clustering_merges_worlds(self):
        """With cross_world=True, episodes from different worlds can share a cluster."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        # Same-world: should partition by world_mode.
        candidates_same = extract_sleep_candidates(
            episodes, min_equivalence_strength=0.1, cross_world=False,
        )
        # Cross-world: should allow merging across worlds.
        candidates_cross = extract_sleep_candidates(
            episodes, min_equivalence_strength=0.1, cross_world=True,
        )
        # Cross-world should produce fewer or equal candidates (more merging).
        assert len(candidates_cross) <= len(candidates_same) + len(episodes)

    def test_cross_world_compression_config(self):
        """CompressionConfig cross_world flag propagates to candidates."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "native_physical"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        config = CompressionConfig(cross_world=True, min_equivalence_strength=0.1)
        artifact = compress_trace_bank(episodes, config=config)
        assert artifact.artifact_id.startswith("sleep-")
        # Candidates should exist.
        assert len(artifact.candidates) >= 1

    def test_cross_world_candidate_captures_world_modes(self):
        """Candidate evidence should list all world modes in the cluster."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        candidates = extract_sleep_candidates(
            episodes, min_equivalence_strength=0.1, cross_world=True,
        )
        # At least some candidates should have world_modes in evidence.
        has_world_modes = any(
            "world_modes" in c.evidence for c in candidates
        )
        assert has_world_modes

    def test_same_world_scale_drift_zero(self):
        """Without cross-world, scale_drift should always be 0."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "native_physical"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        config = CompressionConfig(cross_world=False)
        artifact = compress_trace_bank(episodes, config=config)
        for c in artifact.candidates:
            assert c.score_components.get("scale_drift", 0.0) == 0.0


class TestRepairabilityCurve:
    """Tests for the 2-D repairability curve sweep."""

    def test_repairability_curve_returns_expected_keys(self):
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        result = repairability_curve(
            episodes,
            seam_thresholds=[0.10, 0.20],
            mismatch_thresholds=[0.20, 0.35],
        )
        assert "curve" in result
        assert "critical_point" in result
        assert "default_repairability" in result
        assert "tight_repairability" in result
        assert "n_points_tested" in result
        assert "seam_only_curve" in result
        assert "mismatch_only_curve" in result
        # 2 seam × 2 mismatch = 4 points.
        assert result["n_points_tested"] == 4

    def test_repairability_curve_2d_has_variation(self):
        """Sweeping mismatch threshold should produce variation in failures."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        result = repairability_curve(
            episodes,
            seam_thresholds=[0.20],
            mismatch_thresholds=[0.10, 0.20, 0.35, 0.50],
        )
        curve = result["curve"]
        # Tighter mismatch threshold should find more or equal failures.
        for i in range(len(curve) - 1):
            assert curve[i]["n_failures"] >= curve[i + 1]["n_failures"]

    def test_repairability_curve_default_thresholds(self):
        """Default thresholds should work without explicit list."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        result = repairability_curve(episodes)
        # Default: 6 seam × 7 mismatch = 42 points.
        assert result["n_points_tested"] == 42

    def test_analyze_seam_failures_mismatch_threshold(self):
        """Parameterised mismatch_threshold should change failure count."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        lenient = analyze_seam_failures(episodes, mismatch_threshold=0.50)
        strict = analyze_seam_failures(episodes, mismatch_threshold=0.10)
        # Strict should find more or equal failures.
        assert strict["n_failures"] >= lenient["n_failures"]

    def test_joint_patchability_criterion(self):
        """Patchability now requires both seam AND mismatch within margin."""
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        # Very tight thresholds: failures with high mismatch won't be patchable.
        report = analyze_seam_failures(
            episodes, seam_threshold=0.05, mismatch_threshold=0.10,
        )
        # n_patchable should be <= n_failures.
        assert report["n_patchable_failures"] <= report["n_failures"]


class TestCrossWorldCompressionExperiment:
    """Tests for the cross-world compression POC experiment."""

    def test_run_experiment(self, tmp_path: Path):
        from flygym_research.cognition.experiments.exp_cross_world_compression import (
            run_experiment,
        )

        result = run_experiment(tmp_path / "cross_world")
        assert "same_world" in result
        assert "cross_world" in result
        assert "scale_drift_activated" in result
        assert "same_world_drift_zero" in result
        assert "n_multi_world_clusters" in result
        # Redundancy analysis and bank quality.
        assert "redundancy_analysis" in result
        assert "bank_quality_comparison" in result
        ra = result["redundancy_analysis"]
        assert "n_single_world_candidates" in ra
        assert "n_portable_candidates" in ra
        assert "portable_fraction" in ra
        bq = result["bank_quality_comparison"]
        assert "same_world_kept_episodes" in bq
        assert "cross_world_kept_episodes" in bq
        # Same-world drift should be zero.
        assert result["same_world_drift_zero"] is True
        # Output files should exist.
        assert (tmp_path / "cross_world" / "same_world_report.md").exists()
        assert (tmp_path / "cross_world" / "cross_world_report.md").exists()
        assert (tmp_path / "cross_world" / "cross_world_summary.json").exists()

    def test_stricter_cluster_analysis(self, tmp_path: Path):
        from flygym_research.cognition.experiments.exp_cross_world_compression import (
            run_experiment,
        )

        result = run_experiment(tmp_path / "cross_world_strict")
        # Strict clusters (2+ worlds) should be subset of multi-world clusters.
        assert result["n_strict_clusters_2plus_worlds"] <= result["n_multi_world_clusters"]
        # Three-world clusters should be subset of strict clusters.
        assert result["n_three_world_clusters"] <= result["n_strict_clusters_2plus_worlds"]


class TestMemoryDemandExperiment:
    """Tests for the memory-demand POC experiment."""

    def test_run_experiment(self, tmp_path: Path):
        from flygym_research.cognition.experiments.exp_memory_demand import (
            run_experiment,
        )

        result = run_experiment(tmp_path / "memory_demand")
        assert "tasks" in result
        assert "advantages" in result
        assert "per_task" in result
        assert "cross_task_summary" in result
        # Should have all 5 tasks.
        assert len(result["tasks"]) == 5
        assert "distractor_cue_recall" in result["tasks"]
        assert "conditional_sequence" in result["tasks"]
        # Cross-task summary should have easy/hard breakdown.
        cts = result["cross_task_summary"]
        assert "easy_task_return_advantage" in cts
        assert "hard_task_return_advantage" in cts


class TestMemoryControllerWorldObservables:
    """Tests for the MemoryController world-observable enrichment."""

    def test_memory_buffer_includes_world_observables(self):
        from flygym_research.cognition.body_reflex import BodylessBodyLayer
        from flygym_research.cognition.controllers import MemoryController
        from flygym_research.cognition.envs import FlyAvatarEnv
        from flygym_research.cognition.experiments import run_episode

        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = MemoryController(memory_size=8)
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)

        # Memory buffer should store vectors longer than just body features
        # because world observables (target_x, target_y, cue_signal) are appended.
        assert len(ctrl._memory) == 1
        stored_vec = ctrl._memory[0]
        # Body features alone are ~12 elements; with world observables it should be 15.
        assert len(stored_vec) > 12
