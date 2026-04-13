from pathlib import Path

import pytest

from flygym_research.cognition.body_reflex import BodylessBodyLayer
from flygym_research.cognition.controllers import ReducedDescendingController
from flygym_research.cognition.envs import FlyAvatarEnv
from flygym_research.cognition.experiments import run_episode
from flygym_research.cognition.experiments.exp_composition_gluing import run_experiment as run_composition_gluing
from flygym_research.cognition.experiments.exp_drift_cleanup import run_experiment as run_drift_cleanup
from flygym_research.cognition.experiments.exp_interop_alignment import run_experiment as run_interop_alignment
from flygym_research.cognition.experiments.exp_learned_repairability import run_experiment as run_learned_repairability
from flygym_research.cognition.experiments.exp_memory_closure import run_experiment as run_memory_closure
from flygym_research.cognition.experiments.exp_portable_replay_benchmark import run_experiment as run_portable_replay
from flygym_research.cognition.experiments.exp_seam_repair import run_experiment as run_seam_repair
from flygym_research.cognition.experiments.exp_selective_memory_benchmark import run_experiment as run_selective_memory
from flygym_research.cognition.experiments.exp_sleep_retention_policy import run_experiment as run_retention_policy
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
    backbone_shared_score,
    benchmark_portable_replay,
    build_alignment_registry,
    build_memory_packets,
    cleanup_memory_bank,
    compress_trace_bank,
    extract_sleep_candidates,
    safe_compression_score,
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
        assert loaded.candidates[0].score_components["shared_structure_regime"] == artifact.candidates[0].score_components["shared_structure_regime"]


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


    def test_candidates_include_redundancy_tiers(self):
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied", "native_physical"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        artifact = compress_trace_bank(episodes, config=CompressionConfig(min_equivalence_strength=0.1))
        tiers = {candidate.redundancy_tier for candidate in artifact.candidates}
        assert tiers <= {"local", "portable", "universal"}
        assert all("backbone_shared_score" in candidate.score_components for candidate in artifact.candidates)
        assert all("safe_compression_score" in candidate.score_components for candidate in artifact.candidates)


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


    def test_portable_replay_updates_functional_utility(self):
        episodes = collect_trace_bank(
            seeds=[0],
            world_modes=["avatar_remapped", "simplified_embodied", "native_physical"],
            ablations=[frozenset()],
            perturbation_tags=["baseline"],
            max_steps=10,
        )
        artifact = compress_trace_bank(episodes)
        replay = benchmark_portable_replay(episodes, artifact)
        assert replay["summary"]["n_candidates"] >= 1
        assert all("functional_transfer_gain" in candidate.functional_utility for candidate in artifact.candidates)


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
        artifact = compress_trace_bank([left, right], config=CompressionConfig(min_equivalence_strength=0.1))
        score = backbone_shared_score(artifact.candidates[0], [left, right])
        baseline = safe_compression_score(artifact.candidates[0], [left, right])
        assert "backbone_shared_score" in score
        assert "safe_compression_score" in baseline
        assert baseline["safe_compression_score"] != score["backbone_shared_score"]

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
        portable = run_portable_replay(tmp_path / "portable")
        retention = run_retention_policy(tmp_path / "retention")
        memory = run_selective_memory(tmp_path / "memory")
        repairability = run_learned_repairability(tmp_path / "repairability")
        composition = run_composition_gluing(tmp_path / "composition")

        assert summary["artifact"]["artifact_id"].startswith("sleep-")
        assert "n_failures" in seam
        assert alignment["n_approved"] + alignment["n_rejected"] >= 1
        assert packets["n_packets"] >= 1
        assert cleanup["n_kept"] == 1
        assert portable["portable_replay"]["summary"]["n_candidates"] >= 1
        assert retention["n_candidates"] >= 1
        assert memory["summary_rows"]
        assert "learned_beats_baseline" in repairability or repairability["status"] == "insufficient_data"
        assert composition["summary"]
        report_path = tmp_path / "compressor" / "sleep_report.md"
        assert report_path.exists()
        assert "Sleep Artifact" in report_path.read_text()
