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
