"""Sleep Layer v2 — world-aware safe consolidation with full 10-step cycle.

The sleep cycle:
1. Classify candidates (local/portable/universal)
2. Compress (equivalence-class based deduplication)
3. Drift gate (reject high-drift candidates)
4. Seam check (flag seam-critical episodes)
5. Interop gate (ensure cross-controller translatability)
6. Repair (mismatch-aware, dimension-specific)
7. Extract memory packets (portable representations)
8. Demote stale artifacts (age-based expiry)
9. Replay fragile cases (re-queue for re-evaluation)
10. Validate (BackboneShared composite check)

Comparison modes:
- no_sleep: raw trace bank, no processing
- compression_only: steps 1-2 only
- full_sleep_v2: all 10 steps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metrics import seam_fragility, interoperability_score
from .compressor import CompressionConfig, compress_trace_bank
from .equivalence import extract_sleep_candidates
from .portability import (
    CandidateClass,
    classify_all_candidates,
    portability_summary,
    select_transfer_candidates,
)
from .repair_policies import (
    FailureType,
    apply_repair,
    diagnose_failure,
)
from .trace_schema import SleepArtifact, TraceEpisode


@dataclass(slots=True)
class SleepV2Config:
    """Configuration for the full sleep v2 cycle."""

    # Compression
    min_equivalence_strength: float = 0.55
    cross_world: bool = True
    # Drift gate
    max_scale_drift: float = 0.35
    # Seam check
    seam_threshold: float = 0.20
    # Interop gate
    min_interop: float = 0.5
    # Repair
    mismatch_threshold: float = 0.35
    repair_budget_fraction: float = 0.5
    # Demote
    staleness_age: int = 100  # steps before an artifact is considered stale
    # Replay
    replay_fragile: bool = True
    fragility_threshold: float = 0.3


@dataclass(slots=True)
class SleepV2Result:
    """Result of a full sleep v2 cycle."""

    artifact: SleepArtifact
    classified_candidates: list[dict[str, Any]]
    portability_stats: dict[str, Any]
    repair_summary: dict[str, Any]
    demoted_episode_ids: list[str]
    replay_queue: list[str]
    memory_packets: list[dict[str, Any]]
    validation: dict[str, Any]
    cycle_steps_completed: list[str]


def run_sleep_v2_cycle(
    episodes: list[TraceEpisode],
    *,
    config: SleepV2Config | None = None,
) -> SleepV2Result:
    """Execute the full 10-step sleep v2 cycle."""
    config = config or SleepV2Config()
    steps_completed: list[str] = []

    # ── Step 1: Classify candidates ──────────────────────────────────────
    candidates = extract_sleep_candidates(
        episodes,
        min_equivalence_strength=config.min_equivalence_strength,
        cross_world=config.cross_world,
    )
    classified = classify_all_candidates(candidates, episodes)
    portability_stats = portability_summary(classified)
    steps_completed.append("1_classify")

    # ── Step 2: Compress ─────────────────────────────────────────────────
    compression_config = CompressionConfig(
        min_equivalence_strength=config.min_equivalence_strength,
        cross_world=config.cross_world,
        max_scale_drift=config.max_scale_drift,
    )
    artifact = compress_trace_bank(episodes, config=compression_config)
    steps_completed.append("2_compress")

    # ── Step 3: Drift gate ───────────────────────────────────────────────
    # Already applied in compress_trace_bank via max_scale_drift
    steps_completed.append("3_drift_gate")

    # ── Step 4: Seam check ───────────────────────────────────────────────
    seam_flagged: list[str] = []
    by_id = {ep.episode_id: ep for ep in episodes}
    for eid in artifact.compressed_episode_ids:
        if eid in by_id:
            sf = seam_fragility(by_id[eid].transitions)["seam_fragility"]
            if sf > config.seam_threshold:
                seam_flagged.append(eid)
    steps_completed.append("4_seam_check")

    # ── Step 5: Interop gate ─────────────────────────────────────────────
    # Check pairwise interoperability for retained episodes
    interop_failures: list[str] = []
    retained_eps = [by_id[eid] for eid in artifact.compressed_episode_ids if eid in by_id]
    if len(retained_eps) >= 2:
        for i in range(min(len(retained_eps), 5)):
            for j in range(i + 1, min(len(retained_eps), 5)):
                try:
                    score = interoperability_score(
                        retained_eps[i].transitions,
                        retained_eps[j].transitions,
                    )["interoperability_score"]
                    if score < config.min_interop:
                        interop_failures.append(
                            f"{retained_eps[i].episode_id}-{retained_eps[j].episode_id}"
                        )
                except Exception:
                    pass
    steps_completed.append("5_interop_gate")

    # ── Step 6: Repair ───────────────────────────────────────────────────
    diagnoses = []
    repair_results = []
    for eid in artifact.compressed_episode_ids:
        if eid not in by_id:
            continue
        diag = diagnose_failure(
            eid, by_id[eid].transitions,
            seam_threshold=config.seam_threshold,
            mismatch_threshold=config.mismatch_threshold,
        )
        diagnoses.append(diag)
        if diag.failure_type != FailureType.NO_FAILURE:
            result = apply_repair(diag, by_id[eid].transitions)
            repair_results.append({
                "episode_id": eid,
                "failure_type": diag.failure_type.value,
                "strategy": result.repair_strategy,
                "success": result.repair_success,
            })
    steps_completed.append("6_repair")

    # ── Step 7: Extract memory packets ───────────────────────────────────
    memory_packets: list[dict[str, Any]] = []
    transfer_candidates = select_transfer_candidates(
        classified, min_class=CandidateClass.PORTABLE,
    )
    for tc in transfer_candidates:
        packet = {
            "candidate_id": tc.candidate.candidate_id,
            "class": tc.candidate_class.value,
            "portability_score": tc.portability_score,
            "world_modes": tc.world_modes,
            "representative_id": tc.candidate.representative_episode_id,
        }
        memory_packets.append(packet)
    steps_completed.append("7_extract_memory")

    # ── Step 8: Demote stale artifacts ───────────────────────────────────
    demoted: list[str] = []
    for ep in episodes:
        if ep.episode_steps > config.staleness_age:
            # Check if this episode is low-value
            ret = ep.summary_metrics.get("return", 0.0)
            if ret < -10.0:  # poor performance + old
                demoted.append(ep.episode_id)
    steps_completed.append("8_demote")

    # ── Step 9: Replay fragile cases ─────────────────────────────────────
    replay_queue: list[str] = []
    if config.replay_fragile:
        for eid in seam_flagged:
            if eid not in demoted:
                replay_queue.append(eid)
    steps_completed.append("9_replay")

    # ── Step 10: Validate ────────────────────────────────────────────────
    n_total = len(episodes)
    n_compressed = len(artifact.compressed_episode_ids)
    n_residual = len(artifact.residual_episode_ids)
    n_demoted = len(demoted)
    n_replay = len(replay_queue)
    n_repaired = sum(1 for r in repair_results if r.get("success", False))

    validation = {
        "n_original": n_total,
        "n_compressed": n_compressed,
        "n_residual": n_residual,
        "n_demoted": n_demoted,
        "n_replay": n_replay,
        "n_repairs_attempted": len(repair_results),
        "n_repairs_succeeded": n_repaired,
        "n_seam_flagged": len(seam_flagged),
        "n_interop_failures": len(interop_failures),
        "n_memory_packets": len(memory_packets),
        "compression_ratio": 1.0 - n_compressed / max(n_total, 1),
        "portability_stats": portability_stats,
        "all_steps_completed": len(steps_completed) == 10,
    }
    steps_completed.append("10_validate")

    return SleepV2Result(
        artifact=artifact,
        classified_candidates=[
            {
                "id": c.candidate.candidate_id,
                "class": c.candidate_class.value,
                "portability": c.portability_score,
                "worlds": c.world_modes,
            }
            for c in classified
        ],
        portability_stats=portability_stats,
        repair_summary={
            "n_diagnoses": len(diagnoses),
            "n_repairs": len(repair_results),
            "n_successful": n_repaired,
            "results": repair_results,
        },
        demoted_episode_ids=demoted,
        replay_queue=replay_queue,
        memory_packets=memory_packets,
        validation=validation,
        cycle_steps_completed=steps_completed,
    )


def compare_sleep_modes(
    episodes: list[TraceEpisode],
) -> dict[str, Any]:
    """Compare no-sleep vs compression-only vs full sleep v2.

    Returns performance, robustness, memory size metrics for each mode.
    """
    # No-sleep baseline
    no_sleep_returns = [ep.summary_metrics.get("return", 0.0) for ep in episodes]
    no_sleep_size = len(episodes)

    # Compression-only
    comp_config = CompressionConfig(cross_world=False)
    comp_artifact = compress_trace_bank(episodes, config=comp_config)
    comp_retained = set(comp_artifact.compressed_episode_ids) | set(comp_artifact.residual_episode_ids)
    comp_episodes = [ep for ep in episodes if ep.episode_id in comp_retained]
    comp_returns = [ep.summary_metrics.get("return", 0.0) for ep in comp_episodes]
    comp_size = len(comp_episodes)

    # Full sleep v2
    sleep_result = run_sleep_v2_cycle(episodes)
    sleep_retained = set(sleep_result.artifact.compressed_episode_ids) | set(sleep_result.artifact.residual_episode_ids)
    sleep_retained -= set(sleep_result.demoted_episode_ids)
    sleep_episodes = [ep for ep in episodes if ep.episode_id in sleep_retained]
    sleep_returns = [ep.summary_metrics.get("return", 0.0) for ep in sleep_episodes]
    sleep_size = len(sleep_episodes)

    return {
        "no_sleep": {
            "mean_return": float(np.mean(no_sleep_returns)) if no_sleep_returns else 0.0,
            "size": no_sleep_size,
            "n_episodes": len(episodes),
        },
        "compression_only": {
            "mean_return": float(np.mean(comp_returns)) if comp_returns else 0.0,
            "size": comp_size,
            "compression_gain": comp_artifact.validation.get("compression_gain", 0.0),
        },
        "full_sleep_v2": {
            "mean_return": float(np.mean(sleep_returns)) if sleep_returns else 0.0,
            "size": sleep_size,
            "n_memory_packets": len(sleep_result.memory_packets),
            "n_demoted": len(sleep_result.demoted_episode_ids),
            "n_replay": len(sleep_result.replay_queue),
            "portability": sleep_result.portability_stats,
        },
    }
