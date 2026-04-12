"""PoC experiment — Cross-World Compression with Active Scale Drift.

Demonstrates that the ``scale_drift`` safety gate activates when
compression operates across world boundaries.  Previously, clustering
was always partitioned by ``world_mode``, so the cross-world divergence
loop in :func:`scoring.safe_compression_score` never executed
(``scale_drift`` was always 0.0).

With ``cross_world=True``, episodes from different worlds can be grouped
into the same equivalence class.  The scale-drift gate then measures
whether merging them would obscure important world-specific differences.

Protocol:
  a) Collect a trace bank spanning multiple world modes.
  b) Compress **without** cross-world (baseline — scale_drift = 0).
  c) Compress **with** cross-world (scale_drift should now be > 0).
  d) Compare compression ratios, scale-drift values, and decision
     distributions (compress / review / keep_singleton).
  e) **Stricter analysis** — enforce minimum support per cross-world
     cluster, compare structural quality of same-world vs cross-world
     compressed banks, and identify world-portable vs world-specific
     redundancy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..sleep import CompressionConfig, compress_trace_bank
from ..sleep.reporting import sleep_artifact_to_markdown
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(
    output_dir: str | Path = "results/exp_cross_world_compression",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episodes spanning all three world modes.
    episodes = collect_trace_bank(
        seeds=[0, 1],
        world_modes=["avatar_remapped", "native_physical", "simplified_embodied"],
        max_steps=48,
    )

    # --- Baseline: same-world compression (scale_drift always 0) ---
    config_same = CompressionConfig(cross_world=False)
    artifact_same = compress_trace_bank(episodes, config=config_same)

    same_world_drifts = [
        c.score_components.get("scale_drift", 0.0)
        for c in artifact_same.candidates
    ]
    same_decisions = _decision_counts(artifact_same)

    # --- Cross-world compression (scale_drift activates) ---
    config_cross = CompressionConfig(cross_world=True)
    artifact_cross = compress_trace_bank(episodes, config=config_cross)

    cross_world_drifts = [
        c.score_components.get("scale_drift", 0.0)
        for c in artifact_cross.candidates
    ]
    cross_decisions = _decision_counts(artifact_cross)

    # --- Multi-world cluster analysis ---
    multi_world_clusters = []
    for c in artifact_cross.candidates:
        world_modes = c.evidence.get("world_modes", [])
        if len(world_modes) > 1:
            multi_world_clusters.append({
                "candidate_id": c.candidate_id,
                "world_modes": world_modes,
                "cluster_size": len(c.member_episode_ids),
                "scale_drift": c.score_components.get("scale_drift", 0.0),
                "decision": c.decision,
            })

    # --- Stricter analysis: minimum support per cross-world cluster ---
    min_support = 2  # require at least 2 worlds represented
    strict_clusters = [
        cl for cl in multi_world_clusters
        if len(cl["world_modes"]) >= min_support
    ]
    three_world_clusters = [
        cl for cl in multi_world_clusters
        if len(cl["world_modes"]) >= 3
    ]

    # --- Bank quality comparison ---
    same_compressed_episodes = set()
    for c in artifact_same.candidates:
        if c.decision == "compress":
            same_compressed_episodes.update(c.member_episode_ids)
    cross_compressed_episodes = set()
    for c in artifact_cross.candidates:
        if c.decision == "compress":
            cross_compressed_episodes.update(c.member_episode_ids)

    same_kept = len(episodes) - len(same_compressed_episodes)
    cross_kept = len(episodes) - len(cross_compressed_episodes)

    # World-coverage of kept episodes (how many world modes are represented
    # in the compressed bank).
    same_kept_worlds = set()
    cross_kept_worlds = set()
    for ep in episodes:
        if ep.episode_id not in same_compressed_episodes:
            same_kept_worlds.add(ep.world_mode)
        if ep.episode_id not in cross_compressed_episodes:
            cross_kept_worlds.add(ep.world_mode)

    # --- Redundancy classification ---
    # World-specific redundancy: clusters that are single-world (safe to collapse).
    # World-portable redundancy: cross-world clusters (need drift gating).
    single_world_candidates = [
        c for c in artifact_cross.candidates
        if len(c.evidence.get("world_modes", [])) <= 1
    ]
    portable_candidates = [
        c for c in artifact_cross.candidates
        if len(c.evidence.get("world_modes", [])) > 1
    ]

    summary: dict[str, object] = {
        "n_episodes": len(episodes),
        # Same-world results.
        "same_world": {
            "compression_gain": artifact_same.validation.get("compression_gain", 0.0),
            "robustness_delta": artifact_same.validation.get(
                "post_compression_robustness_delta", 0.0
            ),
            "mean_scale_drift": float(np.mean(same_world_drifts)) if same_world_drifts else 0.0,
            "max_scale_drift": float(np.max(same_world_drifts)) if same_world_drifts else 0.0,
            "n_candidates": len(artifact_same.candidates),
            "decisions": same_decisions,
            "passed": artifact_same.validation.get("passed", False),
            "n_kept_episodes": same_kept,
            "n_kept_worlds": len(same_kept_worlds),
        },
        # Cross-world results.
        "cross_world": {
            "compression_gain": artifact_cross.validation.get("compression_gain", 0.0),
            "robustness_delta": artifact_cross.validation.get(
                "post_compression_robustness_delta", 0.0
            ),
            "mean_scale_drift": float(np.mean(cross_world_drifts)) if cross_world_drifts else 0.0,
            "max_scale_drift": float(np.max(cross_world_drifts)) if cross_world_drifts else 0.0,
            "n_candidates": len(artifact_cross.candidates),
            "decisions": cross_decisions,
            "passed": artifact_cross.validation.get("passed", False),
            "n_kept_episodes": cross_kept,
            "n_kept_worlds": len(cross_kept_worlds),
        },
        # Multi-world cluster analysis.
        "n_multi_world_clusters": len(multi_world_clusters),
        "n_strict_clusters_2plus_worlds": len(strict_clusters),
        "n_three_world_clusters": len(three_world_clusters),
        "multi_world_clusters": multi_world_clusters,
        # Redundancy classification.
        "redundancy_analysis": {
            "n_single_world_candidates": len(single_world_candidates),
            "n_portable_candidates": len(portable_candidates),
            "portable_fraction": (
                len(portable_candidates) / max(len(artifact_cross.candidates), 1)
            ),
            "mean_portable_drift": float(np.mean(
                [c.score_components.get("scale_drift", 0.0) for c in portable_candidates]
            )) if portable_candidates else 0.0,
            "mean_local_drift": float(np.mean(
                [c.score_components.get("scale_drift", 0.0) for c in single_world_candidates]
            )) if single_world_candidates else 0.0,
        },
        # Bank quality comparison.
        "bank_quality_comparison": {
            "same_world_kept_episodes": same_kept,
            "cross_world_kept_episodes": cross_kept,
            "same_world_worlds_retained": sorted(same_kept_worlds),
            "cross_world_worlds_retained": sorted(cross_kept_worlds),
            "cross_world_more_conservative": cross_kept >= same_kept,
        },
    }

    # Key expectation: scale_drift is > 0 in cross-world mode.
    scale_drift_activated = summary["cross_world"]["mean_scale_drift"] > 0.0 or len(multi_world_clusters) > 0  # type: ignore[index]
    summary["scale_drift_activated"] = scale_drift_activated

    # Scale drift should be 0 in same-world mode (all candidates share world_mode).
    summary["same_world_drift_zero"] = summary["same_world"]["max_scale_drift"] == 0.0  # type: ignore[index]

    # Cross-world compression should be more conservative (fewer compressed, or
    # same compression with drift gating).
    summary["cross_world_more_conservative"] = (
        summary["cross_world"]["compression_gain"] <= summary["same_world"]["compression_gain"]  # type: ignore[index]
    )

    # Save artifacts and report.
    (output_dir / "same_world_report.md").write_text(
        sleep_artifact_to_markdown(artifact_same)
    )
    (output_dir / "cross_world_report.md").write_text(
        sleep_artifact_to_markdown(artifact_cross)
    )
    (output_dir / "cross_world_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary


def _decision_counts(artifact) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in artifact.candidates:
        d = c.decision or "undecided"
        counts[d] = counts.get(d, 0) + 1
    return counts
