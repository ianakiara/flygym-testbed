from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..metrics.sleep_metrics import trajectory_equivalence_strength
from .trace_schema import SleepCandidate, TraceEpisode, TraceSegment


def segment_episode(
    episode: TraceEpisode,
    *,
    window_size: int = 16,
    stride: int = 8,
) -> list[TraceSegment]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be greater than zero.")
    segments: list[TraceSegment] = []
    for start in range(0, max(len(episode.transitions) - window_size + 1, 1), stride):
        end = min(start + window_size, len(episode.transitions))
        if end - start < 2:
            continue
        segments.append(
            TraceSegment(
                source_episode_id=episode.episode_id,
                start_step=start,
                end_step=end,
                transitions=episode.transitions[start:end],
                controller_name=episode.controller_name,
                world_mode=episode.world_mode,
                seed=episode.seed,
                perturbation_tag=episode.perturbation_tag,
                metadata={"ablation_channels": list(episode.ablation_channels)},
            )
        )
    return segments


def collect_segments(
    episodes: list[TraceEpisode],
    *,
    window_size: int = 16,
    stride: int = 8,
) -> list[TraceSegment]:
    return [
        segment
        for episode in episodes
        for segment in segment_episode(episode, window_size=window_size, stride=stride)
    ]


def _success_signature(episode: TraceEpisode) -> bool:
    return any(transition.terminated for transition in episode.transitions)


def _local_cluster_key(episode: TraceEpisode) -> tuple[str, str, tuple[str, ...], bool]:
    return (
        episode.world_mode,
        episode.perturbation_tag,
        episode.ablation_channels,
        _success_signature(episode),
    )


def _portable_signature(episode: TraceEpisode) -> tuple[str, tuple[str, ...], bool]:
    return (
        episode.perturbation_tag,
        episode.ablation_channels,
        _success_signature(episode),
    )


def build_equivalence_classes(
    episodes: list[TraceEpisode],
    *,
    min_equivalence_strength: float = 0.55,
) -> list[list[TraceEpisode]]:
    grouped: dict[tuple[str, str, tuple[str, ...], bool], list[TraceEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[_local_cluster_key(episode)].append(episode)

    classes: list[list[TraceEpisode]] = []
    for group in grouped.values():
        local_clusters: list[list[TraceEpisode]] = []
        for episode in group:
            placed = False
            for cluster in local_clusters:
                representative = cluster[0]
                score = trajectory_equivalence_strength(
                    representative.transitions,
                    episode.transitions,
                )["trajectory_equivalence_strength"]
                if score >= min_equivalence_strength:
                    cluster.append(episode)
                    placed = True
                    break
            if not placed:
                local_clusters.append([episode])
        classes.extend(local_clusters)
    return classes


def candidate_from_cluster(
    cluster: list[TraceEpisode],
    *,
    redundancy_tier: str = "local",
    portability_evidence: dict[str, Any] | None = None,
) -> SleepCandidate:
    representative = max(
        cluster,
        key=lambda episode: episode.summary_metrics.get("return", 0.0),
    )
    pairwise_scores = []
    for episode in cluster:
        if episode.episode_id == representative.episode_id:
            continue
        pairwise_scores.append(
            trajectory_equivalence_strength(
                representative.transitions,
                episode.transitions,
            )
        )
    world_modes = sorted({episode.world_mode for episode in cluster})
    evidence: dict[str, Any] = {
        "cluster_size": len(cluster),
        "world_mode": representative.world_mode,
        "world_modes": world_modes,
        "perturbation_tag": representative.perturbation_tag,
        "pairwise_scores": pairwise_scores,
        "ablation_channels": list(representative.ablation_channels),
    }
    if portability_evidence:
        evidence.update(portability_evidence)
    score_components = {
        "mean_equivalence_strength": float(
            np.mean([s["trajectory_equivalence_strength"] for s in pairwise_scores])
        ) if pairwise_scores else 1.0,
        "cluster_size": float(len(cluster)),
        "world_coverage": float(len(world_modes)),
    }
    portability = portability_evidence or {
        "world_modes": world_modes,
        "portability_fraction": 1.0 if redundancy_tier in {"portable", "universal"} else (1.0 if len(world_modes) > 1 else 0.0),
        "supporting_local_candidates": [],
    }
    return SleepCandidate(
        candidate_id=f"cand-{representative.episode_id}",
        representative_episode_id=representative.episode_id,
        member_episode_ids=[episode.episode_id for episode in cluster],
        evidence=evidence,
        score_components=score_components,
        redundancy_tier=redundancy_tier,
        portability_evidence=portability,
    )


def consolidate_redundancy_tiers(
    local_candidates: list[SleepCandidate],
    episodes: list[TraceEpisode],
    *,
    min_equivalence_strength: float = 0.55,
) -> list[SleepCandidate]:
    by_id = {episode.episode_id: episode for episode in episodes}
    total_worlds = sorted({episode.world_mode for episode in episodes})
    grouped: dict[tuple[str, tuple[str, ...], bool], list[list[SleepCandidate]]] = defaultdict(list)

    for candidate in local_candidates:
        representative = by_id[candidate.representative_episode_id]
        signature = _portable_signature(representative)
        placed = False
        for bucket in grouped[signature]:
            exemplar = by_id[bucket[0].representative_episode_id]
            score = trajectory_equivalence_strength(
                exemplar.transitions,
                representative.transitions,
            )["trajectory_equivalence_strength"]
            if score >= min_equivalence_strength:
                bucket.append(candidate)
                placed = True
                break
        if not placed:
            grouped[signature].append([candidate])

    consolidated: list[SleepCandidate] = []
    for buckets in grouped.values():
        for bucket in buckets:
            member_ids = sorted({episode_id for candidate in bucket for episode_id in candidate.member_episode_ids})
            cluster = [by_id[episode_id] for episode_id in member_ids]
            world_modes = sorted({episode.world_mode for episode in cluster})
            portability_fraction = len(world_modes) / max(len(total_worlds), 1)
            if len(world_modes) == len(total_worlds) and len(world_modes) > 1:
                redundancy_tier = "universal"
            elif len(world_modes) > 1:
                redundancy_tier = "portable"
            else:
                redundancy_tier = "local"
            candidate = candidate_from_cluster(
                cluster,
                redundancy_tier=redundancy_tier,
                portability_evidence={
                    "world_modes": world_modes,
                    "total_worlds": total_worlds,
                    "portability_fraction": portability_fraction,
                    "supporting_local_candidates": [item.candidate_id for item in bucket],
                },
            )
            candidate.residual_episode_ids = sorted(
                {episode_id for item in bucket for episode_id in item.residual_episode_ids}
            )
            candidate.retained_exception_rationale = {
                key: value
                for item in bucket
                for key, value in item.retained_exception_rationale.items()
            }
            candidate.evidence["local_cluster_ids"] = [item.candidate_id for item in bucket]
            candidate.score_components["portability_fraction"] = portability_fraction
            consolidated.append(candidate)
    return consolidated


def extract_sleep_candidates(
    episodes: list[TraceEpisode],
    *,
    min_equivalence_strength: float = 0.55,
) -> list[SleepCandidate]:
    local_candidates = [
        candidate_from_cluster(cluster)
        for cluster in build_equivalence_classes(
            episodes,
            min_equivalence_strength=min_equivalence_strength,
        )
    ]
    return consolidate_redundancy_tiers(
        local_candidates,
        episodes,
        min_equivalence_strength=min_equivalence_strength,
    )
