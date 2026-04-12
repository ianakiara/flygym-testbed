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
        raise ValueError("window_size and stride must be positive.")
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



def _cluster_key(episode: TraceEpisode) -> tuple[str, str, bool]:
    return (episode.world_mode, episode.perturbation_tag, _success_signature(episode))



def build_equivalence_classes(
    episodes: list[TraceEpisode],
    *,
    min_equivalence_strength: float = 0.55,
) -> list[list[TraceEpisode]]:
    grouped: dict[tuple[str, str, bool], list[TraceEpisode]] = defaultdict(list)
    for episode in episodes:
        grouped[_cluster_key(episode)].append(episode)

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



def candidate_from_cluster(cluster: list[TraceEpisode]) -> SleepCandidate:
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
    evidence: dict[str, Any] = {
        "cluster_size": len(cluster),
        "world_mode": representative.world_mode,
        "perturbation_tag": representative.perturbation_tag,
        "pairwise_scores": pairwise_scores,
    }
    score_components = {
        "mean_equivalence_strength": float(
            np.mean([s["trajectory_equivalence_strength"] for s in pairwise_scores])
        )
        if pairwise_scores
        else 1.0,
        "cluster_size": float(len(cluster)),
    }
    return SleepCandidate(
        candidate_id=f"cand-{representative.episode_id}",
        representative_episode_id=representative.episode_id,
        member_episode_ids=[episode.episode_id for episode in cluster],
        evidence=evidence,
        score_components=score_components,
    )



def extract_sleep_candidates(
    episodes: list[TraceEpisode],
    *,
    min_equivalence_strength: float = 0.55,
) -> list[SleepCandidate]:
    return [
        candidate_from_cluster(cluster)
        for cluster in build_equivalence_classes(
            episodes,
            min_equivalence_strength=min_equivalence_strength,
        )
    ]
