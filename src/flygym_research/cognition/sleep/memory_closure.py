from __future__ import annotations

from collections import defaultdict

import numpy as np

from .equivalence import collect_segments
from .trace_schema import SleepArtifact, TraceEpisode, TraceSegment



def _segment_label(segment: TraceSegment) -> str:
    rewards = np.array([transition.reward for transition in segment.transitions], dtype=np.float64)
    reward_trend = "flat"
    if len(rewards) >= 2:
        delta = float(rewards[-1] - rewards[0])
        if delta > 0.1:
            reward_trend = "improving"
        elif delta < -0.1:
            reward_trend = "degrading"
    external_events = sum(
        1 for transition in segment.transitions if transition.info.get("external_world_event")
    )
    event_tag = "perturbed" if external_events else "stable"
    return f"{segment.world_mode}:{segment.controller_name}:{reward_trend}:{event_tag}"



def build_memory_packets(
    episodes: list[TraceEpisode],
    artifact: SleepArtifact | None = None,
    *,
    window_size: int = 16,
    stride: int = 8,
) -> dict[str, object]:
    segments = collect_segments(episodes, window_size=window_size, stride=stride)
    grouped: dict[str, list[TraceSegment]] = defaultdict(list)
    for segment in segments:
        grouped[_segment_label(segment)].append(segment)

    canonical_packets: list[dict[str, object]] = []
    for label, members in sorted(grouped.items()):
        representative = max(
            members,
            key=lambda segment: segment.summary_metrics.get("return", 0.0),
        )
        canonical_packets.append(
            {
                "label": label,
                "segment_id": representative.segment_id,
                "source_episode_id": representative.source_episode_id,
                "n_members": len(members),
                "summary_metrics": representative.summary_metrics,
                "provenance": [segment.segment_id for segment in members],
            }
        )

    return {
        "canonical_packets": canonical_packets,
        "n_packets": len(canonical_packets),
        "artifact_id": artifact.artifact_id if artifact is not None else None,
        "residual_episode_ids": artifact.residual_episode_ids if artifact else [],
    }
