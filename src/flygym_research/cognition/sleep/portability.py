"""Portability scoring and candidate classification for cross-world transfer.

Classifies sleep candidates into three tiers:
- **Universal**: appears in clusters spanning 3+ worlds (highest value)
- **Portable**: appears in clusters spanning 2 worlds
- **Local**: appears in single-world clusters only

Also computes a per-candidate portability score for use in:
- Sleep retention priority
- Replay guidance ranking
- Transfer curriculum construction
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .trace_schema import SleepCandidate, TraceEpisode


class CandidateClass(str, Enum):
    """Classification tier for sleep candidates."""

    UNIVERSAL = "universal"   # 3+ worlds
    PORTABLE = "portable"     # 2 worlds
    LOCAL = "local"           # 1 world


@dataclass(slots=True)
class ClassifiedCandidate:
    """A sleep candidate with portability classification."""

    candidate: SleepCandidate
    candidate_class: CandidateClass
    portability_score: float
    world_count: int
    world_modes: list[str]
    mean_cross_world_strength: float


def classify_candidate(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
) -> ClassifiedCandidate:
    """Classify a single candidate by its cross-world coverage."""
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    world_modes = sorted(set(ep.world_mode for ep in members))
    world_count = len(world_modes)

    # Portability score: combines world coverage, cluster size, and equivalence strength
    mean_equiv = candidate.score_components.get("mean_equivalence_strength", 0.5)
    cluster_size = len(candidate.member_episode_ids)

    # Score components
    world_coverage = min(world_count / 3.0, 1.0)  # saturates at 3 worlds
    size_bonus = min(cluster_size / 10.0, 1.0)     # saturates at 10 members
    strength_factor = float(np.clip(mean_equiv, 0.0, 1.0))

    portability_score = (
        0.5 * world_coverage
        + 0.3 * strength_factor
        + 0.2 * size_bonus
    )

    # Mean cross-world strength: average equivalence strength for cross-world pairs
    cross_scores = candidate.score_components.get("cross_world_strengths", [])
    if isinstance(cross_scores, list) and cross_scores:
        mean_cw_strength = float(np.mean(cross_scores))
    else:
        mean_cw_strength = mean_equiv if world_count > 1 else 0.0

    # Classification
    if world_count >= 3:
        cclass = CandidateClass.UNIVERSAL
    elif world_count == 2:
        cclass = CandidateClass.PORTABLE
    else:
        cclass = CandidateClass.LOCAL

    return ClassifiedCandidate(
        candidate=candidate,
        candidate_class=cclass,
        portability_score=portability_score,
        world_count=world_count,
        world_modes=world_modes,
        mean_cross_world_strength=mean_cw_strength,
    )


def classify_all_candidates(
    candidates: list[SleepCandidate],
    episodes: list[TraceEpisode],
) -> list[ClassifiedCandidate]:
    """Classify all candidates and sort by portability score (descending)."""
    classified = [classify_candidate(c, episodes) for c in candidates]
    classified.sort(key=lambda c: c.portability_score, reverse=True)
    return classified


def portability_summary(
    classified: list[ClassifiedCandidate],
) -> dict[str, Any]:
    """Summary statistics for a classified candidate set."""
    n_universal = sum(1 for c in classified if c.candidate_class == CandidateClass.UNIVERSAL)
    n_portable = sum(1 for c in classified if c.candidate_class == CandidateClass.PORTABLE)
    n_local = sum(1 for c in classified if c.candidate_class == CandidateClass.LOCAL)

    scores = [c.portability_score for c in classified]
    return {
        "n_candidates": len(classified),
        "n_universal": n_universal,
        "n_portable": n_portable,
        "n_local": n_local,
        "universal_fraction": n_universal / max(len(classified), 1),
        "portable_fraction": n_portable / max(len(classified), 1),
        "local_fraction": n_local / max(len(classified), 1),
        "mean_portability_score": float(np.mean(scores)) if scores else 0.0,
        "max_portability_score": float(np.max(scores)) if scores else 0.0,
        "hierarchy": "universal > portable > local",
    }


def select_transfer_candidates(
    classified: list[ClassifiedCandidate],
    *,
    min_class: CandidateClass = CandidateClass.PORTABLE,
    max_candidates: int = 10,
) -> list[ClassifiedCandidate]:
    """Select candidates suitable for transfer experiments.

    Filters by minimum class tier and returns top candidates by portability.
    """
    tier_order = {CandidateClass.UNIVERSAL: 3, CandidateClass.PORTABLE: 2, CandidateClass.LOCAL: 1}
    min_tier = tier_order[min_class]

    eligible = [
        c for c in classified
        if tier_order[c.candidate_class] >= min_tier
    ]
    # Already sorted by portability_score descending
    return eligible[:max_candidates]
