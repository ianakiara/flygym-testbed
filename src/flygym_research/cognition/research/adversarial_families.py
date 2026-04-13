"""Adversarial family generators for PR #5.

Generates adversarial candidates designed to fool scoring models:
  - reversed target bias mid-episode
  - conflicting world parameters
  - partial trajectory splice
  - seam-only corruption
  - overfit portable
  - compressed but unstable
"""

from __future__ import annotations

import numpy as np

from ..sleep.trace_schema import SleepCandidate, TraceEpisode


def generate_adversarial_family(
    candidates: list[SleepCandidate],
    episodes: list[TraceEpisode],
    *,
    rng: np.random.Generator | None = None,
    n_per_type: int = 10,
) -> list[tuple[SleepCandidate, str]]:
    """Generate adversarial candidates across multiple failure modes.

    Returns list of (candidate, adversarial_type) tuples.
    """
    rng = rng or np.random.default_rng(99)
    results: list[tuple[SleepCandidate, str]] = []
    source = candidates[: min(len(candidates), n_per_type * 3)]

    for i, cand in enumerate(source):
        if len(results) >= n_per_type * 6:
            break

        # 1. Overfit portable — inflated portability, no real transfer
        results.append((
            SleepCandidate(
                candidate_id=f"adv-overfit-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "overfit_portable"},
                score_components={
                    **cand.score_components,
                    "portability_fraction": 1.0,
                    "mean_equivalence_strength": 0.95,
                },
                redundancy_tier="universal",
                portability_evidence={
                    **cand.portability_evidence,
                    "portability_fraction": 1.0,
                    "world_modes": [
                        "avatar_remapped",
                        "native_physical",
                        "simplified_embodied",
                    ],
                },
            ),
            "adversarial_overfit",
        ))

        # 2. Compressed but unstable — high compression, high collapse risk
        results.append((
            SleepCandidate(
                candidate_id=f"adv-unstable-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "compressed_unstable"},
                score_components={
                    **cand.score_components,
                    "mean_equivalence_strength": float(rng.uniform(0.88, 0.97)),
                    "cluster_size": float(rng.uniform(8.0, 15.0)),
                    "scale_drift": float(rng.uniform(0.3, 0.6)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "adversarial_unstable",
        ))

        # 3. High-return brittle — great returns but structurally fragile
        results.append((
            SleepCandidate(
                candidate_id=f"adv-brittle-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "high_return_brittle"},
                score_components={
                    **cand.score_components,
                    "mean_return": float(rng.uniform(50.0, 100.0)),
                    "seam_risk": float(rng.uniform(0.4, 0.7)),
                    "interop_loss": float(rng.uniform(0.3, 0.5)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "adversarial_brittle",
        ))

        # 4. Seam-kill-test — passes all local metrics, kills at seam
        results.append((
            SleepCandidate(
                candidate_id=f"adv-seamkill-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "seam_kill_test"},
                score_components={
                    **cand.score_components,
                    "redundancy_score": float(rng.uniform(0.7, 0.9)),
                    "seam_risk": float(rng.uniform(0.5, 0.8)),
                    "interop_loss": float(rng.uniform(0.0, 0.1)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "adversarial_seam_kill",
        ))

        # 5. Memory-fragile — good short-term, degrades with horizon
        results.append((
            SleepCandidate(
                candidate_id=f"adv-memfrag-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "memory_fragile"},
                score_components={
                    **cand.score_components,
                    "degeneracy_penalty": float(rng.uniform(0.2, 0.45)),
                    "scale_drift": float(rng.uniform(0.2, 0.4)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "adversarial_memory_fragile",
        ))

        # 6. Interop-fragile — seems aligned, fails cross-controller
        results.append((
            SleepCandidate(
                candidate_id=f"adv-interopfrag-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "interop_fragile"},
                score_components={
                    **cand.score_components,
                    "interop_loss": float(rng.uniform(0.35, 0.65)),
                    "redundancy_score": float(rng.uniform(0.6, 0.85)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "adversarial_interop_fragile",
        ))

    return results
