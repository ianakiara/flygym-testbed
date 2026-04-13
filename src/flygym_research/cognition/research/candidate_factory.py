"""Candidate factory for PR #5 — generates large, labeled candidate pools.

Produces 200–400 candidates with explicit family labels:
  A. valid_shared          — good seam + interop + low drift + good transfer
  B. private_only_coherent — high Ω but low shared utility, bad interop
  C. seam_fragile          — internally coherent, collapses at handoff
  D. scale_artifact        — works at one granularity only
  E. transfer_fragile      — good same-world, bad cross-world replay
  F. degenerate_converger  — superficially aligned, collapse under stress
  G. false_portable        — cluster similarity, no functional utility
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..metrics import interoperability_score, seam_fragility
from ..sleep import compress_trace_bank
from ..sleep.trace_schema import SleepCandidate, TraceEpisode


# ---------------------------------------------------------------------------
# Family labelling
# ---------------------------------------------------------------------------

FAMILY_LABELS = [
    "valid_shared",
    "private_only_coherent",
    "seam_fragile",
    "scale_artifact",
    "transfer_fragile",
    "degenerate_converger",
    "false_portable",
    "ambiguous",
]


def label_candidate_family(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
) -> str:
    """Assign a ground-truth family label based on structural properties."""
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    representative = by_id.get(candidate.representative_episode_id)
    if not representative or not members:
        return "ambiguous"

    # Seam fragility
    seam_scores = [
        seam_fragility(m.transitions).get("seam_fragility", 0.0) for m in members
    ]
    mean_seam = float(np.mean(seam_scores)) if seam_scores else 0.0

    # Interop loss
    interop_losses = []
    for m in members:
        if m.episode_id == representative.episode_id:
            continue
        interop = interoperability_score(representative.transitions, m.transitions)
        interop_losses.append(1.0 - float(interop.get("interoperability_score", 1.0)))
    mean_interop_loss = float(np.mean(interop_losses)) if interop_losses else 0.0

    # World coverage
    world_modes = {m.world_mode for m in members}

    # Score components
    scale_drift = float(candidate.score_components.get("scale_drift", 0.0))
    portability_frac = float(candidate.score_components.get("portability_fraction", 0.0))
    degeneracy = float(candidate.score_components.get("degeneracy_penalty", 0.0))

    # Transfer check — cross-world members have good returns?
    world_returns: dict[str, list[float]] = defaultdict(list)
    for m in members:
        world_returns[m.world_mode].append(m.summary_metrics.get("return", 0.0))
    cross_world_variance = 0.0
    if len(world_returns) >= 2:
        per_world_means = [float(np.mean(v)) for v in world_returns.values()]
        cross_world_variance = float(np.std(per_world_means))

    # Classification rules (ordered by specificity)
    if degeneracy > 0.25 and mean_interop_loss > 0.2:
        return "degenerate_converger"
    if mean_seam >= 0.2 and mean_interop_loss < 0.2:
        return "seam_fragile"
    if mean_interop_loss >= 0.25 and mean_seam < 0.15:
        return "private_only_coherent"
    if scale_drift >= 0.2 or (len(world_modes) <= 1 and mean_seam < 0.15):
        return "scale_artifact"
    if portability_frac >= 0.34 and cross_world_variance > 0.3:
        return "transfer_fragile"
    if portability_frac >= 0.34 and mean_interop_loss > 0.15:
        return "false_portable"
    if mean_seam < 0.15 and mean_interop_loss < 0.2 and scale_drift < 0.15:
        return "valid_shared"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Synthetic adversarial candidate injection
# ---------------------------------------------------------------------------


def _inject_synthetic_adversarial(
    candidates: list[SleepCandidate],
    episodes: list[TraceEpisode],
    *,
    rng: np.random.Generator | None = None,
    n_per_type: int = 15,
) -> list[tuple[SleepCandidate, str]]:
    """Create synthetic adversarial candidates from existing ones."""
    rng = rng or np.random.default_rng(42)
    adversarial: list[tuple[SleepCandidate, str]] = []
    source = candidates[: min(len(candidates), n_per_type * 2)]

    for i, cand in enumerate(source):
        if len(adversarial) >= n_per_type * 6:
            break

        # Type F: Degenerate converger — inflate equivalence, keep high degeneracy
        adversarial.append((
            SleepCandidate(
                candidate_id=f"adv-degen-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "degenerate_converger"},
                score_components={
                    **cand.score_components,
                    "mean_equivalence_strength": 0.92,
                    "degeneracy_penalty": float(rng.uniform(0.3, 0.6)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "degenerate_converger",
        ))

        # Type G: False portable — high portability but no functional utility
        adversarial.append((
            SleepCandidate(
                candidate_id=f"adv-falseport-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "false_portable"},
                score_components={
                    **cand.score_components,
                    "portability_fraction": float(rng.uniform(0.6, 1.0)),
                    "functional_transfer_gain": float(rng.uniform(-0.3, 0.0)),
                },
                redundancy_tier="portable",
            ),
            "false_portable",
        ))

        # Type B: Private-only coherent — high Ω, bad interop
        adversarial.append((
            SleepCandidate(
                candidate_id=f"adv-private-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "private_only_coherent"},
                score_components={
                    **cand.score_components,
                    "redundancy_score": float(rng.uniform(0.7, 0.95)),
                    "interop_loss": float(rng.uniform(0.3, 0.6)),
                },
                redundancy_tier="local",
            ),
            "private_only_coherent",
        ))

        # Type C: Seam-fragile
        adversarial.append((
            SleepCandidate(
                candidate_id=f"adv-seamfrag-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "seam_fragile"},
                score_components={
                    **cand.score_components,
                    "seam_risk": float(rng.uniform(0.25, 0.55)),
                    "interop_loss": float(rng.uniform(0.0, 0.15)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "seam_fragile",
        ))

        # Type D: Scale-artifact
        adversarial.append((
            SleepCandidate(
                candidate_id=f"adv-scale-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "scale_artifact"},
                score_components={
                    **cand.score_components,
                    "scale_drift": float(rng.uniform(0.25, 0.55)),
                },
                redundancy_tier=cand.redundancy_tier,
            ),
            "scale_artifact",
        ))

        # Type E: Transfer-fragile
        adversarial.append((
            SleepCandidate(
                candidate_id=f"adv-xferfrag-{i}-{cand.candidate_id[:8]}",
                representative_episode_id=cand.representative_episode_id,
                member_episode_ids=cand.member_episode_ids,
                evidence={**cand.evidence, "adversarial": "transfer_fragile"},
                score_components={
                    **cand.score_components,
                    "portability_fraction": float(rng.uniform(0.35, 0.7)),
                    "functional_transfer_gain": float(rng.uniform(-0.4, -0.1)),
                    "scale_drift": float(rng.uniform(0.15, 0.35)),
                },
                redundancy_tier="portable",
            ),
            "transfer_fragile",
        ))

    return adversarial


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def build_candidate_pool(
    episodes: list[TraceEpisode],
    *,
    min_pool_size: int = 200,
    rng: np.random.Generator | None = None,
) -> tuple[list[SleepCandidate], dict[str, str]]:
    """Build a large labeled candidate pool.

    Returns (all_candidates, family_labels) where family_labels maps
    candidate_id → family label string.
    """
    rng = rng or np.random.default_rng(42)
    artifact = compress_trace_bank(episodes)
    real_candidates = list(artifact.candidates)

    # Label real candidates
    family_labels: dict[str, str] = {}
    for cand in real_candidates:
        family_labels[cand.candidate_id] = label_candidate_family(cand, episodes)

    # Inject synthetic adversarial to reach min_pool_size
    deficit = max(0, min_pool_size - len(real_candidates))
    n_per_type = max(5, deficit // 6 + 1)
    adversarial = _inject_synthetic_adversarial(
        real_candidates, episodes, rng=rng, n_per_type=n_per_type,
    )

    all_candidates = list(real_candidates)
    for adv_cand, adv_label in adversarial:
        all_candidates.append(adv_cand)
        family_labels[adv_cand.candidate_id] = adv_label

    return all_candidates, family_labels
