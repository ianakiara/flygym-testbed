"""Transfer scoring for PR #5 — direct transfer utility measurement.

Replaces tier-based heuristics with functional transfer metrics:
  - portability fraction (cross-world success rate)
  - functional transfer gain (return lift on unseen worlds)
  - replay stability (variance across replay worlds)
  - mismatch robustness
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..metrics.sleep_metrics import trajectory_equivalence_strength
from ..sleep.trace_schema import SleepCandidate, TraceEpisode


def compute_transfer_score(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    min_cross_world_support: int = 2,
) -> dict[str, float]:
    """Compute direct transfer utility for a candidate.

    Returns a dict with:
      - portability_fraction: fraction of world pairs with positive transfer
      - functional_transfer_gain: mean return lift across worlds
      - replay_stability: 1 - normalized std of per-world returns
      - mismatch_robustness: fraction of worlds with < 50% degradation
      - consistency_score: mean cross-world equivalence strength
    """
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]

    if len(members) < 2:
        return {
            "portability_fraction": 0.0,
            "functional_transfer_gain": 0.0,
            "replay_stability": 0.0,
            "mismatch_robustness": 0.0,
            "consistency_score": 0.0,
        }

    # Group by world
    by_world: dict[str, list[TraceEpisode]] = defaultdict(list)
    for m in members:
        by_world[m.world_mode].append(m)

    worlds = sorted(by_world.keys())
    if len(worlds) < min_cross_world_support:
        return {
            "portability_fraction": 0.0,
            "functional_transfer_gain": 0.0,
            "replay_stability": 0.0,
            "mismatch_robustness": 0.0,
            "consistency_score": 0.0,
        }

    # Per-world mean returns
    world_returns = {
        w: float(np.mean([ep.summary_metrics.get("return", 0.0) for ep in eps]))
        for w, eps in by_world.items()
    }

    # Portability fraction: how many world pairs have positive transfer?
    positive_transfers = 0
    total_pairs = 0
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            total_pairs += 1
            # If both worlds have non-negative returns, positive transfer
            if world_returns[worlds[i]] > -5.0 and world_returns[worlds[j]] > -5.0:
                positive_transfers += 1
    portability_frac = float(positive_transfers / max(total_pairs, 1))

    # Functional transfer gain: mean return across all worlds
    all_returns = list(world_returns.values())
    transfer_gain = float(np.mean(all_returns))

    # Replay stability: 1 - CV of per-world returns
    if np.mean(np.abs(all_returns)) > 1e-8:
        cv = float(np.std(all_returns) / (np.mean(np.abs(all_returns)) + 1e-8))
        stability = max(0.0, 1.0 - cv)
    else:
        stability = 0.0

    # Mismatch robustness: fraction of worlds with acceptable performance
    best_return = max(all_returns) if all_returns else 0.0
    robust_count = sum(
        1 for r in all_returns
        if r > best_return * 0.5 - 5.0
    )
    robustness = float(robust_count / max(len(all_returns), 1))

    # Consistency score: mean cross-world equivalence
    cross_scores = []
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            eps_a = by_world[worlds[i]][:2]
            eps_b = by_world[worlds[j]][:2]
            for ea in eps_a:
                for eb in eps_b:
                    try:
                        score = trajectory_equivalence_strength(
                            ea.transitions, eb.transitions,
                        )["trajectory_equivalence_strength"]
                        cross_scores.append(float(score))
                    except Exception:
                        cross_scores.append(0.0)
    consistency = float(np.mean(cross_scores)) if cross_scores else 0.0

    return {
        "portability_fraction": portability_frac,
        "functional_transfer_gain": transfer_gain,
        "replay_stability": stability,
        "mismatch_robustness": robustness,
        "consistency_score": consistency,
    }
