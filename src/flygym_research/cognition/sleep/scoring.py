from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..metrics import (
    counterfactual_divergence,
    degeneracy_score,
    interoperability_score,
    seam_fragility,
)
from ..metrics.sleep_metrics import (
    compression_gain,
    residual_utility,
    seam_critical_exception_score,
)
from .trace_schema import SleepCandidate, TraceEpisode



def _mean_metric(episodes: list[TraceEpisode], key: str) -> float:
    if not episodes:
        return 0.0
    values = [episode.summary_metrics.get(key, 0.0) for episode in episodes]
    return float(np.mean(values))



def _episodes_by_id(episodes: list[TraceEpisode]) -> dict[str, TraceEpisode]:
    return {episode.episode_id: episode for episode in episodes}



def safe_compression_score(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    alpha: float = 1.0,
    beta: float = 0.6,
    gamma: float = 0.6,
    delta: float = 0.5,
) -> dict[str, float]:
    by_id = _episodes_by_id(episodes)
    members = [by_id[episode_id] for episode_id in candidate.member_episode_ids]
    representative = by_id[candidate.representative_episode_id]

    transitions_dict = {
        episode.episode_id: episode.transitions
        for episode in members
    }
    degeneracy = degeneracy_score(transitions_dict)
    q_red = float(
        np.clip(
            0.5 * degeneracy.get("degeneracy_ratio", 0.0)
            + 0.5 * degeneracy.get("information_loss", 0.0),
            0.0,
            1.0,
        )
    )

    seam_scores = [
        seam_critical_exception_score(
            episode.transitions,
            representative.transitions,
        )["seam_critical_exception_score"]
        for episode in members
    ]
    seam_risk = float(np.mean(seam_scores)) if seam_scores else 0.0

    interop_losses = []
    for episode in members:
        if episode.episode_id == representative.episode_id:
            continue
        interop = interoperability_score(representative.transitions, episode.transitions)
        interop_losses.append(1.0 - interop["interoperability_score"])
    interop_loss = float(np.mean(interop_losses)) if interop_losses else 0.0

    by_world: dict[str, dict[str, list]] = defaultdict(dict)
    for episode in members:
        by_world[episode.world_mode][episode.controller_name] = episode.transitions
    divergences = []
    worlds = sorted(by_world)
    for idx, world_a in enumerate(worlds):
        for world_b in worlds[idx + 1 :]:
            common = set(by_world[world_a]) & set(by_world[world_b])
            if len(common) < 2:
                continue
            divergence = counterfactual_divergence(
                {name: by_world[world_a][name] for name in common},
                {name: by_world[world_b][name] for name in common},
            )
            divergences.append(divergence["mean_divergence"])
    scale_drift = float(np.mean(divergences)) if divergences else 0.0

    compression = compression_gain(len(members), 1)
    safe_score = alpha * q_red - beta * seam_risk - gamma * interop_loss - delta * scale_drift
    return {
        "quotient_redundancy": q_red,
        "seam_risk": seam_risk,
        "interop_loss": interop_loss,
        "scale_drift": scale_drift,
        "compression_gain": compression["compression_gain"],
        "safe_compression_score": float(safe_score),
        "mean_return": _mean_metric(members, "return"),
        "mean_success": _mean_metric(members, "success"),
    }



def residual_score(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> dict[str, float]:
    by_id = _episodes_by_id(episodes)
    members = [by_id[episode_id] for episode_id in candidate.member_episode_ids]
    seam_exception_count = sum(
        1
        for episode in members
        if seam_fragility(episode.transitions)["seam_fragility"] > 0.2
    )
    return residual_utility(
        len(members),
        len(candidate.residual_episode_ids),
        seam_exception_count,
    )
