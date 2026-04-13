from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..metrics import (
    counterfactual_divergence,
    degeneracy_score,
    interoperability_score,
    seam_fragility,
    shared_structure_profile,
    transfer_hierarchy,
)
from ..metrics.sleep_metrics import compression_gain, residual_utility, seam_critical_exception_score
from .trace_schema import SleepCandidate, TraceEpisode


def _mean_metric(episodes: list[TraceEpisode], key: str) -> float:
    if not episodes:
        return 0.0
    values = [episode.summary_metrics.get(key, 0.0) for episode in episodes]
    return float(np.mean(values))


def _episodes_by_id(episodes: list[TraceEpisode]) -> dict[str, TraceEpisode]:
    return {episode.episode_id: episode for episode in episodes}


def _primary_divergence_metric(divergence: dict[str, object]) -> float:
    if "mean_translation_divergence" in divergence:
        value = divergence["mean_translation_divergence"]
    elif "mean_reward_divergence" in divergence:
        value = divergence["mean_reward_divergence"]
    else:
        value = 0.0
    return float(value) if isinstance(value, (int, float)) else 0.0


def _resolve_portability_fraction(
    candidate: SleepCandidate,
    members: list[TraceEpisode],
) -> float:
    raw_fraction = candidate.portability_evidence.get("portability_fraction")
    if raw_fraction is None:
        return float(len({episode.world_mode for episode in members}))
    portability_fraction = float(raw_fraction)
    total_worlds = len(candidate.portability_evidence.get("total_worlds", []))
    if portability_fraction > 1.0 and total_worlds > 0:
        return portability_fraction / total_worlds
    return portability_fraction


def backbone_shared_score(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    functional_transfer_gain: float | None = None,
) -> dict[str, float | str]:
    by_id = _episodes_by_id(episodes)
    members = [by_id[episode_id] for episode_id in candidate.member_episode_ids]
    representative = by_id[candidate.representative_episode_id]
    transitions_dict = {episode.episode_id: episode.transitions for episode in members}
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
            divergences.append(_primary_divergence_metric(divergence))
    scale_drift = float(np.mean(divergences)) if divergences else 0.0
    compression = compression_gain(len(members), 1)
    portability_fraction = _resolve_portability_fraction(candidate, members)
    mean_success = _mean_metric(members, "success")
    if functional_transfer_gain is None:
        functional_transfer_gain = candidate.functional_utility.get("functional_transfer_gain")
    if functional_transfer_gain is None:
        functional_transfer_gain = float(
            np.clip(
                portability_fraction * max(mean_success - interop_loss - 0.5 * seam_risk, 0.0),
                0.0,
                1.0,
            )
        )
    redundancy = float(
        np.clip(
            0.4 * q_red
            + 0.35 * compression["compression_gain"]
            + 0.25 * candidate.score_components.get("mean_equivalence_strength", 1.0),
            0.0,
            1.0,
        )
    )
    degeneracy_penalty = float(np.clip(q_red * (1.0 - mean_success), 0.0, 1.0))

    # Honor explicit score_components overrides for synthetic/adversarial
    # candidates whose transitions are shared with their source candidate.
    # Real candidates don't have these keys in score_components.
    if "seam_risk" in candidate.score_components:
        seam_risk = float(candidate.score_components["seam_risk"])
    if "interop_loss" in candidate.score_components:
        interop_loss = float(candidate.score_components["interop_loss"])
    if "scale_drift" in candidate.score_components:
        scale_drift = float(candidate.score_components["scale_drift"])
    if "degeneracy_penalty" in candidate.score_components:
        degeneracy_penalty = float(candidate.score_components["degeneracy_penalty"])

    profile = shared_structure_profile(
        redundancy=redundancy,
        portability_fraction=portability_fraction,
        functional_transfer_gain=functional_transfer_gain,
        seam_risk=seam_risk,
        interop_loss=interop_loss,
        scale_drift=scale_drift,
        degeneracy_penalty=degeneracy_penalty,
    )
    hierarchy = transfer_hierarchy(
        functional_transfer_gain=functional_transfer_gain,
        portability_fraction=portability_fraction,
    )
    return {
        "quotient_redundancy": q_red,
        "redundancy_score": redundancy,
        "seam_risk": seam_risk,
        "interop_loss": interop_loss,
        "scale_drift": scale_drift,
        "compression_gain": compression["compression_gain"],
        "degeneracy_penalty": degeneracy_penalty,
        "functional_transfer_gain": float(functional_transfer_gain),
        "portability_fraction": portability_fraction,
        "backbone_shared_score": float(profile["backbone_shared"]),
        "shared_structure_regime": str(profile["shared_structure_regime"]),
        "transfer_hierarchy_tier": str(hierarchy["transfer_hierarchy_tier"]),
        "mean_return": _mean_metric(members, "return"),
        "mean_success": mean_success,
    }


def safe_compression_score(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    alpha: float = 1.0,
    beta: float = 0.6,
    gamma: float = 0.6,
    delta: float = 0.5,
) -> dict[str, float | str]:
    result = backbone_shared_score(candidate, episodes)
    weighted_score = float(
        alpha * result["redundancy_score"]
        + 0.25 * alpha * result["functional_transfer_gain"]
        - beta * result["seam_risk"]
        - gamma * result["interop_loss"]
        - delta * result["scale_drift"]
        - 0.7 * result["degeneracy_penalty"]
    )
    result["safe_compression_score"] = weighted_score
    return result


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
