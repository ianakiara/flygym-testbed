from __future__ import annotations

from ..metrics import interoperability_score, seam_fragility
from .trace_schema import SleepCandidate, TraceEpisode



def select_residual_exceptions(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    seam_threshold: float = 0.2,
    interop_threshold: float = 0.45,
) -> SleepCandidate:
    by_id = {episode.episode_id: episode for episode in episodes}
    representative = by_id[candidate.representative_episode_id]
    residual_ids: list[str] = []
    rationale: dict[str, str] = {}

    for episode_id in candidate.member_episode_ids:
        if episode_id == representative.episode_id:
            continue
        episode = by_id[episode_id]
        seam = seam_fragility(episode.transitions)["seam_fragility"]
        interop = interoperability_score(
            representative.transitions,
            episode.transitions,
        )["interoperability_score"]
        if seam > seam_threshold:
            residual_ids.append(episode_id)
            rationale[episode_id] = f"seam_fragility={seam:.3f}"
        elif interop < interop_threshold:
            residual_ids.append(episode_id)
            rationale[episode_id] = f"interoperability={interop:.3f}"

    candidate.residual_episode_ids = sorted(set(residual_ids))
    candidate.retained_exception_rationale = rationale
    return candidate
