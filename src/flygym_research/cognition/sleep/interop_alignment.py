from __future__ import annotations

from collections import defaultdict

from ..metrics import interoperability_score, translation_preserves_environment
from .trace_schema import TraceEpisode



def build_alignment_registry(
    episodes: list[TraceEpisode],
    *,
    min_alignment_score: float = 0.55,
) -> dict[str, object]:
    by_world: dict[str, dict[str, TraceEpisode]] = defaultdict(dict)
    for episode in episodes:
        by_world[episode.world_mode].setdefault(episode.controller_name, episode)

    approved: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for world_mode, episodes_by_controller in sorted(by_world.items()):
        controllers = sorted(episodes_by_controller)
        for idx, left_name in enumerate(controllers):
            for right_name in controllers[idx + 1 :]:
                left = episodes_by_controller[left_name]
                right = episodes_by_controller[right_name]
                interop = interoperability_score(left.transitions, right.transitions)
                translation = translation_preserves_environment(
                    left.transitions,
                    right.transitions,
                )
                record = {
                    "world_mode": world_mode,
                    "left_controller": left_name,
                    "right_controller": right_name,
                    "interoperability_score": interop["interoperability_score"],
                    "translation_r2": translation["environment_preservation_r2"],
                    "translation_validity": translation["translation_validity"],
                }
                if (
                    interop["interoperability_score"] >= min_alignment_score
                    and translation["translation_validity"]
                ):
                    approved.append(record)
                else:
                    rejected.append(record)

    return {
        "approved_translations": approved,
        "rejected_translations": rejected,
        "n_approved": len(approved),
        "n_rejected": len(rejected),
    }
