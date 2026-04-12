from __future__ import annotations

import numpy as np

from ..interfaces import DescendingCommand
from ..metrics import seam_fragility, translation_preserves_environment
from ..metrics.sleep_metrics import repairability_score
from .trace_schema import TraceEpisode



def analyze_seam_failures(
    episodes: list[TraceEpisode],
    *,
    seam_threshold: float = 0.2,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    patchable = 0
    by_world: dict[str, list[TraceEpisode]] = {}
    for episode in episodes:
        by_world.setdefault(episode.world_mode, []).append(episode)
        seam = seam_fragility(episode.transitions)["seam_fragility"]
        target_bias = [
            np.asarray(t.action.target_bias, dtype=np.float64)
            for t in episode.transitions
            if isinstance(t.action, DescendingCommand)
        ]
        world_target = [
            np.asarray(
                t.observation.world.observables.get("target_vector", np.zeros(2)),
                dtype=np.float64,
            )
            for t in episode.transitions
        ]
        mismatch = 0.0
        if target_bias and world_target:
            n = min(len(target_bias), len(world_target))
            mismatch = float(
                np.mean(
                    [
                        np.linalg.norm(target_bias[idx] - world_target[idx])
                        for idx in range(n)
                    ]
                )
            )
        if seam > seam_threshold or mismatch > 0.35:
            if seam <= seam_threshold * 1.5:
                patchable += 1
            failures.append(
                {
                    "episode_id": episode.episode_id,
                    "controller_name": episode.controller_name,
                    "world_mode": episode.world_mode,
                    "seam_fragility": seam,
                    "target_bias_mismatch": mismatch,
                    "recommended_action": (
                        "attach residual exception"
                        if seam > seam_threshold * 1.5
                        else "retune adapter gains"
                    ),
                }
            )

    cross_mode: list[dict[str, object]] = []
    worlds = sorted(by_world)
    for idx, world_a in enumerate(worlds):
        for world_b in worlds[idx + 1 :]:
            left = {episode.controller_name: episode for episode in by_world[world_a]}
            right = {episode.controller_name: episode for episode in by_world[world_b]}
            for name in sorted(set(left) & set(right)):
                translation = translation_preserves_environment(
                    left[name].transitions,
                    right[name].transitions,
                )
                if not translation["translation_validity"]:
                    cross_mode.append(
                        {
                            "controller_name": name,
                            "world_pair": f"{world_a}->{world_b}",
                            "environment_preservation_r2": translation[
                                "environment_preservation_r2"
                            ],
                            "recommended_action": "flag translation as unstable",
                        }
                    )

    report = {
        "failures": failures,
        "cross_mode_failures": cross_mode,
        "n_failures": len(failures) + len(cross_mode),
        "n_patchable_failures": patchable + len(cross_mode),
    }
    report.update(repairability_score(report))
    return report
