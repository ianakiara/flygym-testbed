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


def repairability_curve(
    episodes: list[TraceEpisode],
    *,
    thresholds: list[float] | None = None,
) -> dict[str, object]:
    """Sweep seam thresholds to produce a repairability curve.

    Instead of a single pass/fail at one threshold, this function evaluates
    repairability across a range of thresholds.  Tighter (lower) thresholds
    expose more failures and reveal the true difficulty of the repair task.

    Returns a dict with:
    - ``curve``: list of ``{threshold, n_failures, n_patchable, repairability}``
    - ``critical_threshold``: lowest threshold where repairability drops below 0.9
    - ``default_threshold_repairability``: repairability at threshold=0.20
    - ``tight_threshold_repairability``: repairability at the tightest threshold
    """
    if thresholds is None:
        thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]

    curve: list[dict[str, float]] = []
    for threshold in thresholds:
        report = analyze_seam_failures(episodes, seam_threshold=threshold)
        n_failures = int(report["n_failures"])  # type: ignore[arg-type]
        n_patchable = int(report["n_patchable_failures"])  # type: ignore[arg-type]
        repair = float(report.get("repairability_score", 0.0))  # type: ignore[arg-type]
        curve.append({
            "seam_threshold": threshold,
            "n_failures": float(n_failures),
            "n_patchable": float(n_patchable),
            "repairability": repair,
        })

    # Find the critical threshold: tightest where repairability < 0.9.
    critical_threshold: float | None = None
    for point in curve:
        if point["repairability"] < 0.9:
            critical_threshold = point["seam_threshold"]
            break

    default_repair = next(
        (p["repairability"] for p in curve if abs(p["seam_threshold"] - 0.20) < 1e-6),
        None,
    )
    tight_repair = curve[0]["repairability"] if curve else None

    return {
        "curve": curve,
        "critical_threshold": critical_threshold,
        "default_threshold_repairability": default_repair,
        "tight_threshold_repairability": tight_repair,
        "n_thresholds_tested": len(thresholds),
    }
