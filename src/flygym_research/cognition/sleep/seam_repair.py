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
    mismatch_threshold: float = 0.35,
) -> dict[str, object]:
    """Detect seam failures using *both* seam fragility and target-bias
    mismatch thresholds.

    Parameters
    ----------
    seam_threshold : float
        Episodes with ``seam_fragility > seam_threshold`` are failures.
    mismatch_threshold : float
        Episodes with ``target_bias_mismatch > mismatch_threshold`` are
        failures.  Previously hard-coded at 0.35; now parameterised so
        the repairability curve can sweep this dimension too.
    """
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
        seam_fail = seam > seam_threshold
        mismatch_fail = mismatch > mismatch_threshold
        if seam_fail or mismatch_fail:
            # Patchability is now a *joint* condition: both seam and
            # mismatch must be within their respective repair margins.
            seam_patchable = seam <= seam_threshold * 1.5
            mismatch_patchable = mismatch <= mismatch_threshold * 1.5
            if seam_patchable and mismatch_patchable:
                patchable += 1
            failures.append(
                {
                    "episode_id": episode.episode_id,
                    "controller_name": episode.controller_name,
                    "world_mode": episode.world_mode,
                    "seam_fragility": seam,
                    "target_bias_mismatch": mismatch,
                    "seam_patchable": seam_patchable,
                    "mismatch_patchable": mismatch_patchable,
                    "recommended_action": (
                        "attach residual exception"
                        if not seam_patchable
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
    seam_thresholds: list[float] | None = None,
    mismatch_thresholds: list[float] | None = None,
) -> dict[str, object]:
    """Sweep seam *and* mismatch thresholds to produce a 2-D repairability
    surface.

    The previous 1-D sweep (seam_threshold only) was flat because the
    dominant failure source was ``target_bias_mismatch > 0.35`` — tightening
    the seam knob alone had no effect.  This 2-D version sweeps both
    dimensions so the curve actually varies.

    Returns a dict with:
    - ``curve``: list of ``{seam_threshold, mismatch_threshold, n_failures,
      n_patchable, repairability}``
    - ``critical_point``: first point where repairability < 0.9
    - ``default_repairability``: repairability at (0.20, 0.35)
    - ``tight_repairability``: repairability at the tightest point
    - ``seam_only_curve``: 1-D slice holding mismatch at default 0.35
    - ``mismatch_only_curve``: 1-D slice holding seam at default 0.20
    """
    if seam_thresholds is None:
        seam_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    if mismatch_thresholds is None:
        mismatch_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

    curve: list[dict[str, float]] = []
    for st in seam_thresholds:
        for mt in mismatch_thresholds:
            report = analyze_seam_failures(
                episodes, seam_threshold=st, mismatch_threshold=mt,
            )
            n_failures = int(report["n_failures"])  # type: ignore[arg-type]
            n_patchable = int(report["n_patchable_failures"])  # type: ignore[arg-type]
            repair = float(report.get("repairability_score", 0.0))  # type: ignore[arg-type]
            curve.append({
                "seam_threshold": st,
                "mismatch_threshold": mt,
                "n_failures": float(n_failures),
                "n_patchable": float(n_patchable),
                "repairability": repair,
            })

    # Find the critical point: first (tightest) where repairability < 0.9.
    critical_point: dict[str, float] | None = None
    for point in curve:
        if point["repairability"] < 0.9:
            critical_point = point
            break

    default_repair = next(
        (
            p["repairability"]
            for p in curve
            if abs(p["seam_threshold"] - 0.20) < 1e-6
            and abs(p["mismatch_threshold"] - 0.35) < 1e-6
        ),
        None,
    )
    tight_repair = curve[0]["repairability"] if curve else None

    # Extract 1-D slices for backward compatibility and easier plotting.
    seam_only_curve = [
        p for p in curve if abs(p["mismatch_threshold"] - 0.35) < 1e-6
    ]
    mismatch_only_curve = [
        p for p in curve if abs(p["seam_threshold"] - 0.20) < 1e-6
    ]

    return {
        "curve": curve,
        "critical_point": critical_point,
        "default_repairability": default_repair,
        "tight_repairability": tight_repair,
        "n_points_tested": len(curve),
        "seam_only_curve": seam_only_curve,
        "mismatch_only_curve": mismatch_only_curve,
    }
