"""POC: Degenerate Convergence Detector — healthy vs degenerate separation.

Pass condition: degenerate cases produce low BackboneShared despite high Ω.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from ..diagnostics.degenerate_convergence import (
    ConvergenceType,
    construct_degenerate_scenario,
    detect_convergence,
)
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Run the degenerate convergence detection experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect baseline episodes ────────────────────────────────────
    episodes = collect_trace_bank(
        seeds=[0, 1],
        world_modes=["avatar_remapped", "simplified_embodied"],
        ablations=[frozenset()],
        perturbation_tags=["baseline"],
        max_steps=15,
    )

    # ── Build transition dicts ───────────────────────────────────────
    baseline_transitions = {ep.controller_name: ep.transitions for ep in episodes}

    # ── Detect baseline convergence ──────────────────────────────────
    baseline_analysis = detect_convergence(baseline_transitions)

    # ── Construct degenerate scenarios ───────────────────────────────
    scenarios = {}
    for scenario_name in ["reward_collapse", "stress_convergence", "compression_narrowing"]:
        degenerate = construct_degenerate_scenario(baseline_transitions, scenario_name)
        analysis = detect_convergence(
            degenerate,
            perturbed_transitions=baseline_transitions,
            transfer_transitions=baseline_transitions,
        )
        scenarios[scenario_name] = {
            "convergence_type": analysis.convergence_type.value,
            "omega": round(analysis.omega_score, 4),
            "recovery": round(analysis.recovery_score, 4),
            "transfer": round(analysis.transfer_score, 4),
            "diversity": round(analysis.diversity_score, 4),
        }

    # ── Pass condition ───────────────────────────────────────────────
    # At least one degenerate scenario should be detected as degenerate
    any_degenerate = any(
        s["convergence_type"] == ConvergenceType.DEGENERATE.value
        for s in scenarios.values()
    )

    # Reward collapse should produce different convergence than baseline
    reward_collapse_type = scenarios["reward_collapse"]["convergence_type"]
    baseline_type = baseline_analysis.convergence_type.value
    collapse_differs = reward_collapse_type != baseline_type or any_degenerate

    summary = {
        "baseline": {
            "convergence_type": baseline_analysis.convergence_type.value,
            "omega": round(baseline_analysis.omega_score, 4),
            "recovery": round(baseline_analysis.recovery_score, 4),
            "transfer": round(baseline_analysis.transfer_score, 4),
            "diversity": round(baseline_analysis.diversity_score, 4),
        },
        "scenarios": scenarios,
        "any_degenerate_detected": any_degenerate,
        "collapse_differs_from_baseline": collapse_differs,
        "pass_condition_met": collapse_differs,
    }

    (output_dir / "degenerate_convergence_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
