"""POC: Repair v2 — mismatch-aware repair vs uniform vs seam-only.

Pass condition: mismatch-aware repair outperforms uniform repair.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from ..sleep.repair_policies import (
    AdaptivePatchability,
    FailureType,
    compare_repair_strategies,
    diagnose_failure,
)
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Run the repair strategy comparison experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect episodes ─────────────────────────────────────────────
    episodes = collect_trace_bank(
        seeds=[0, 1, 2],
        world_modes=["avatar_remapped", "simplified_embodied"],
        ablations=[frozenset()],
        perturbation_tags=["baseline"],
        max_steps=15,
    )

    # ── Diagnose all episodes ────────────────────────────────────────
    diagnoses = []
    episodes_with_transitions = []
    for ep in episodes:
        diag = diagnose_failure(ep.episode_id, ep.transitions)
        diagnoses.append(diag)
        episodes_with_transitions.append((ep.episode_id, ep.transitions))

    # ── Failure type distribution ────────────────────────────────────
    type_counts = {}
    for d in diagnoses:
        t = d.failure_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    # ── Compare all strategies ───────────────────────────────────────
    comparison = compare_repair_strategies(episodes_with_transitions)

    # ── Adaptive patchability ────────────────────────────────────────
    failure_diagnoses = [d for d in diagnoses if d.failure_type != FailureType.NO_FAILURE]
    if failure_diagnoses:
        adaptive = AdaptivePatchability.from_distribution(failure_diagnoses)
        n_adaptive_patchable = sum(
            1 for d in failure_diagnoses if adaptive.is_patchable(d)
        )
        adaptive_info = {
            "seam_margin": round(adaptive.seam_margin, 3),
            "mismatch_margin": round(adaptive.mismatch_margin, 3),
            "budget": adaptive.budget,
            "n_patchable": n_adaptive_patchable,
            "patchable_rate": round(n_adaptive_patchable / len(failure_diagnoses), 3),
        }
    else:
        adaptive_info = {"note": "no failures to repair"}

    # ── Pass condition ───────────────────────────────────────────────
    mismatch_aware_rate = comparison.get("mismatch_aware", {}).get("success_rate", 0.0)
    uniform_rate = comparison.get("uniform", {}).get("success_rate", 0.0)
    mismatch_aware_wins = mismatch_aware_rate >= uniform_rate

    summary = {
        "n_episodes": len(episodes),
        "n_diagnoses": len(diagnoses),
        "failure_type_distribution": type_counts,
        "strategy_comparison": comparison,
        "adaptive_patchability": adaptive_info,
        "pass_condition_met": mismatch_aware_wins,
        "mismatch_aware_success_rate": round(mismatch_aware_rate, 3),
        "uniform_success_rate": round(uniform_rate, 3),
    }

    (output_dir / "repair_v2_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
