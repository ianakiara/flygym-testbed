"""POC: BackboneShared composite metric — validates Ω − λ·Interop − μ·Seam − ν·Drift.

Pass condition: BackboneShared correctly distinguishes same-world from cross-world.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from ..metrics.backbone_shared import (
    BackboneSharedConfig,
    compare_world_modes,
    compute_backbone_shared,
)
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Run the BackboneShared validation experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect episodes from different worlds ───────────────────────
    same_world_episodes = collect_trace_bank(
        seeds=[0, 1],
        world_modes=["avatar_remapped"],
        ablations=[frozenset()],
        perturbation_tags=["baseline"],
        max_steps=15,
    )
    cross_world_episodes = collect_trace_bank(
        seeds=[0, 1],
        world_modes=["simplified_embodied"],
        ablations=[frozenset()],
        perturbation_tags=["baseline"],
        max_steps=15,
    )

    # ── Build transition dicts ───────────────────────────────────────
    same_world_transitions = {
        ep.controller_name: ep.transitions for ep in same_world_episodes
    }
    cross_world_transitions = {
        ep.controller_name: ep.transitions for ep in cross_world_episodes
    }

    # ── Compute BackboneShared ───────────────────────────────────────
    same_result = compute_backbone_shared(same_world_transitions)
    cross_result = compute_backbone_shared(
        same_world_transitions,
        cross_world_transitions=cross_world_transitions,
    )

    # ── Compare world modes ──────────────────────────────────────────
    comparison = compare_world_modes(
        same_world_transitions, cross_world_transitions,
    )

    # ── Test different configs ───────────────────────────────────────
    strict_config = BackboneSharedConfig(gate_threshold=0.2)
    lenient_config = BackboneSharedConfig(gate_threshold=-0.5)
    strict_result = compute_backbone_shared(
        same_world_transitions,
        cross_world_transitions=cross_world_transitions,
        config=strict_config,
    )
    lenient_result = compute_backbone_shared(
        same_world_transitions,
        cross_world_transitions=cross_world_transitions,
        config=lenient_config,
    )

    # ── Pass condition ───────────────────────────────────────────────
    # Same-world should have higher BackboneShared than cross-world
    distinguishes = comparison["distinguishes_modes"]

    summary = {
        "same_world_backbone_shared": round(same_result["backbone_shared"], 4),
        "cross_world_backbone_shared": round(cross_result["backbone_shared"], 4),
        "same_world_components": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in same_result.items() if k != "components"
        },
        "cross_world_components": {
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in cross_result.items() if k != "components"
        },
        "comparison": {
            "delta": round(comparison["backbone_shared_delta"], 4),
            "same_gate": comparison["same_world_gate"],
            "cross_gate": comparison["cross_world_gate"],
            "distinguishes_modes": distinguishes,
        },
        "config_sensitivity": {
            "strict_gate": strict_result["gate_approved"],
            "lenient_gate": lenient_result["gate_approved"],
        },
        "pass_condition_met": distinguishes,
        "scale_drift_active": cross_result["scale_drift"] > 0.0,
    }

    (output_dir / "backbone_shared_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
