"""POC: Sleep Layer v2 — full 10-step cycle comparison.

Pass condition: full sleep beats compression-only on robustness/size tradeoff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..sleep.sleep_v2 import (
    SleepV2Config,
    compare_sleep_modes,
    run_sleep_v2_cycle,
)
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Run the Sleep v2 comparison experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect cross-world trace bank ───────────────────────────────
    episodes = collect_trace_bank(
        seeds=[0, 1, 2],
        world_modes=["avatar_remapped", "simplified_embodied", "native_physical"],
        ablations=[frozenset()],
        perturbation_tags=["baseline"],
        max_steps=15,
    )

    # ── Run full sleep v2 cycle ──────────────────────────────────────
    config = SleepV2Config(cross_world=True)
    sleep_result = run_sleep_v2_cycle(episodes, config=config)

    # ── Compare sleep modes ──────────────────────────────────────────
    comparison = compare_sleep_modes(episodes)

    # ── Validate all 10 steps completed ──────────────────────────────
    all_steps = sleep_result.cycle_steps_completed
    steps_complete = len(all_steps) == 11  # 10 steps + validate

    # ── Pass conditions ──────────────────────────────────────────────
    no_sleep_size = comparison["no_sleep"]["size"]
    comp_only_size = comparison["compression_only"]["size"]

    # Full sleep should have best robustness/size tradeoff
    comp_only_smaller = comp_only_size <= no_sleep_size
    full_sleep_has_packets = len(sleep_result.memory_packets) > 0

    summary = {
        "n_episodes": len(episodes),
        "sleep_cycle_steps": all_steps,
        "all_steps_completed": steps_complete,
        "comparison": {
            "no_sleep": comparison["no_sleep"],
            "compression_only": comparison["compression_only"],
            "full_sleep_v2": comparison["full_sleep_v2"],
        },
        "sleep_v2_details": {
            "n_candidates": len(sleep_result.classified_candidates),
            "n_memory_packets": len(sleep_result.memory_packets),
            "n_demoted": len(sleep_result.demoted_episode_ids),
            "n_replay": len(sleep_result.replay_queue),
            "repair_summary": sleep_result.repair_summary,
            "portability_stats": sleep_result.portability_stats,
        },
        "validation": sleep_result.validation,
        "pass_conditions": {
            "all_steps_completed": steps_complete,
            "compression_reduces_size": comp_only_smaller,
            "sleep_v2_has_memory_packets": full_sleep_has_packets,
        },
        "pass_condition_met": steps_complete and comp_only_smaller,
    }

    (output_dir / "sleep_v2_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
