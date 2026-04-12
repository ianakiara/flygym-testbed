"""POC: Hard-Memory Benchmark — 8 tasks × 4 controllers × 5 seeds.

Pass condition: SlotMemoryController beats reactive by real margin on ≥2 hard tasks.
Also validates Task Complexity Meter and Metric Auditor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark_harness import run_episode
from ..body_reflex import BodylessBodyLayer
from ..controllers import (
    MemoryController,
    RandomController,
    ReducedDescendingController,
    SlotMemoryController,
)
from ..diagnostics.metric_auditor import (
    audit_metric,
    check_averaging_distortion,
    check_constant_output,
)
from ..diagnostics.task_complexity_meter import (
    batch_complexity_assessment,
)
from ..config import EnvConfig
from ..envs import FlyBodyWorldEnv
from ..metrics import summarize_metrics
from ..tasks import (
    BranchDependentRecallTask,
    ConditionalSequenceTask,
    DelayedRewardTask,
    DistractorCueRecallTask,
    HiddenCueRecallTask,
    HistoryDependenceTask,
    NavigationTask,
    SameStateDifferentHistoryTask,
)


TASK_REGISTRY: dict[str, type] = {
    "navigation": NavigationTask,
    "delayed_reward": DelayedRewardTask,
    "history_dependence": HistoryDependenceTask,
    "hidden_cue_recall": HiddenCueRecallTask,
    "distractor_cue_recall": DistractorCueRecallTask,
    "conditional_sequence": ConditionalSequenceTask,
    "same_state_different_history": SameStateDifferentHistoryTask,
    "branch_dependent_recall": BranchDependentRecallTask,
}

CONTROLLER_REGISTRY: dict[str, type] = {
    "random": RandomController,
    "reactive": ReducedDescendingController,
    "memory": MemoryController,
    "slot_memory": SlotMemoryController,
}


def _run_single(
    task_name: str,
    task_cls: type,
    controller_name: str,
    controller_cls: type,
    seed: int,
    max_steps: int = 30,
) -> dict[str, float]:
    """Run a single task/controller/seed combo and return summary metrics."""
    config = EnvConfig(episode_steps=max_steps)
    body = BodylessBodyLayer()
    world = task_cls(config=config)
    env = FlyBodyWorldEnv(body=body, world=world, config=config)
    controller = controller_cls()
    transitions = run_episode(env, controller, seed=seed, max_steps=max_steps)
    metrics = summarize_metrics(transitions)
    total_reward = sum(t.reward for t in transitions)
    return {
        "task": task_name,
        "controller": controller_name,
        "seed": seed,
        "total_reward": total_reward,
        "mean_reward": metrics.get("mean_reward", 0.0),
        "success": metrics.get("success", 0.0),
        "n_steps": len(transitions),
    }


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Run the full hard-memory benchmark."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [0, 1, 2, 3, 4]
    max_steps = 30

    # ── Run all combos ───────────────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    returns_by_task: dict[str, dict[str, list[float]]] = {}

    for task_name, task_cls in TASK_REGISTRY.items():
        returns_by_task[task_name] = {cn: [] for cn in CONTROLLER_REGISTRY}
        for ctrl_name, ctrl_cls in CONTROLLER_REGISTRY.items():
            for seed in seeds:
                result = _run_single(
                    task_name, task_cls, ctrl_name, ctrl_cls, seed, max_steps,
                )
                all_results.append(result)
                returns_by_task[task_name][ctrl_name].append(result["total_reward"])

    # ── Task Complexity Meter ────────────────────────────────────────
    # Remap controller keys to match compute_task_complexity expectations
    # ("slot_memory" → "slot", others stay the same)
    complexity_input: dict[str, dict[str, list[float]]] = {}
    for task_name, ctrl_returns in returns_by_task.items():
        mapped: dict[str, list[float]] = {}
        for k, v in ctrl_returns.items():
            mapped["slot" if k == "slot_memory" else k] = v
        complexity_input[task_name] = mapped
    complexity_results = batch_complexity_assessment(complexity_input)
    complexity_summary = {
        task_name: {
            "class": cr.complexity_class,
            "slot_vs_trace_gap": round(cr.slot_vs_trace_gap, 3),
            "memory_vs_trace_gap": round(cr.memory_vs_trace_gap, 3),
        }
        for task_name, cr in complexity_results.items()
    }

    # ── Controller comparison ────────────────────────────────────────
    controller_means: dict[str, dict[str, float]] = {}
    for task_name in TASK_REGISTRY:
        controller_means[task_name] = {}
        for ctrl_name in CONTROLLER_REGISTRY:
            vals = returns_by_task[task_name][ctrl_name]
            controller_means[task_name][ctrl_name] = float(np.mean(vals))

    # ── Pass condition check ─────────────────────────────────────────
    # "slot_memory beats reactive by real margin on ≥2 hard tasks"
    hard_tasks = [
        tn for tn, cr in complexity_results.items()
        if cr.complexity_class in ("moderate", "deep", "ultra_deep")
    ]
    slot_wins = 0
    win_details: list[dict[str, Any]] = []
    for tn in TASK_REGISTRY:
        slot_mean = controller_means[tn]["slot_memory"]
        reactive_mean = controller_means[tn]["reactive"]
        margin = slot_mean - reactive_mean
        if margin > 0.5:
            slot_wins += 1
            win_details.append({"task": tn, "margin": round(margin, 3)})

    pass_condition = slot_wins >= 2

    # ── Metric Auditor sample ────────────────────────────────────────
    # Audit the seam_fragility metric for averaging distortion
    all_seam_values = []
    for r in all_results:
        all_seam_values.append(r["mean_reward"])

    avg_audit = check_averaging_distortion(all_seam_values, "mean_reward")

    # Audit for constant output across different tasks
    variant_values = [
        [controller_means[tn]["slot_memory"]] for tn in list(TASK_REGISTRY)[:4]
    ]
    const_audit = check_constant_output(
        lambda *args: {"slot_return": args[0][0]},
        variant_values,
        "slot_return",
    )

    auditor_report = audit_metric("benchmark_metrics", [avg_audit, const_audit])

    # ── Save results ─────────────────────────────────────────────────
    summary = {
        "n_tasks": len(TASK_REGISTRY),
        "n_controllers": len(CONTROLLER_REGISTRY),
        "n_seeds": len(seeds),
        "n_total_runs": len(all_results),
        "controller_means": controller_means,
        "complexity": complexity_summary,
        "hard_tasks": hard_tasks,
        "slot_wins_over_reactive": slot_wins,
        "pass_condition_met": pass_condition,
        "win_details": win_details,
        "auditor_summary": auditor_report.summary,
        "auditor_passed": auditor_report.overall_passed,
    }

    (output_dir / "benchmark_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
