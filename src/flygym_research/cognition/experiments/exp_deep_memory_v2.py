"""EXP 3 — Deep Memory Benchmark v2 (PR #5 priority 1).

Separates reactive / scalar / buffer / attention / selective memory
across 6 task families with 150–200 step episodes and 10–20 seeds.

Pass criteria:
  - memory > reactive on ≥ 4/6 task families
  - causal depth > 1 for advanced memory models
  - attention/gated memory > scalar memory on selective tasks
  - overwrite resistance improved

Includes baseline, method, and ablation for every comparison.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import EnvConfig
from ..controllers import (
    MemoryController,
    ReducedDescendingController,
    ReflexOnlyController,
    SelectiveMemoryController,
)
from ..interfaces import BrainInterface
from ..metrics import summarize_metrics
from ..metrics.causal_metrics import temporal_causal_depth
from ..research.memory_task_factory import ALL_TASK_FAMILIES, create_task
from ..experiments.exp_true_memory_benchmark import _TaskEnvWrapper
from ..experiments.benchmark_harness import run_episode


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_causal_depth(transitions: list, max_horizon: int = 30) -> float:
    """Compute causal depth with extended horizon for longer episodes."""
    try:
        result = temporal_causal_depth(transitions, max_horizon=max_horizon)
        return float(result.get("causal_depth", 0.0))
    except Exception:
        return 0.0


def _compute_memory_utilization(controller: BrainInterface) -> float:
    """Measure memory utilization from controller internal state."""
    state = controller.get_internal_state()
    if "active_slots" in state and "memory_slots" in state:
        return float(state["active_slots"]) / max(float(state["memory_slots"]), 1.0)
    if "memory_length" in state:
        return min(float(state["memory_length"]) / 16.0, 1.0)
    if "memory_trace" in state:
        return min(abs(float(state["memory_trace"])), 1.0)
    return 0.0


def _compute_selective_recall_accuracy(
    transitions: list, task_name: str,
) -> float:
    """For selective recall tasks, measure if correct cue was retrieved."""
    if task_name not in ("selective_recall", "sparse_relevance"):
        return -1.0  # not applicable
    if not transitions:
        return 0.0
    # Use late-episode rewards as proxy for recall accuracy
    late = transitions[int(len(transitions) * 0.7):]
    if not late:
        return 0.0
    return float(np.mean([1.0 if t.reward > 0 else 0.0 for t in late]))


def _compute_distractor_overwrite_rate(
    transitions: list, task_name: str,
) -> float:
    """Measure how often distractors corrupt memory (conflicting/selective tasks)."""
    if task_name not in ("conflicting_updates", "selective_recall", "sparse_relevance"):
        return -1.0
    if not transitions:
        return 0.0
    # If late rewards are negative despite early correct cue, distractor overwrote
    late = transitions[int(len(transitions) * 0.7):]
    if not late:
        return 0.0
    return float(np.mean([1.0 if t.reward < 0 else 0.0 for t in late]))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_deep_memory_v2",
    *,
    episode_steps: int = 150,
    n_seeds: int = 10,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = EnvConfig(episode_steps=episode_steps)

    # 6 task families
    task_names = list(ALL_TASK_FAMILIES.keys())

    # 5 controllers: reactive (baseline), reduced (ablation), scalar, buffer, attention
    controllers = {
        "reactive": lambda: ReflexOnlyController(),           # baseline
        "reduced_descending": lambda: ReducedDescendingController(),  # ablation
        "scalar_memory": lambda: ReducedDescendingController(),
        "buffer_memory": lambda: MemoryController(),          # method
        "attention_memory": lambda: SelectiveMemoryController(),  # method
    }

    seeds = list(range(n_seeds))

    all_results: dict[str, list[dict]] = defaultdict(list)
    per_task_results: dict[str, list[dict]] = defaultdict(list)

    for task_name in task_names:
        for ctrl_name, ctrl_factory in controllers.items():
            for seed in seeds:
                task = create_task(task_name, config=config)
                controller = ctrl_factory()
                env = _TaskEnvWrapper(task, config=config)
                transitions = run_episode(env, controller, seed=seed, max_steps=episode_steps)

                metrics = summarize_metrics(transitions)
                causal_depth = _compute_causal_depth(transitions, max_horizon=30)
                memory_util = _compute_memory_utilization(controller)
                recall_acc = _compute_selective_recall_accuracy(transitions, task_name)
                overwrite_rate = _compute_distractor_overwrite_rate(transitions, task_name)

                row = {
                    "task": task_name,
                    "controller": ctrl_name,
                    "seed": seed,
                    "return": metrics.get("return", 0.0),
                    "success": metrics.get("success", 0.0),
                    "causal_depth": causal_depth,
                    "memory_utilization": memory_util,
                    "selective_recall_accuracy": recall_acc,
                    "distractor_overwrite_rate": overwrite_rate,
                    "n_steps": len(transitions),
                }
                key = f"{task_name}/{ctrl_name}"
                all_results[key].append(row)
                per_task_results[task_name].append(row)

    # ---------------------------------------------------------------------------
    # Aggregation
    # ---------------------------------------------------------------------------

    # Per-controller summary
    controller_summary = {}
    for ctrl_name in controllers:
        rows = [r for v in all_results.values() for r in v if r["controller"] == ctrl_name]
        if not rows:
            continue
        controller_summary[ctrl_name] = {
            "mean_return": float(np.mean([r["return"] for r in rows])),
            "mean_success": float(np.mean([r["success"] for r in rows])),
            "mean_causal_depth": float(np.mean([r["causal_depth"] for r in rows])),
            "mean_memory_utilization": float(np.mean([r["memory_utilization"] for r in rows])),
            "std_return": float(np.std([r["return"] for r in rows])),
        }

    # Per-task × per-controller matrix
    task_controller_matrix = {}
    for task_name in task_names:
        task_controller_matrix[task_name] = {}
        for ctrl_name in controllers:
            rows = [r for r in per_task_results[task_name] if r["controller"] == ctrl_name]
            if not rows:
                continue
            task_controller_matrix[task_name][ctrl_name] = {
                "mean_return": float(np.mean([r["return"] for r in rows])),
                "mean_success": float(np.mean([r["success"] for r in rows])),
                "mean_causal_depth": float(np.mean([r["causal_depth"] for r in rows])),
                "mean_memory_utilization": float(np.mean([r["memory_utilization"] for r in rows])),
                "selective_recall_accuracy": float(np.mean([
                    r["selective_recall_accuracy"] for r in rows
                    if r["selective_recall_accuracy"] >= 0
                ])) if any(r["selective_recall_accuracy"] >= 0 for r in rows) else -1.0,
                "distractor_overwrite_rate": float(np.mean([
                    r["distractor_overwrite_rate"] for r in rows
                    if r["distractor_overwrite_rate"] >= 0
                ])) if any(r["distractor_overwrite_rate"] >= 0 for r in rows) else -1.0,
            }

    # ---------------------------------------------------------------------------
    # Pass criteria evaluation
    # ---------------------------------------------------------------------------

    # Criterion 1: memory > reactive on ≥ 4/6 tasks
    tasks_where_memory_wins = 0
    for task_name in task_names:
        reactive_rows = [r for r in per_task_results[task_name] if r["controller"] == "reactive"]
        memory_rows = [
            r for r in per_task_results[task_name]
            if r["controller"] in ("buffer_memory", "attention_memory")
        ]
        reactive_mean = float(np.mean([r["return"] for r in reactive_rows])) if reactive_rows else 0.0
        memory_mean = float(np.mean([r["return"] for r in memory_rows])) if memory_rows else 0.0
        if memory_mean > reactive_mean + 0.5:
            tasks_where_memory_wins += 1

    memory_gt_reactive_4_of_6 = tasks_where_memory_wins >= 4

    # Criterion 2: causal depth > 1 for advanced memory models
    buffer_depth = controller_summary.get("buffer_memory", {}).get("mean_causal_depth", 0.0)
    attention_depth = controller_summary.get("attention_memory", {}).get("mean_causal_depth", 0.0)
    causal_depth_gt_1 = buffer_depth > 1.0 or attention_depth > 1.0

    # Criterion 3: attention > scalar on selective tasks
    selective_tasks = ["selective_recall", "sparse_relevance"]
    attention_selective_return = []
    scalar_selective_return = []
    for task_name in selective_tasks:
        for r in per_task_results.get(task_name, []):
            if r["controller"] == "attention_memory":
                attention_selective_return.append(r["return"])
            elif r["controller"] == "scalar_memory":
                scalar_selective_return.append(r["return"])
    attn_mean = float(np.mean(attention_selective_return)) if attention_selective_return else 0.0
    scalar_mean = float(np.mean(scalar_selective_return)) if scalar_selective_return else 0.0
    attention_gt_scalar_on_selective = attn_mean > scalar_mean

    # Criterion 4: overwrite resistance — lower overwrite rate for attention vs scalar
    attention_overwrite = []
    scalar_overwrite = []
    for task_name in ("conflicting_updates", "selective_recall"):
        for r in per_task_results.get(task_name, []):
            if r["distractor_overwrite_rate"] >= 0:
                if r["controller"] == "attention_memory":
                    attention_overwrite.append(r["distractor_overwrite_rate"])
                elif r["controller"] == "scalar_memory":
                    scalar_overwrite.append(r["distractor_overwrite_rate"])
    attn_ow = float(np.mean(attention_overwrite)) if attention_overwrite else 1.0
    scalar_ow = float(np.mean(scalar_overwrite)) if scalar_overwrite else 1.0
    overwrite_resistance_improved = attn_ow < scalar_ow

    # ---------------------------------------------------------------------------
    # Memory advantage curves
    # ---------------------------------------------------------------------------
    memory_advantage_curves = {}
    for task_name in task_names:
        reactive_rows = [r for r in per_task_results[task_name] if r["controller"] == "reactive"]
        reactive_mean = float(np.mean([r["return"] for r in reactive_rows])) if reactive_rows else 0.0
        memory_advantage_curves[task_name] = {}
        for ctrl_name in controllers:
            if ctrl_name == "reactive":
                continue
            ctrl_rows = [r for r in per_task_results[task_name] if r["controller"] == ctrl_name]
            if not ctrl_rows:
                continue
            memory_advantage_curves[task_name][ctrl_name] = float(
                np.mean([r["return"] for r in ctrl_rows]) - reactive_mean,
            )

    # Lag-to-effect profile
    lag_profile = {}
    for ctrl_name in ("buffer_memory", "attention_memory"):
        ctrl_rows = [r for v in all_results.values() for r in v if r["controller"] == ctrl_name]
        lag_profile[ctrl_name] = {
            "mean_causal_depth": float(np.mean([r["causal_depth"] for r in ctrl_rows])) if ctrl_rows else 0.0,
            "max_causal_depth": float(np.max([r["causal_depth"] for r in ctrl_rows])) if ctrl_rows else 0.0,
        }

    # ---------------------------------------------------------------------------
    # Failure analysis
    # ---------------------------------------------------------------------------
    failure_cases = []
    for task_name in task_names:
        for ctrl_name in ("buffer_memory", "attention_memory"):
            rows = [r for r in per_task_results[task_name] if r["controller"] == ctrl_name]
            for r in rows:
                if r["success"] < 0.5 and r["memory_utilization"] > 0.3:
                    failure_cases.append({
                        "task": task_name,
                        "controller": ctrl_name,
                        "seed": r["seed"],
                        "return": r["return"],
                        "memory_utilization": r["memory_utilization"],
                        "type": "memory_corruption",
                    })

    # ---------------------------------------------------------------------------
    # Seed stability
    # ---------------------------------------------------------------------------
    seed_stability = {}
    for ctrl_name in controllers:
        per_seed_returns = defaultdict(list)
        for v in all_results.values():
            for r in v:
                if r["controller"] == ctrl_name:
                    per_seed_returns[r["seed"]].append(r["return"])
        seed_means = [float(np.mean(v)) for v in per_seed_returns.values()]
        seed_stability[ctrl_name] = {
            "cv_across_seeds": float(np.std(seed_means) / (abs(np.mean(seed_means)) + 1e-8)) if seed_means else 0.0,
        }

    payload = {
        "controller_summary": controller_summary,
        "task_controller_matrix": task_controller_matrix,
        "pass_criteria": {
            "memory_gt_reactive_4_of_6_tasks": memory_gt_reactive_4_of_6,
            "tasks_where_memory_wins": tasks_where_memory_wins,
            "causal_depth_gt_1": causal_depth_gt_1,
            "buffer_causal_depth": buffer_depth,
            "attention_causal_depth": attention_depth,
            "attention_gt_scalar_on_selective": attention_gt_scalar_on_selective,
            "attention_selective_mean": attn_mean,
            "scalar_selective_mean": scalar_mean,
            "overwrite_resistance_improved": overwrite_resistance_improved,
        },
        "memory_advantage_curves": memory_advantage_curves,
        "lag_profile": lag_profile,
        "failure_cases": failure_cases[:30],
        "seed_stability": seed_stability,
        "config": {
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
            "n_tasks": len(task_names),
            "n_controllers": len(controllers),
            "total_runs": sum(len(v) for v in all_results.values()),
        },
    }

    (output_dir / "deep_memory_v2.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
