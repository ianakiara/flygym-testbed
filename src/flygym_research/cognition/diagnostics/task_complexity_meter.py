"""Task Complexity Meter — formalizes memory-vs-trace gap as diagnostic.

Principle: If explicit memory does not beat reactive trace, the task is
not memory-demanding enough.  This module quantifies that gap and
classifies tasks into complexity tiers.

Usage::

    from flygym_research.cognition.diagnostics import task_complexity_meter

    result = task_complexity_meter.assess_task(
        task_class=DistractorCueRecallTask,
        controllers={"reactive": ctrl_a, "memory": ctrl_b, "slot": ctrl_c},
        n_seeds=5,
    )
    print(result["complexity_class"])  # "shallow" | "moderate" | "deep"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TaskComplexityResult:
    """Result of a task complexity assessment."""

    task_name: str
    reactive_mean_return: float
    memory_mean_return: float
    slot_mean_return: float
    random_mean_return: float
    memory_vs_trace_gap: float
    slot_vs_trace_gap: float
    slot_vs_memory_gap: float
    complexity_class: str
    n_seeds: int
    per_seed_data: list[dict[str, float]] = field(default_factory=list)


def classify_complexity(
    reactive_return: float,
    memory_return: float,
    slot_return: float,
    random_return: float,
) -> tuple[str, dict[str, float]]:
    """Classify task complexity based on controller return gaps.

    Returns (complexity_class, gaps_dict).

    Complexity tiers:
    - "trivial": all controllers including random perform similarly.
    - "shallow": reactive trace solves it; memory adds nothing.
    - "moderate": memory adds marginal advantage over trace.
    - "deep": only structured memory (slot) significantly outperforms.
    - "ultra_deep": even slot memory struggles vs reactive baseline.
    """
    memory_vs_trace = memory_return - reactive_return
    slot_vs_trace = slot_return - reactive_return
    slot_vs_memory = slot_return - memory_return
    reactive_vs_random = reactive_return - random_return

    gaps = {
        "memory_vs_trace_gap": memory_vs_trace,
        "slot_vs_trace_gap": slot_vs_trace,
        "slot_vs_memory_gap": slot_vs_memory,
        "reactive_vs_random_gap": reactive_vs_random,
    }

    # Thresholds for classification (on a normalized return scale)
    if reactive_vs_random < 0.5:
        complexity_class = "trivial"
    elif abs(memory_vs_trace) < 0.5 and abs(slot_vs_trace) < 0.5:
        complexity_class = "shallow"
    elif slot_vs_trace >= 1.0:
        complexity_class = "deep"
    elif memory_vs_trace >= 0.5 or slot_vs_trace >= 0.5:
        complexity_class = "moderate"
    else:
        complexity_class = "ultra_deep"

    return complexity_class, gaps


def compute_task_complexity(
    returns_by_controller: dict[str, list[float]],
    task_name: str = "unnamed",
) -> TaskComplexityResult:
    """Compute complexity metrics from per-seed returns by controller type.

    Parameters
    ----------
    returns_by_controller : dict
        Keys: "random", "reactive", "memory", "slot".
        Values: list of per-seed returns.
    task_name : str
        Name of the task being assessed.
    """
    random_returns = returns_by_controller.get("random", [0.0])
    reactive_returns = returns_by_controller.get("reactive", [0.0])
    memory_returns = returns_by_controller.get("memory", [0.0])
    slot_returns = returns_by_controller.get("slot", [0.0])

    random_mean = float(np.mean(random_returns))
    reactive_mean = float(np.mean(reactive_returns))
    memory_mean = float(np.mean(memory_returns))
    slot_mean = float(np.mean(slot_returns))

    complexity_class, gaps = classify_complexity(
        reactive_mean, memory_mean, slot_mean, random_mean,
    )

    n_seeds = max(
        len(random_returns), len(reactive_returns),
        len(memory_returns), len(slot_returns),
    )

    per_seed: list[dict[str, float]] = []
    for i in range(n_seeds):
        entry: dict[str, float] = {}
        if i < len(random_returns):
            entry["random"] = random_returns[i]
        if i < len(reactive_returns):
            entry["reactive"] = reactive_returns[i]
        if i < len(memory_returns):
            entry["memory"] = memory_returns[i]
        if i < len(slot_returns):
            entry["slot"] = slot_returns[i]
        per_seed.append(entry)

    return TaskComplexityResult(
        task_name=task_name,
        reactive_mean_return=reactive_mean,
        memory_mean_return=memory_mean,
        slot_mean_return=slot_mean,
        random_mean_return=random_mean,
        memory_vs_trace_gap=gaps["memory_vs_trace_gap"],
        slot_vs_trace_gap=gaps["slot_vs_trace_gap"],
        slot_vs_memory_gap=gaps["slot_vs_memory_gap"],
        complexity_class=complexity_class,
        n_seeds=n_seeds,
        per_seed_data=per_seed,
    )


def batch_complexity_assessment(
    task_results: dict[str, dict[str, list[float]]],
) -> dict[str, TaskComplexityResult]:
    """Assess complexity for a batch of tasks.

    Parameters
    ----------
    task_results : dict
        Keys: task name.
        Values: dict of controller_type → list of per-seed returns.

    Returns
    -------
    dict of task_name → TaskComplexityResult
    """
    return {
        task_name: compute_task_complexity(returns, task_name=task_name)
        for task_name, returns in task_results.items()
    }
