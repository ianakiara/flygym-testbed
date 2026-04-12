"""Shared experiment utilities — common helpers for all experiment scripts.

Provides standardised episode running, CSV output, and markdown report
generation so that every experiment produces comparable artifacts.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    MemoryController,
    NoAscendingFeedbackController,
    PlannerController,
    RandomController,
    RawControlController,
    ReducedDescendingController,
    ReflexOnlyController,
)
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..experiments.benchmark_harness import BenchmarkResult, run_episode
from ..interfaces import BrainInterface, StepTransition
from ..metrics import summarize_metrics
from ..worlds import AvatarRemappedWorld, NativePhysicalWorld, SimplifiedEmbodiedWorld


def all_controllers() -> dict[str, BrainInterface]:
    """Return all baseline controllers."""
    return {
        "reflex_only": ReflexOnlyController(),
        "random": RandomController(),
        "reduced_descending": ReducedDescendingController(),
        "no_ascending_feedback": NoAscendingFeedbackController(),
        "raw_control": RawControlController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
    }


def standard_seeds() -> list[int]:
    return [0, 1, 2, 3, 4]


@dataclass
class ExperimentResult:
    experiment_name: str
    condition: str
    controller_name: str
    world_mode: str
    seed: int
    metrics: dict[str, float]
    extra: dict[str, Any] = field(default_factory=dict)


def run_condition(
    env_factory,
    controller: BrainInterface,
    controller_name: str,
    condition: str,
    experiment_name: str,
    *,
    seeds: list[int] | None = None,
    max_steps: int | None = None,
) -> list[ExperimentResult]:
    """Run a controller across seeds and collect results."""
    seeds = seeds or standard_seeds()
    results: list[ExperimentResult] = []
    for seed in seeds:
        env = env_factory()
        transitions = run_episode(env, controller, seed=seed, max_steps=max_steps)
        metrics = summarize_metrics(transitions)
        world_mode = (
            transitions[-1].observation.world.mode if transitions else "unknown"
        )
        results.append(
            ExperimentResult(
                experiment_name=experiment_name,
                condition=condition,
                controller_name=controller_name,
                world_mode=world_mode,
                seed=seed,
                metrics=metrics,
            )
        )
    return results


def write_csv(results: list[ExperimentResult], path: str | Path) -> None:
    """Write experiment results to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    all_metric_keys: set[str] = set()
    for r in results:
        all_metric_keys.update(r.metrics.keys())
    metric_keys = sorted(all_metric_keys)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "experiment",
            "condition",
            "controller",
            "world_mode",
            "seed",
        ] + metric_keys
        writer.writerow(header)
        for r in results:
            row = [
                r.experiment_name,
                r.condition,
                r.controller_name,
                r.world_mode,
                r.seed,
            ] + [r.metrics.get(k, "") for k in metric_keys]
            writer.writerow(row)


def aggregate_results(
    results: list[ExperimentResult],
) -> dict[str, dict[str, float]]:
    """Aggregate results by condition, computing mean and std per metric."""
    from collections import defaultdict

    by_condition: dict[str, list[dict[str, float]]] = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r.metrics)

    aggregated: dict[str, dict[str, float]] = {}
    for condition, metric_dicts in by_condition.items():
        agg: dict[str, float] = {}
        all_keys = set()
        for md in metric_dicts:
            all_keys.update(md.keys())
        for key in sorted(all_keys):
            values = [md.get(key, 0.0) for md in metric_dicts]
            agg[f"{key}_mean"] = float(np.mean(values))
            agg[f"{key}_std"] = float(np.std(values))
        aggregated[condition] = agg
    return aggregated


def format_report(
    experiment_name: str,
    hypothesis: str,
    method: str,
    results: list[ExperimentResult],
    conclusions: list[str],
    caveats: list[str],
    failure_cases: list[str],
    next_steps: list[str],
) -> str:
    """Generate a standardised markdown report."""
    agg = aggregate_results(results)

    lines = [
        f"# {experiment_name}",
        "",
        "## Hypothesis",
        "",
        hypothesis,
        "",
        "## Method",
        "",
        method,
        "",
        "## Baselines",
        "",
    ]

    controllers_used = sorted(set(r.controller_name for r in results))
    for c in controllers_used:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("### Aggregated results by condition")
    lines.append("")

    for condition, metrics in sorted(agg.items()):
        lines.append(f"#### {condition}")
        lines.append("")
        lines.append("| Metric | Mean | Std |")
        lines.append("| --- | --- | --- |")
        keys = sorted(set(k.rsplit("_", 1)[0] for k in metrics if k.endswith("_mean")))
        for key in keys:
            mean = metrics.get(f"{key}_mean", 0.0)
            std = metrics.get(f"{key}_std", 0.0)
            lines.append(f"| {key} | {mean:.4f} | {std:.4f} |")
        lines.append("")

    lines.append("## Results")
    lines.append("")
    for conclusion in conclusions:
        lines.append(f"- {conclusion}")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    for caveat in caveats:
        lines.append(f"- {caveat}")
    lines.append("")

    lines.append("## Failure cases")
    lines.append("")
    for fc in failure_cases:
        lines.append(f"- {fc}")
    lines.append("")

    lines.append("## Next steps")
    lines.append("")
    for ns in next_steps:
        lines.append(f"- {ns}")
    lines.append("")

    return "\n".join(lines)
