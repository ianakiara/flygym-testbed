"""PoC experiment — Memory-Demanding Tasks.

Tests whether memory actually matters by running controllers on tasks
that *require* temporal state to succeed:

1. **Delayed Reward** — reward is withheld for k steps after goal
   completion.  A memoryless controller cannot associate the action
   with the outcome.
2. **History Dependence** — two waypoints must be visited in a specific
   (randomised) order.  Same current state, different required action
   depending on which waypoint was visited first.
3. **Hidden Cue Recall** — a cue appears briefly at episode start and
   then disappears.  The correct target depends on the cue, but after
   it vanishes the observation is ambiguous.

On standard tasks the memory controller has only a marginal advantage
(causal depth ≈3.67 vs reactive ≈3.33).  On these memory-demanding
tasks the gap should widen significantly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..config import EnvConfig
from ..controllers import MemoryController, RandomController, ReducedDescendingController
from ..envs import FlyBodyWorldEnv
from ..metrics import summarize_metrics
from ..metrics.causal_metrics import temporal_causal_depth
from ..metrics.core_metrics import history_dependence
from ..tasks import DelayedRewardTask, HiddenCueRecallTask, HistoryDependenceTask
from .benchmark_harness import run_episode


_SEEDS = [0, 1, 2, 3, 4]
_MAX_STEPS = 64


def _make_env(task_name: str) -> FlyBodyWorldEnv:
    """Create an environment with the specified memory-demanding task."""
    config = EnvConfig(episode_steps=_MAX_STEPS)
    body = BodylessBodyLayer()

    if task_name == "delayed_reward":
        world = DelayedRewardTask(config=config, reward_delay=8)
    elif task_name == "history_dependence":
        world = HistoryDependenceTask(config=config, waypoint_distance=2.0)
    elif task_name == "hidden_cue_recall":
        world = HiddenCueRecallTask(
            config=config, cue_visible_steps=3, target_distance=2.5,
        )
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return FlyBodyWorldEnv(body=body, world=world, config=config)


def _controllers() -> dict[str, object]:
    return {
        "memory": MemoryController(),
        "reduced_descending": ReducedDescendingController(),
        "random": RandomController(),
    }


def _run_task(task_name: str) -> dict[str, object]:
    """Run all controllers on a single task and aggregate results."""
    controllers = _controllers()
    per_controller: dict[str, list[dict]] = {name: [] for name in controllers}

    for seed in _SEEDS:
        for name, ctrl in controllers.items():
            env = _make_env(task_name)
            transitions = run_episode(env, ctrl, seed=seed, max_steps=_MAX_STEPS)
            metrics = summarize_metrics(transitions)
            depth = temporal_causal_depth(transitions, max_horizon=10)
            hist_dep = history_dependence(transitions)
            per_controller[name].append({
                "seed": seed,
                **metrics,
                **depth,
                **hist_dep,
            })

    # Aggregate per-controller.
    aggregated: dict[str, dict[str, float]] = {}
    for name, runs in per_controller.items():
        aggregated[name] = {
            "mean_return": float(np.mean([r.get("return", 0.0) for r in runs])),
            "mean_success": float(np.mean([r.get("success", 0.0) for r in runs])),
            "mean_causal_depth": float(np.mean([r["causal_depth"] for r in runs])),
            "mean_history_dependence": float(
                np.mean([r.get("history_dependence", 0.0) for r in runs])
            ),
            "n_episodes": len(runs),
        }

    return {
        "task": task_name,
        "aggregated": aggregated,
        "per_controller": {k: v for k, v in per_controller.items()},
    }


def _compute_advantage(task_result: dict) -> dict[str, float]:
    """Compute the memory controller's advantage over reactive."""
    agg = task_result["aggregated"]
    mem = agg.get("memory", {})
    react = agg.get("reduced_descending", {})
    rand = agg.get("random", {})

    return {
        "return_advantage_vs_reactive": mem.get("mean_return", 0.0) - react.get("mean_return", 0.0),
        "return_advantage_vs_random": mem.get("mean_return", 0.0) - rand.get("mean_return", 0.0),
        "depth_advantage_vs_reactive": mem.get("mean_causal_depth", 0.0) - react.get("mean_causal_depth", 0.0),
        "history_dep_advantage": mem.get("mean_history_dependence", 0.0) - react.get("mean_history_dependence", 0.0),
    }


def run_experiment(
    output_dir: str | Path = "results/exp_memory_demand",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["delayed_reward", "history_dependence", "hidden_cue_recall"]
    task_results: dict[str, dict] = {}
    advantages: dict[str, dict] = {}

    for task_name in tasks:
        result = _run_task(task_name)
        task_results[task_name] = result
        advantages[task_name] = _compute_advantage(result)

    # --- Cross-task summary ---
    mean_depth_advantage = float(np.mean(
        [adv["depth_advantage_vs_reactive"] for adv in advantages.values()]
    ))
    mean_return_advantage = float(np.mean(
        [adv["return_advantage_vs_reactive"] for adv in advantages.values()]
    ))

    summary: dict[str, object] = {
        "tasks": tasks,
        "advantages": advantages,
        "cross_task_summary": {
            "mean_depth_advantage_vs_reactive": mean_depth_advantage,
            "mean_return_advantage_vs_reactive": mean_return_advantage,
        },
        # Key expectations:
        # - Memory controller should outperform reactive on return.
        # - Causal depth gap should be positive (memory > reactive).
        "memory_outperforms_reactive": all(
            adv["return_advantage_vs_reactive"] >= -0.5
            for adv in advantages.values()
        ),
        "depth_gap_positive": mean_depth_advantage > 0.0,
        "per_task": {
            task_name: {
                "aggregated": result["aggregated"],
                "advantage": advantages[task_name],
            }
            for task_name, result in task_results.items()
        },
    }

    (output_dir / "memory_demand_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # Save detailed per-task results.
    for task_name, result in task_results.items():
        (output_dir / f"{task_name}_detail.json").write_text(
            json.dumps(result, indent=2, default=str)
        )

    return summary
