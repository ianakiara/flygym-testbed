from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import MemoryController, ReducedDescendingController, SelectiveMemoryController
from ..envs import FlyBodyWorldEnv
from ..experiments.experiment_utils import ExperimentResult, write_csv
from ..metrics import summarize_metrics, temporal_causal_depth
from ..tasks import ConditionalSequenceTask, DelayedInterferenceTask, DistractorCueRecallTask
from .benchmark_harness import run_episode

BENCHMARK_SEEDS = (0, 1, 2)


def _make_env(task_name: str) -> FlyBodyWorldEnv:
    cfg = EnvConfig(episode_steps=48)
    body = BodylessBodyLayer(config=BodyLayerConfig())
    if task_name == "distractor_cue_recall":
        return FlyBodyWorldEnv(body=body, world=DistractorCueRecallTask(config=cfg), config=cfg)
    if task_name == "conditional_sequence":
        return FlyBodyWorldEnv(body=body, world=ConditionalSequenceTask(config=cfg), config=cfg)
    if task_name == "delayed_interference":
        return FlyBodyWorldEnv(body=body, world=DelayedInterferenceTask(config=cfg), config=cfg)
    raise ValueError(task_name)


def run_experiment(
    output_dir: str | Path = "results/exp_selective_memory_benchmark",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "selective_memory": SelectiveMemoryController(),
    }
    tasks = ["distractor_cue_recall", "conditional_sequence", "delayed_interference"]
    rows: list[ExperimentResult] = []
    summary_rows = []
    for task_name in tasks:
        for controller_name, controller in controllers.items():
            metrics_per_seed = []
            for seed in BENCHMARK_SEEDS:
                env = _make_env(task_name)
                transitions = run_episode(env, controller, seed=seed, max_steps=48)
                metrics = summarize_metrics(transitions)
                metrics.update(temporal_causal_depth(transitions, max_horizon=6))
                metrics_per_seed.append(metrics)
                rows.append(
                    ExperimentResult(
                        experiment_name="Selective Memory Benchmark",
                        condition=f"{task_name}_{controller_name}",
                        controller_name=controller_name,
                        world_mode=task_name,
                        seed=seed,
                        metrics=metrics,
                    )
                )
            summary_rows.append(
                {
                    "task": task_name,
                    "controller": controller_name,
                    "return_mean": float(np.mean([m.get("return", 0.0) for m in metrics_per_seed])),
                    "success_mean": float(np.mean([m.get("success", 0.0) for m in metrics_per_seed])),
                    "causal_depth_mean": float(np.mean([m.get("causal_depth", 0.0) for m in metrics_per_seed])),
                }
            )
    write_csv(rows, output_dir / "selective_memory_benchmark.csv")
    summary = {"summary_rows": summary_rows}
    (output_dir / "selective_memory_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
