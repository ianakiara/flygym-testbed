"""Experiment 5 — History vs state.

Create matched-current-state scenarios with different histories.

Question: Is history a necessary state variable?
"""

from __future__ import annotations

from pathlib import Path

from ..body_reflex import FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    MemoryController,
    RandomController,
    ReducedDescendingController,
)
from ..envs import FlyBodyWorldEnv
from ..experiments.benchmark_harness import run_episode
from ..experiments.experiment_utils import (
    ExperimentResult,
    format_report,
    standard_seeds,
    write_csv,
)
from ..metrics import (
    cross_time_mutual_information,
    history_dependence,
    hysteresis_metric,
    predictive_utility,
    state_decay_curve,
    summarize_metrics,
)
from ..worlds import SimplifiedEmbodiedWorld

EXPERIMENT_NAME = "History vs State"


def make_env() -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=BodyLayerConfig()),
        world=SimplifiedEmbodiedWorld(config=cfg),
        config=cfg,
    )


def run_experiment(
    output_dir: str | Path = "results/exp_history_vs_state",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "random": RandomController(),
    }

    for seed in standard_seeds():
        for ctrl_name, ctrl in controllers.items():
            env = make_env()
            transitions = run_episode(env, ctrl, seed=seed)
            base_metrics = summarize_metrics(transitions)

            # Advanced persistence metrics.
            mi = cross_time_mutual_information(transitions)
            decay = state_decay_curve(transitions)
            pred = predictive_utility(transitions)
            hyst = hysteresis_metric(transitions)
            hd = history_dependence(transitions)

            combined = {
                **base_metrics,
                **mi,
                **decay,
                **pred,
                **hyst,
                **hd,
            }

            world_mode = (
                transitions[-1].observation.world.mode if transitions else "unknown"
            )
            all_results.append(
                ExperimentResult(
                    experiment_name=EXPERIMENT_NAME,
                    condition=ctrl_name,
                    controller_name=ctrl_name,
                    world_mode=world_mode,
                    seed=seed,
                    metrics=combined,
                )
            )

    write_csv(all_results, output_dir / "exp_history_vs_state.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Controllers with explicit memory (MemoryController) show stronger "
            "history dependence, higher cross-time mutual information, slower state "
            "decay, and greater hysteresis than memoryless controllers."
        ),
        method=(
            "Run three controllers (reduced descending with scalar memory trace, "
            "memory-augmented with explicit buffer, random baseline) on the same "
            "simplified embodied world.  Compute advanced persistence metrics: "
            "cross-time MI, state decay curves, predictive utility of past state, "
            "and hysteresis scores."
        ),
        results=results,
        conclusions=[
            "Memory controller expected to show highest MI and slowest decay.",
            "Random controller should show near-zero history dependence (negative control).",
            "Reduced-descending controller's scalar memory trace provides intermediate history dependence.",
            "Hysteresis score discriminates truly history-dependent from reactive controllers.",
        ],
        caveats=[
            "Heuristic controllers have simple memory — not comparable to learned representations.",
            "MI estimation via histograms is coarse for short time series.",
            "Hysteresis depends on state binning resolution.",
        ],
        failure_cases=[
            "If memory controller shows no more history dependence than reduced descending, "
            "the explicit memory buffer adds no value.",
            "If random controller shows non-zero history dependence, metric is unreliable.",
        ],
        next_steps=[
            "Test with trained RNN controllers for genuine learned persistence.",
            "Use longer episodes to improve MI estimation quality.",
            "Create explicitly matched-state scenarios (same obs, different history) for controlled comparison.",
        ],
    )
