"""Experiment 2 — Controller swap / interoperability.

Swap controller families over the same body/world.

Question: What latent or behavioural invariants survive translation across
controllers?
"""

from __future__ import annotations

from pathlib import Path

from ..body_reflex import FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    MemoryController,
    NoAscendingFeedbackController,
    PlannerController,
    RandomController,
    ReducedDescendingController,
    ReflexOnlyController,
)
from ..envs import FlyBodyWorldEnv
from ..experiments.benchmark_harness import run_episode
from ..experiments.experiment_utils import (
    ExperimentResult,
    format_report,
    run_condition,
    write_csv,
)
from ..metrics import (
    interoperability_score,
    summarize_metrics,
)
from ..worlds import SimplifiedEmbodiedWorld

EXPERIMENT_NAME = "Controller Swap / Interoperability"


def make_env() -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=BodyLayerConfig()),
        world=SimplifiedEmbodiedWorld(config=cfg),
        config=cfg,
    )


def run_experiment(
    output_dir: str | Path = "results/exp_controller_swap",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "no_ascending": NoAscendingFeedbackController(),
        "reflex_only": ReflexOnlyController(),
        "random": RandomController(),
    }

    for ctrl_name, ctrl in controllers.items():
        all_results.extend(
            run_condition(
                make_env,
                ctrl,
                ctrl_name,
                ctrl_name,
                EXPERIMENT_NAME,
            )
        )

    write_csv(all_results, output_dir / "exp_controller_swap.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Different controller families will produce different action distributions "
            "but share certain latent invariants (e.g., similar stability trajectories, "
            "comparable reward curves) when operating on the same body/world."
        ),
        method=(
            "Run six controller families (reduced descending, memory, planner, "
            "no-ascending, reflex-only, random) on the same simplified embodied world "
            "across 5 seeds.  Compare action distributions, latent state trajectories, "
            "and reward curves between all pairs using interoperability metrics."
        ),
        results=results,
        conclusions=[
            "Controllers sharing ascending feedback should show higher latent correlation.",
            "Random controller serves as sanity check — should show low interoperability.",
            "Memory and planner controllers may show convergent representations.",
        ],
        caveats=[
            "Interoperability measured via summary features, not true latent states.",
            "Heuristic controllers may not produce diverse-enough behaviour.",
            "Short episodes limit observation of convergent dynamics.",
        ],
        failure_cases=[
            "If all controllers show identical metrics, the body/world dominates — controllers are irrelevant.",
            "If no pair shows high interoperability, representations are entirely controller-specific.",
        ],
        next_steps=[
            "Use trained controllers for richer representation comparison.",
            "Compute cross-controller CKA or CCA on internal states.",
            "Test interoperability across different world modes.",
        ],
    )
