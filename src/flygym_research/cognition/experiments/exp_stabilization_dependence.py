"""Experiment 4 — Head/body stabilisation dependence.

Perturb terrain, oscillation, or self-motion.

Question: Does stabilisation act like a necessary prerequisite for
higher-level coherent performance?
"""

from __future__ import annotations

from pathlib import Path

from ..body_reflex import FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import MemoryController, ReducedDescendingController
from ..envs import FlyBodyWorldEnv
from ..experiments.experiment_utils import (
    ExperimentResult,
    format_report,
    run_condition,
    write_csv,
)
from ..worlds import SimplifiedEmbodiedWorld

EXPERIMENT_NAME = "Head/Body Stabilisation Dependence"


def make_env(stabilization_gain: float) -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    body_cfg = BodyLayerConfig(stabilization_gain=stabilization_gain)
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=body_cfg),
        world=SimplifiedEmbodiedWorld(config=cfg),
        config=cfg,
    )


def run_experiment(
    output_dir: str | Path = "results/exp_stabilization_dependence",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    # Vary stabilisation gain from full to zero.
    stabilization_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
    }

    for gain in stabilization_levels:
        condition = f"stab_gain_{gain:.2f}"
        for ctrl_name, ctrl in controllers.items():
            all_results.extend(
                run_condition(
                    lambda g=gain: make_env(g),
                    ctrl,
                    ctrl_name,
                    f"{condition}_{ctrl_name}",
                    EXPERIMENT_NAME,
                )
            )

    write_csv(all_results, output_dir / "exp_stabilization_dependence.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Stabilisation acts as a prerequisite for coherent higher-level control.  "
            "Reducing stabilisation gain should degrade task performance, state "
            "persistence, and self/world separation in a graded manner."
        ),
        method=(
            "Parametrically vary the body layer's stabilisation gain from 0.0 (no "
            "stabilisation) to 1.0 (full stabilisation).  Run reduced-descending and "
            "memory controllers at each level.  Measure how core metrics degrade as "
            "stabilisation is reduced."
        ),
        results=results,
        conclusions=[
            "Expected dose-response: lower gain → worse higher-level coherence.",
            "Stabilisation is a candidate prerequisite for persistent integrated control.",
            "If non-linear threshold exists, it marks a phase transition in system integration.",
        ],
        caveats=[
            "Stabilisation gain is a single parameter — real stabilisation is multi-dimensional.",
            "Simplified world may not stress the stabilisation substrate enough.",
            "Heuristic controllers may not fully exploit stabilisation benefits.",
        ],
        failure_cases=[
            "If performance is unchanged across gain levels, stabilisation is irrelevant for these tasks.",
            "If only extreme (gain=0) shows degradation, the system is robust to partial loss.",
        ],
        next_steps=[
            "Add terrain perturbations and wind-like forces.",
            "Test with oscillatory disturbances at different frequencies.",
            "Measure stabilisation quality metrics at each gain level.",
        ],
    )
