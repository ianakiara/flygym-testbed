"""Experiment 7 — Reduced-control vs raw-control.

Question: Does reduced descending control improve compositionality and
stability, or just cap performance?
"""

from __future__ import annotations

from pathlib import Path

from ..body_reflex import FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    RandomController,
    RawControlController,
    ReducedDescendingController,
    MemoryController,
)
from ..envs import FlyBodyWorldEnv
from ..experiments.experiment_utils import (
    ExperimentResult,
    format_report,
    run_condition,
    write_csv,
)
from ..worlds import SimplifiedEmbodiedWorld

EXPERIMENT_NAME = "Reduced-Control vs Raw-Control"


def make_env() -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=BodyLayerConfig()),
        world=SimplifiedEmbodiedWorld(config=cfg),
        config=cfg,
    )


def run_experiment(
    output_dir: str | Path = "results/exp_reduced_vs_raw",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "raw_control": RawControlController(),
        "memory": MemoryController(),
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

    write_csv(all_results, output_dir / "exp_reduced_vs_raw.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Reduced descending control improves compositionality (lower seam "
            "fragility) and stability (higher stabilisation quality) compared to "
            "raw full-body control, at the potential cost of peak task performance."
        ),
        method=(
            "Compare reduced descending controller, raw control controller, "
            "memory controller, and random baseline on the same simplified embodied "
            "world.  Focus on seam fragility, stabilisation quality, and task "
            "performance trade-offs."
        ),
        results=results,
        conclusions=[
            "Reduced control expected to show lower seam fragility than raw control.",
            "Raw control may achieve higher task performance but at cost of stability.",
            "Stability–performance trade-off quantifies the value of reduced interfaces.",
            "Memory controller combines reduced control with internal state — best of both worlds.",
        ],
        caveats=[
            "RawControlController uses heuristic actuator patterns — not optimised.",
            "Seam fragility metric is a proxy — true compositionality is harder to measure.",
            "Short episodes may not reveal long-horizon compositionality benefits.",
        ],
        failure_cases=[
            "If raw control shows better stability, reduced interface adds unnecessary abstraction.",
            "If seam fragility is identical, the descending adapter adds no value.",
        ],
        next_steps=[
            "Train both control modes end-to-end and compare.",
            "Add perturbation robustness tests.",
            "Measure compositionality via skill transfer across tasks.",
        ],
    )
