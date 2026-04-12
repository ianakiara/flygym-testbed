"""Experiment 6 — Avatar-world remapping.

Use same body substrate with different world abstractions.

Question: Can a higher controller retain coherent organisation across worlds,
or is performance only world-specific?
"""

from __future__ import annotations

from pathlib import Path

from ..body_reflex import BodylessBodyLayer, FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    MemoryController,
    PlannerController,
    RandomController,
    ReducedDescendingController,
)
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..experiments.experiment_utils import (
    ExperimentResult,
    format_report,
    run_condition,
    write_csv,
)
from ..worlds import (
    AvatarRemappedWorld,
    NativePhysicalWorld,
    SimplifiedEmbodiedWorld,
)

EXPERIMENT_NAME = "Avatar-World Remapping"


def make_native_env() -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=BodyLayerConfig()),
        world=NativePhysicalWorld(config=cfg),
        config=cfg,
    )


def make_simplified_env() -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=BodyLayerConfig()),
        world=SimplifiedEmbodiedWorld(config=cfg),
        config=cfg,
    )


def make_avatar_env() -> FlyAvatarEnv:
    cfg = EnvConfig()
    return FlyAvatarEnv(
        body=FlyBodyLayer(config=BodyLayerConfig()),
        env_config=cfg,
    )


def run_experiment(
    output_dir: str | Path = "results/exp_avatar_world_transfer",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "random": RandomController(),
    }

    world_factories = {
        "native": make_native_env,
        "simplified": make_simplified_env,
        "avatar": make_avatar_env,
    }

    for world_name, factory in world_factories.items():
        for ctrl_name, ctrl in controllers.items():
            all_results.extend(
                run_condition(
                    factory,
                    ctrl,
                    ctrl_name,
                    f"{world_name}_{ctrl_name}",
                    EXPERIMENT_NAME,
                )
            )

    write_csv(all_results, output_dir / "exp_avatar_world_transfer.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Controllers that use ascending feedback and internal state will "
            "retain coherent organisation (stable metrics) across different world "
            "modes, while reactive controllers will show world-specific performance."
        ),
        method=(
            "Run four controllers across three world modes (native physical, "
            "simplified embodied, avatar remapped) with the same fly body substrate.  "
            "Compare metric profiles across worlds to identify which controller "
            "properties transfer and which are world-specific."
        ),
        results=results,
        conclusions=[
            "Controllers with memory expected to show more consistent cross-world performance.",
            "World-specific gains indicate overfitting to world dynamics.",
            "Random controller should show uniformly low performance across worlds (negative control).",
            "Avatar world tests abstract control capability detached from physical dynamics.",
        ],
        caveats=[
            "Three world modes vary in complexity — not perfectly matched.",
            "Avatar world uses BodylessBodyLayer-like dynamics which differ from physical.",
            "Same body substrate doesn't mean same effective dynamics across worlds.",
        ],
        failure_cases=[
            "If all controllers perform equally across worlds, world remapping is trivial.",
            "If no controller transfers well, the architecture doesn't support world-agnostic control.",
        ],
        next_steps=[
            "Train controllers in one world, test in another (zero-shot transfer).",
            "Measure representation similarity across worlds via CKA.",
            "Add more diverse world modes with distinct dynamics.",
        ],
    )
