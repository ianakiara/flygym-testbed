"""Experiment 8 — Shared target/object test.

In multi-condition setups, test whether stable target representation survives
controller swap, viewpoint change, world change, and sensory noise.

Question: Is there a controller-invariant target/object representation?
"""

from __future__ import annotations

from pathlib import Path

from ..body_reflex import FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    MemoryController,
    PlannerController,
    RandomController,
    ReducedDescendingController,
)
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..experiments.benchmark_harness import run_episode
from ..experiments.experiment_utils import (
    ExperimentResult,
    format_report,
    standard_seeds,
    write_csv,
)
from ..metrics import (
    shared_objectness_score,
    summarize_metrics,
    target_representation_stability,
)
from ..worlds import SimplifiedEmbodiedWorld

EXPERIMENT_NAME = "Shared Target/Object Test"


def make_env() -> FlyBodyWorldEnv:
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
    output_dir: str | Path = "results/exp_shared_target_test",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "random": RandomController(),
    }

    env_factories = {
        "simplified": make_env,
        "avatar": make_avatar_env,
    }

    # Collect transitions for cross-condition objectness comparison.
    all_transitions: dict[str, list] = {}

    for world_name, factory in env_factories.items():
        for ctrl_name, ctrl in controllers.items():
            condition = f"{world_name}_{ctrl_name}"
            for seed in standard_seeds():
                env = factory()
                transitions = run_episode(env, ctrl, seed=seed)
                base_metrics = summarize_metrics(transitions)
                target_metrics = target_representation_stability(transitions)
                combined = {**base_metrics, **target_metrics}

                all_results.append(
                    ExperimentResult(
                        experiment_name=EXPERIMENT_NAME,
                        condition=condition,
                        controller_name=ctrl_name,
                        world_mode=world_name,
                        seed=seed,
                        metrics=combined,
                    )
                )
                all_transitions.setdefault(condition, []).extend(transitions)

    # Compute cross-condition objectness scores for key pairs.
    pairs = [
        ("simplified_reduced_descending", "simplified_memory"),
        ("simplified_reduced_descending", "avatar_reduced_descending"),
        ("simplified_memory", "avatar_memory"),
        ("simplified_reduced_descending", "simplified_random"),
    ]
    for cond_a, cond_b in pairs:
        if cond_a in all_transitions and cond_b in all_transitions:
            score = shared_objectness_score(
                all_transitions[cond_a], all_transitions[cond_b]
            )
            all_results.append(
                ExperimentResult(
                    experiment_name=EXPERIMENT_NAME,
                    condition=f"cross_{cond_a}_vs_{cond_b}",
                    controller_name="cross_comparison",
                    world_mode="cross",
                    seed=-1,
                    metrics=score,
                )
            )

    write_csv(all_results, output_dir / "exp_shared_target_test.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Stable target representations survive controller swap and world change "
            "when the body substrate is preserved.  Controllers with memory should "
            "show higher shared objectness scores than reactive or random controllers."
        ),
        method=(
            "Run four controllers across two world modes (simplified embodied, avatar).  "
            "Compute target representation stability per condition and shared objectness "
            "scores across key condition pairs: controller swap (same world), world swap "
            "(same controller), and sanity check (vs random)."
        ),
        results=results,
        conclusions=[
            "Cross-controller objectness score measures representation invariance.",
            "Cross-world objectness score measures world-agnostic target tracking.",
            "Random controller should show low objectness as negative control.",
            "High cross-condition scores are candidate markers for controller-invariant object tracking.",
        ],
        caveats=[
            "Objectness proxy uses target_vector trajectory statistics — not true object representations.",
            "Shared objectness is a necessary but not sufficient condition for genuine object tracking.",
            "Heuristic controllers don't form learned representations — this tests architecture properties.",
        ],
        failure_cases=[
            "If all cross-condition scores are equally high, the metric is too coarse.",
            "If random controller shows high objectness, target_vector statistics are trivially stable.",
        ],
        next_steps=[
            "Use neural controllers to test learned target representations.",
            "Add viewpoint perturbation (rotation of observation frame).",
            "Inject sensory noise to test representation robustness.",
        ],
    )
