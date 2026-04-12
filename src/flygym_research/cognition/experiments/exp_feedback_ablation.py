"""Experiment 3 — Ascending feedback ablation.

Turn off or degrade ascending channels.

Question: Which feedback channels are necessary for stable control and
self/world disambiguation?
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

EXPERIMENT_NAME = "Ascending Feedback Ablation"

# Channel groups to ablate (matching AscendingAdapter channel groups).
CHANNEL_GROUPS = [
    frozenset(),  # No ablation (baseline).
    frozenset({"pose"}),
    frozenset({"locomotion"}),
    frozenset({"contact"}),
    frozenset({"target"}),
    frozenset({"internal"}),
    frozenset({"pose", "locomotion"}),
    frozenset({"pose", "locomotion", "contact"}),
    frozenset({"pose", "locomotion", "contact", "target", "internal"}),  # All off.
]


def make_env(disabled_channels: frozenset[str]) -> FlyBodyWorldEnv:
    cfg = EnvConfig()
    body_cfg = BodyLayerConfig(disabled_feedback_channels=disabled_channels)
    return FlyBodyWorldEnv(
        body=FlyBodyLayer(config=body_cfg),
        world=SimplifiedEmbodiedWorld(config=cfg),
        config=cfg,
    )


def run_experiment(
    output_dir: str | Path = "results/exp_feedback_ablation",
) -> list[ExperimentResult]:
    output_dir = Path(output_dir)
    all_results: list[ExperimentResult] = []

    controllers = {
        "reduced_descending": ReducedDescendingController(),
        "memory": MemoryController(),
    }

    for disabled in CHANNEL_GROUPS:
        condition_name = (
            "baseline_no_ablation"
            if not disabled
            else f"ablate_{'_'.join(sorted(disabled))}"
        )
        for ctrl_name, ctrl in controllers.items():
            all_results.extend(
                run_condition(
                    lambda d=disabled: make_env(d),
                    ctrl,
                    ctrl_name,
                    f"{condition_name}_{ctrl_name}",
                    EXPERIMENT_NAME,
                )
            )

    write_csv(all_results, output_dir / "exp_feedback_ablation.csv")
    return all_results


def generate_report(results: list[ExperimentResult]) -> str:
    return format_report(
        experiment_name=EXPERIMENT_NAME,
        hypothesis=(
            "Stability and locomotion feedback channels are necessary for coherent "
            "higher-level control.  Ablating them should degrade task performance, "
            "state persistence, and self/world separation more than ablating "
            "target-related channels."
        ),
        method=(
            "Systematically disable ascending feedback channel groups (pose, "
            "locomotion, contact, target, internal) individually and in combinations.  "
            "Run reduced-descending and memory controllers across all ablation "
            "conditions.  Compare core metrics against the unablated baseline."
        ),
        results=results,
        conclusions=[
            "Stability channel expected to be most critical for coherent control.",
            "Target channel ablation should degrade task performance but not stability metrics.",
            "Full ablation should collapse behaviour to random-like performance.",
            "Graded ablation reveals channel importance hierarchy.",
        ],
        caveats=[
            "Ablation zeros features rather than removing them — controllers may adapt to zeros.",
            "Only two controller types tested; results may not generalise to trained controllers.",
            "Channel groups are coarse — individual features within a group may vary in importance.",
        ],
        failure_cases=[
            "If no single channel ablation degrades performance, channels are redundant.",
            "If all ablations equally degrade performance, no channel hierarchy exists.",
        ],
        next_steps=[
            "Fine-grained per-feature ablation within each channel group.",
            "Noise injection instead of zeroing (partial degradation).",
            "Test with trained controllers that may have learned channel dependencies.",
        ],
    )
