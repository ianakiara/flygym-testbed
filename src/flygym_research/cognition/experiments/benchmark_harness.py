from __future__ import annotations

from dataclasses import dataclass

from ..interfaces import BrainInterface, StepTransition
from ..metrics import summarize_metrics


@dataclass(slots=True)
class BenchmarkResult:
    controller_name: str
    world_mode: str
    seed: int
    metrics: dict[str, float]
    transitions: list[StepTransition]


def run_episode(
    env, controller: BrainInterface, *, seed: int = 0, max_steps: int | None = None
):
    observation = env.reset(seed=seed)
    controller.reset(seed=seed)
    transitions: list[StepTransition] = []
    limit = max_steps if max_steps is not None else env.config.episode_steps
    for _ in range(limit):
        action = controller.act(observation)
        transition = env.step(action)
        transitions.append(transition)
        observation = transition.observation
        if transition.terminated or transition.truncated:
            break
    return transitions


def run_baseline_suite(
    env_factory, controllers: dict[str, BrainInterface], *, seeds: list[int]
):
    results: list[BenchmarkResult] = []
    for seed in seeds:
        for name, controller in controllers.items():
            env = env_factory()
            transitions = run_episode(env, controller, seed=seed)
            results.append(
                BenchmarkResult(
                    controller_name=name,
                    world_mode=(
                        transitions[-1].observation.world.mode
                        if transitions
                        else "unknown"
                    ),
                    seed=seed,
                    metrics=summarize_metrics(transitions),
                    transitions=transitions,
                )
            )
    return results
