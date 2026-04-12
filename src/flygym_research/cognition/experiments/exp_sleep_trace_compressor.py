from __future__ import annotations

import json
from pathlib import Path

from ..body_reflex import BodylessBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..controllers import (
    BodylessAvatarController,
    MemoryController,
    NoAscendingFeedbackController,
    PlannerController,
    RandomController,
    RawControlController,
    ReducedDescendingController,
    ReflexOnlyController,
)
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..sleep import CompressionConfig, TraceEpisode, TraceStore, compress_trace_bank
from ..sleep.reporting import sleep_artifact_to_markdown
from ..worlds import NativePhysicalWorld, SimplifiedEmbodiedWorld
from .benchmark_harness import run_episode



def _controllers() -> dict[str, object]:
    return {
        "reflex_only": ReflexOnlyController(),
        "random": RandomController(),
        "reduced_descending": ReducedDescendingController(),
        "no_ascending": NoAscendingFeedbackController(),
        "raw_control": RawControlController(),
        "memory": MemoryController(),
        "planner": PlannerController(),
        "bodyless": BodylessAvatarController(),
    }



def _make_env(
    world_mode: str,
    *,
    disabled_channels: frozenset[str],
    perturbation_tag: str,
):
    env_config = EnvConfig(
        avatar_noise_scale=0.08 if perturbation_tag == "noisy" else 0.02,
        avatar_external_event_period=3 if perturbation_tag == "noisy" else 7,
    )
    body = BodylessBodyLayer(config=BodyLayerConfig(disabled_feedback_channels=disabled_channels))
    if world_mode == "avatar_remapped":
        return FlyAvatarEnv(body=body, env_config=env_config)
    if world_mode == "native_physical":
        return FlyBodyWorldEnv(
            body=body,
            world=NativePhysicalWorld(config=env_config),
            config=env_config,
        )
    if world_mode == "simplified_embodied":
        return FlyBodyWorldEnv(
            body=body,
            world=SimplifiedEmbodiedWorld(config=env_config),
            config=env_config,
        )
    raise ValueError(f"Unknown world mode: {world_mode}")



def collect_trace_bank(
    *,
    seeds: list[int] | None = None,
    world_modes: list[str] | None = None,
    ablations: list[frozenset[str]] | None = None,
    perturbation_tags: list[str] | None = None,
    max_steps: int = 48,
) -> list[TraceEpisode]:
    seeds = seeds or [0, 1]
    world_modes = world_modes or [
        "avatar_remapped",
        "native_physical",
        "simplified_embodied",
    ]
    ablations = ablations or [frozenset(), frozenset({"pose"})]
    perturbation_tags = perturbation_tags or ["baseline", "noisy"]

    episodes: list[TraceEpisode] = []
    for world_mode in world_modes:
        for perturbation_tag in perturbation_tags:
            for disabled_channels in ablations:
                for controller_name, controller in _controllers().items():
                    for seed in seeds:
                        env = _make_env(
                            world_mode,
                            disabled_channels=disabled_channels,
                            perturbation_tag=perturbation_tag,
                        )
                        transitions = run_episode(env, controller, seed=seed, max_steps=max_steps)
                        episodes.append(
                            TraceEpisode(
                                controller_name=controller_name,
                                world_mode=world_mode,
                                seed=seed,
                                transitions=transitions,
                                controller_state=controller.get_internal_state(),
                                body_state=env.body.get_internal_state(),
                                ablation_channels=tuple(sorted(disabled_channels)),
                                perturbation_tag=perturbation_tag,
                                metadata={
                                    "episode_steps": len(transitions),
                                    "history_length": env.config.history_length,
                                },
                            )
                        )
    return episodes



def run_experiment(
    output_dir: str | Path = "results/exp_sleep_trace_compressor",
    *,
    config: CompressionConfig | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = TraceStore(output_dir)
    episodes = collect_trace_bank()
    trace_path = store.save_trace_bank(
        episodes,
        filename="trace_bank.json",
        metadata={"experiment": "exp_sleep_trace_compressor"},
    )
    artifact = compress_trace_bank(episodes, trace_bank_path=str(trace_path), config=config)
    store.save_sleep_artifact(artifact, filename="sleep_artifact.json")
    (output_dir / "sleep_report.md").write_text(sleep_artifact_to_markdown(artifact))
    summary = {
        "artifact": artifact.to_dict(),
        "n_episodes": len(episodes),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
