from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..envs import FlyAvatarEnv, FlyBodyWorldEnv
from ..metrics import summarize_metrics
from ..worlds import NativePhysicalWorld, SimplifiedEmbodiedWorld
from .scoring import backbone_shared_score, safe_compression_score
from .trace_schema import SleepArtifact, TraceEpisode


def _make_env_for_world(episode: TraceEpisode, world_mode: str):
    env_config = EnvConfig(
        avatar_noise_scale=0.08 if episode.perturbation_tag == "noisy" else 0.02,
        avatar_external_event_period=3 if episode.perturbation_tag == "noisy" else 7,
    )
    body = BodylessBodyLayer(
        config=BodyLayerConfig(disabled_feedback_channels=frozenset(episode.ablation_channels))
    )
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


def replay_episode_actions(
    source_episode: TraceEpisode,
    target_world_mode: str,
) -> dict[str, float]:
    env = _make_env_for_world(source_episode, target_world_mode)
    env.reset(seed=source_episode.seed)
    replayed = []
    for transition in source_episode.transitions:
        replayed_transition = env.step(transition.action)
        replayed.append(replayed_transition)
        if replayed_transition.terminated or replayed_transition.truncated:
            break
    return summarize_metrics(replayed)


def benchmark_portable_replay(
    episodes: list[TraceEpisode],
    artifact: SleepArtifact,
) -> dict[str, object]:
    by_id = {episode.episode_id: episode for episode in episodes}
    baselines: dict[str, dict[str, float]] = defaultdict(dict)
    by_world: dict[str, list[TraceEpisode]] = defaultdict(list)
    for episode in episodes:
        by_world[episode.world_mode].append(episode)
    for world_mode, world_episodes in by_world.items():
        baselines[world_mode] = {
            key: float(np.mean([ep.summary_metrics.get(key, 0.0) for ep in world_episodes]))
            for key in ("return", "success", "stability_mean")
        }
    per_candidate = []
    tier_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for candidate in artifact.candidates:
        representative = by_id[candidate.representative_episode_id]
        supported_worlds = set(candidate.portability_evidence.get("world_modes", [representative.world_mode]))
        replay_rows = []
        for world_mode in sorted(by_world):
            if world_mode == representative.world_mode:
                continue
            replay_metrics = replay_episode_actions(representative, world_mode)
            baseline = baselines[world_mode]
            replay_rows.append(
                {
                    "target_world_mode": world_mode,
                    "return_lift": replay_metrics.get("return", 0.0) - baseline["return"],
                    "success_lift": replay_metrics.get("success", 0.0) - baseline["success"],
                    "stability_delta": replay_metrics.get("stability_mean", 0.0) - baseline["stability_mean"],
                    "mismatch": float(world_mode not in supported_worlds),
                }
            )
        mean_return_lift = float(np.mean([row["return_lift"] for row in replay_rows])) if replay_rows else 0.0
        mean_success_lift = float(np.mean([row["success_lift"] for row in replay_rows])) if replay_rows else 0.0
        failure_under_mismatch = float(
            np.mean([
                1.0 if row["mismatch"] and row["return_lift"] < 0.0 else 0.0
                for row in replay_rows
            ])
        ) if replay_rows else 0.0
        functional_transfer_gain = float(
            np.clip(
                0.5 * np.tanh(mean_return_lift / 25.0)
                + 0.5 * np.clip(mean_success_lift, -1.0, 1.0),
                0.0,
                1.0,
            )
        )
        candidate.functional_utility.update(
            {
                "mean_return_lift": mean_return_lift,
                "mean_success_lift": mean_success_lift,
                "failure_under_mismatch": failure_under_mismatch,
                "functional_transfer_gain": functional_transfer_gain,
            }
        )
        candidate.score_components.update(
            backbone_shared_score(
                candidate,
                episodes,
                functional_transfer_gain=functional_transfer_gain,
            )
        )
        candidate.score_components["safe_compression_score"] = safe_compression_score(
            candidate,
            episodes,
        )["safe_compression_score"]
        candidate_row = {
            "candidate_id": candidate.candidate_id,
            "redundancy_tier": candidate.redundancy_tier,
            "backbone_shared_score": float(candidate.score_components.get("backbone_shared_score", 0.0)),
            "safe_compression_score": float(candidate.score_components.get("safe_compression_score", 0.0)),
            "portability_fraction": float(candidate.score_components.get("portability_fraction", 0.0)),
            "functional_transfer_gain": functional_transfer_gain,
            "mean_return_lift": mean_return_lift,
            "mean_success_lift": mean_success_lift,
            "failure_under_mismatch": failure_under_mismatch,
        }
        per_candidate.append({**candidate_row, "replay_rows": replay_rows})
        tier_rows[candidate.redundancy_tier].append(candidate_row)
    by_tier = {}
    for tier, rows in tier_rows.items():
        by_tier[tier] = {
            "n_candidates": len(rows),
            "mean_backbone_shared": float(np.mean([row["backbone_shared_score"] for row in rows])),
            "mean_safe_compression": float(np.mean([row["safe_compression_score"] for row in rows])),
            "mean_functional_transfer_gain": float(np.mean([row["functional_transfer_gain"] for row in rows])),
            "mean_return_lift": float(np.mean([row["mean_return_lift"] for row in rows])),
            "mean_success_lift": float(np.mean([row["mean_success_lift"] for row in rows])),
            "mean_failure_under_mismatch": float(np.mean([row["failure_under_mismatch"] for row in rows])),
        }
    return {
        "per_candidate": per_candidate,
        "by_tier": by_tier,
        "summary": {
            "n_candidates": len(per_candidate),
            "portable_candidates": int(sum(1 for c in artifact.candidates if c.redundancy_tier in {"portable", "universal"})),
            "universal_candidates": int(sum(1 for c in artifact.candidates if c.redundancy_tier == "universal")),
        },
    }
