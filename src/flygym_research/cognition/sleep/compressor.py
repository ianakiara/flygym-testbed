from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..metrics.sleep_metrics import compression_gain, post_compression_robustness_delta
from .equivalence import extract_sleep_candidates
from .reporting import artifact_summary
from .residuals import select_residual_exceptions
from .scoring import residual_score, safe_compression_score
from .trace_schema import SleepArtifact, TraceEpisode, episode_bank_fingerprint


@dataclass(slots=True)
class CompressionConfig:
    min_equivalence_strength: float = 0.55
    min_safe_score: float = 0.05
    max_seam_risk: float = 0.25
    max_interop_loss: float = 0.45
    max_scale_drift: float = 0.35
    cross_world: bool = False



def _mean_metric(episodes: list[TraceEpisode], key: str) -> float:
    if not episodes:
        return 0.0
    return float(np.mean([episode.summary_metrics.get(key, 0.0) for episode in episodes]))



def compress_trace_bank(
    episodes: list[TraceEpisode],
    *,
    trace_bank_path: str | None = None,
    config: CompressionConfig | None = None,
) -> SleepArtifact:
    config = config or CompressionConfig()
    candidates = extract_sleep_candidates(
        episodes,
        min_equivalence_strength=config.min_equivalence_strength,
        cross_world=config.cross_world,
    )

    compressed_episode_ids: list[str] = []
    residual_episode_ids: list[str] = []
    for candidate in candidates:
        select_residual_exceptions(candidate, episodes)
        score = safe_compression_score(candidate, episodes)
        candidate.score_components.update(score)
        candidate.score_components.update(residual_score(candidate, episodes))
        if len(candidate.member_episode_ids) == 1:
            candidate.decision = "keep_singleton"
            compressed_episode_ids.append(candidate.representative_episode_id)
            continue
        if (
            score["safe_compression_score"] >= config.min_safe_score
            and score["seam_risk"] <= config.max_seam_risk
            and score["interop_loss"] <= config.max_interop_loss
            and score["scale_drift"] <= config.max_scale_drift
        ):
            candidate.decision = "compress"
            compressed_episode_ids.append(candidate.representative_episode_id)
            residual_episode_ids.extend(candidate.residual_episode_ids)
        else:
            candidate.decision = "review"
            compressed_episode_ids.extend(candidate.member_episode_ids)
            residual_episode_ids.extend(candidate.residual_episode_ids)

    unique_compressed = sorted(set(compressed_episode_ids))
    unique_residual = sorted(set(residual_episode_ids))
    compression = compression_gain(len(episodes), len(unique_compressed) + len(unique_residual))
    baseline_metrics = {
        "return": _mean_metric(episodes, "return"),
        "success": _mean_metric(episodes, "success"),
        "seam_fragility": _mean_metric(episodes, "seam_fragility"),
        "interoperability": float(
            np.mean(
                [
                    candidate.score_components.get("mean_equivalence_strength", 1.0)
                    for candidate in candidates
                ]
            )
        )
        if candidates
        else 1.0,
    }
    compressed_pool = [
        episode
        for episode in episodes
        if episode.episode_id in set(unique_compressed) | set(unique_residual)
    ]
    # Build a weight map so each representative is weighted by the number
    # of original episodes it stands for.  Without this, a representative
    # from a cluster of 10 counts the same as a singleton, which unfairly
    # drags down the compressed-pool average.
    _weight_map: dict[str, int] = {}
    for candidate in candidates:
        if candidate.decision == "compress":
            n_compressed = len(candidate.member_episode_ids) - len(candidate.residual_episode_ids)
            _weight_map[candidate.representative_episode_id] = max(n_compressed, 1)
            for rid in candidate.residual_episode_ids:
                _weight_map[rid] = 1
        else:
            for eid in candidate.member_episode_ids:
                _weight_map[eid] = 1
            for rid in candidate.residual_episode_ids:
                _weight_map[rid] = 1

    def _weighted_mean(pool: list[TraceEpisode], key: str) -> float:
        if not pool:
            return 0.0
        values = [ep.summary_metrics.get(key, 0.0) for ep in pool]
        weights = [float(_weight_map.get(ep.episode_id, 1)) for ep in pool]
        total_w = sum(weights)
        if total_w < 1e-12:
            return float(np.mean(values))
        return float(np.average(values, weights=weights))

    compressed_metrics = {
        "return": _weighted_mean(compressed_pool, "return"),
        "success": _weighted_mean(compressed_pool, "success"),
        "seam_fragility": _weighted_mean(compressed_pool, "seam_fragility"),
        "interoperability": baseline_metrics["interoperability"],
    }
    robustness = post_compression_robustness_delta(baseline_metrics, compressed_metrics)
    passed = (
        robustness["post_compression_robustness_delta"] >= -0.05
        and compression["compression_gain"] >= 0.0
    )
    artifact = SleepArtifact(
        artifact_id=f"sleep-{episode_bank_fingerprint(episodes)}",
        trace_bank_path=trace_bank_path,
        candidates=candidates,
        compressed_episode_ids=unique_compressed,
        residual_episode_ids=unique_residual,
        validation={
            **compression,
            **robustness,
            "baseline_return": baseline_metrics["return"],
            "compressed_return": compressed_metrics["return"],
            "baseline_success": baseline_metrics["success"],
            "compressed_success": compressed_metrics["success"],
            "pass_rate": 1.0 if passed else 0.0,
            "passed": passed,
        },
        reports={},
        metadata={
            "n_original_episodes": len(episodes),
            "n_retained_entries": len(unique_compressed) + len(unique_residual),
        },
    )
    artifact.reports = {"summary": artifact_summary(artifact)}
    return artifact
