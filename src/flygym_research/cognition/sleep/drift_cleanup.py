from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..metrics.sleep_metrics import drift_staleness_score
from .trace_schema import SleepArtifact



def cleanup_memory_bank(
    artifacts: list[SleepArtifact],
    *,
    stale_threshold: float = 0.65,
) -> dict[str, Any]:
    decisions: dict[str, str] = {}
    grouped = defaultdict(list)
    for artifact in artifacts:
        grouped[artifact.metadata.get("source", artifact.artifact_id)].append(artifact)

    for group in grouped.values():
        group_sorted = sorted(
            group,
            key=lambda artifact: artifact.validation.get("compression_gain", 0.0),
            reverse=True,
        )
        best = group_sorted[0]
        decisions[best.artifact_id] = "keep"
        for artifact in group_sorted[1:]:
            score = drift_staleness_score(artifact.metadata, artifact.validation)
            if score["drift_staleness_score"] >= stale_threshold:
                decisions[artifact.artifact_id] = "prune"
            else:
                decisions[artifact.artifact_id] = "demote_to_residual"

    return {
        "artifact_decisions": decisions,
        "n_kept": sum(1 for decision in decisions.values() if decision == "keep"),
        "n_pruned": sum(1 for decision in decisions.values() if decision == "prune"),
        "n_demoted": sum(
            1 for decision in decisions.values() if decision == "demote_to_residual"
        ),
    }
