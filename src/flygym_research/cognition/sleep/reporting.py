from __future__ import annotations

from collections import Counter

from .trace_schema import SleepArtifact



def artifact_summary(artifact: SleepArtifact) -> dict[str, float | int]:
    validation = artifact.validation
    return {
        "n_candidates": len(artifact.candidates),
        "n_compressed": len(artifact.compressed_episode_ids),
        "n_residual": len(artifact.residual_episode_ids),
        "compression_gain": float(validation.get("compression_gain", 0.0)),
        "pass": int(bool(validation.get("passed", False))),
    }



def sleep_artifact_to_markdown(artifact: SleepArtifact) -> str:
    lines = [
        f"# Sleep Artifact {artifact.artifact_id}",
        "",
        "## Summary",
        "",
    ]
    summary = artifact_summary(artifact)
    for key, value in summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Candidate decisions", ""])
    decision_counts = Counter(candidate.decision for candidate in artifact.candidates)
    for decision, count in sorted(decision_counts.items()):
        lines.append(f"- {decision}: {count}")
    lines.extend(["", "## Validation", ""])
    for key, value in sorted(artifact.validation.items()):
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Residual exceptions", ""])
    if not artifact.residual_episode_ids:
        lines.append("- None")
    else:
        for candidate in artifact.candidates:
            for episode_id, reason in candidate.retained_exception_rationale.items():
                lines.append(f"- {episode_id}: {reason}")
    return "\n".join(lines)
