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
        "mean_backbone_shared": float(validation.get("mean_backbone_shared", 0.0)),
        "mean_safe_compression": float(validation.get("mean_safe_compression", 0.0)),
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
    lines.extend(["", "## Redundancy tiers", ""])
    tier_counts = Counter(candidate.redundancy_tier for candidate in artifact.candidates)
    for tier, count in sorted(tier_counts.items()):
        lines.append(f"- {tier}: {count}")
    lines.extend(["", "## Validation", ""])
    for key, value in sorted(artifact.validation.items()):
        lines.append(f"- **{key}**: {value}")
    lines.extend(["", "## Candidate backbone summaries", ""])
    for candidate in artifact.candidates:
        lines.append(
            f"- {candidate.candidate_id}: tier={candidate.redundancy_tier}, "
            f"backbone={candidate.score_components.get('backbone_shared_score', 0.0):.3f}, "
            f"safe={candidate.score_components.get('safe_compression_score', 0.0):.3f}, "
            f"seam={candidate.score_components.get('seam_risk', 0.0):.3f}, "
            f"drift={candidate.score_components.get('scale_drift', 0.0):.3f}, "
            f"portable={candidate.score_components.get('portability_fraction', 0.0):.3f}, "
            f"regime={candidate.score_components.get('shared_structure_regime', 'unknown')}, "
            f"transfer={candidate.score_components.get('functional_transfer_gain', 0.0):.3f}"
        )
    lines.extend(["", "## Residual exceptions", ""])
    if not artifact.residual_episode_ids:
        lines.append("- None")
    else:
        for candidate in artifact.candidates:
            for episode_id, reason in candidate.retained_exception_rationale.items():
                lines.append(f"- {episode_id}: {reason}")
    return "\n".join(lines)
