from __future__ import annotations

import json
from pathlib import Path

from ..sleep import compress_trace_bank
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(
    output_dir: str | Path = "results/exp_composition_gluing",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=24)
    artifact = compress_trace_bank(episodes)
    categories = {
        "admissible_parts": [],
        "near_admissible_parts": [],
        "false_friends": [],
        "seam_only_failures": [],
        "observer_only_failures": [],
        "cross_world_seam_failures": [],
    }
    for candidate in artifact.candidates:
        score = candidate.score_components
        row = {
            "candidate_id": candidate.candidate_id,
            "tier": candidate.redundancy_tier,
            "backbone_shared_score": score.get("backbone_shared_score", 0.0),
            "seam_risk": score.get("seam_risk", 0.0),
            "interop_loss": score.get("interop_loss", 0.0),
            "scale_drift": score.get("scale_drift", 0.0),
            "degeneracy_penalty": score.get("degeneracy_penalty", 0.0),
        }
        if row["backbone_shared_score"] > 0.2 and row["seam_risk"] < 0.2 and row["degeneracy_penalty"] < 0.2:
            categories["admissible_parts"].append(row)
        elif row["backbone_shared_score"] > 0.0 and row["seam_risk"] < 0.3:
            categories["near_admissible_parts"].append(row)
        if row["degeneracy_penalty"] >= 0.2 and row["backbone_shared_score"] <= 0.1:
            categories["false_friends"].append(row)
        if row["seam_risk"] >= 0.25 and row["interop_loss"] < 0.25:
            categories["seam_only_failures"].append(row)
        if row["interop_loss"] >= 0.25 and row["seam_risk"] < 0.25:
            categories["observer_only_failures"].append(row)
        if candidate.redundancy_tier in {"portable", "universal"} and row["seam_risk"] >= 0.25:
            categories["cross_world_seam_failures"].append(row)
    summary = {key: len(value) for key, value in categories.items()}
    payload = {"summary": summary, "categories": categories}
    (output_dir / "composition_gluing.json").write_text(json.dumps(payload, indent=2))
    return payload
