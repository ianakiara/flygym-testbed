from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..sleep import benchmark_portable_replay, compress_trace_bank
from .exp_sleep_trace_compressor import collect_trace_bank


def _policy_score(policy: str, row: dict[str, float]) -> float:
    if policy == "best_return_only":
        return row.get("mean_return", 0.0)
    if policy == "low_drift_only":
        return -row.get("scale_drift", 0.0)
    if policy == "backbone_shared_only":
        return row.get("backbone_shared_score", 0.0)
    if policy == "transfer_guided":
        return row.get("functional_transfer_gain", 0.0)
    if policy == "combined_policy":
        return (
            row.get("backbone_shared_score", 0.0)
            + row.get("functional_transfer_gain", 0.0)
            + row.get("mean_success", 0.0)
            - row.get("scale_drift", 0.0)
            - row.get("seam_risk", 0.0)
            - row.get("degeneracy_penalty", 0.0)
        )
    raise ValueError(policy)


def run_experiment(
    output_dir: str | Path = "results/exp_sleep_retention_policy",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=32)
    artifact = compress_trace_bank(episodes)
    replay = benchmark_portable_replay(episodes, artifact)
    by_candidate = {row["candidate_id"]: row for row in replay["per_candidate"]}
    candidate_rows = []
    for candidate in artifact.candidates:
        row = {
            "candidate_id": candidate.candidate_id,
            "redundancy_tier": candidate.redundancy_tier,
            **candidate.score_components,
            **candidate.functional_utility,
        }
        row.update(by_candidate.get(candidate.candidate_id, {}))
        candidate_rows.append(row)
    keep_n = max(1, len(candidate_rows) // 2)
    policy_results = {}
    for policy in [
        "best_return_only",
        "low_drift_only",
        "backbone_shared_only",
        "transfer_guided",
        "combined_policy",
    ]:
        selected = sorted(candidate_rows, key=lambda row: _policy_score(policy, row), reverse=True)[:keep_n]
        policy_results[policy] = {
            "selected_candidate_ids": [row["candidate_id"] for row in selected],
            "mean_backbone_shared": float(np.mean([row.get("backbone_shared_score", 0.0) for row in selected])),
            "mean_return_lift": float(np.mean([row.get("mean_return_lift", 0.0) for row in selected])),
            "mean_success_lift": float(np.mean([row.get("mean_success_lift", 0.0) for row in selected])),
            "mean_functional_transfer_gain": float(np.mean([row.get("functional_transfer_gain", 0.0) for row in selected])),
        }
    summary = {"policy_results": policy_results, "n_candidates": len(candidate_rows), "keep_n": keep_n}
    (output_dir / "retention_policy_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
