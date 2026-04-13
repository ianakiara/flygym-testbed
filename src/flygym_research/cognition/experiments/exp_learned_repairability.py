from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..sleep import analyze_seam_failures, compress_trace_bank
from .exp_sleep_trace_compressor import collect_trace_bank


def _world_code(world_mode: str) -> float:
    return {
        "avatar_remapped": 0.0,
        "native_physical": 1.0,
        "simplified_embodied": 2.0,
    }.get(world_mode, 0.0)


def run_experiment(
    output_dir: str | Path = "results/exp_learned_repairability",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=24)
    artifact = compress_trace_bank(episodes)
    report = analyze_seam_failures(episodes)
    by_candidate = {
        episode_id: candidate
        for candidate in artifact.candidates
        for episode_id in candidate.member_episode_ids
    }
    rows = []
    for failure in report["failures"]:
        candidate = by_candidate.get(failure["episode_id"])
        rows.append(
            {
                "seam_fragility": float(failure["seam_fragility"]),
                "target_bias_mismatch": float(failure["target_bias_mismatch"]),
                "world_code": _world_code(str(failure["world_mode"])),
                "backbone_shared_score": float(candidate.score_components.get("backbone_shared_score", 0.0)) if candidate else 0.0,
                "portability_fraction": float(candidate.score_components.get("portability_fraction", 0.0)) if candidate else 0.0,
                "patchable_label": float(failure["recommended_action"] == "retune adapter gains"),
            }
        )
    if len(rows) < 4:
        summary = {"status": "insufficient_data", "n_rows": len(rows)}
        (output_dir / "learned_repairability.json").write_text(json.dumps(summary, indent=2))
        return summary
    split = max(1, int(len(rows) * 0.7))
    train = rows[:split]
    test = rows[split:]
    X_train = np.array([
        [
            row["seam_fragility"],
            row["target_bias_mismatch"],
            row["world_code"],
            row["backbone_shared_score"],
            row["portability_fraction"],
            1.0,
        ]
        for row in train
    ])
    y_train = np.array([row["patchable_label"] for row in train], dtype=np.float64)
    weights, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    learned_hits = 0
    baseline_hits = 0
    for row in test:
        features = np.array([
            row["seam_fragility"],
            row["target_bias_mismatch"],
            row["world_code"],
            row["backbone_shared_score"],
            row["portability_fraction"],
            1.0,
        ])
        pred = float(np.clip(features @ weights, 0.0, 1.0)) >= 0.5
        baseline = row["seam_fragility"] <= 0.3
        label = bool(row["patchable_label"])
        learned_hits += int(pred == label)
        baseline_hits += int(baseline == label)
    summary = {
        "n_rows": len(rows),
        "baseline_accuracy": baseline_hits / max(len(test), 1),
        "learned_accuracy": learned_hits / max(len(test), 1),
        "learned_beats_baseline": learned_hits >= baseline_hits,
    }
    (output_dir / "learned_repairability.json").write_text(json.dumps(summary, indent=2))
    return summary
