from __future__ import annotations

import json
from pathlib import Path

from ..sleep import TraceStore, analyze_seam_failures
from ..sleep.seam_repair import repairability_curve
from .exp_sleep_trace_compressor import collect_trace_bank



def run_experiment(
    output_dir: str | Path = "results/exp_seam_repair",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=32)
    store = TraceStore(output_dir)
    store.save_trace_bank(
        episodes,
        filename="trace_bank.json",
        metadata={"experiment": "exp_seam_repair"},
    )

    # Standard seam analysis at default threshold.
    report = analyze_seam_failures(episodes)

    # Repairability curve across a range of thresholds.
    curve_result = repairability_curve(
        episodes,
        thresholds=[0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
    )

    # Tightened analysis at threshold=0.10 (stricter than default 0.20).
    tight_report = analyze_seam_failures(episodes, seam_threshold=0.10)

    combined: dict[str, object] = {
        **report,
        "repairability_curve": curve_result,
        "tight_threshold_report": {
            "seam_threshold": 0.10,
            "n_failures": tight_report["n_failures"],
            "n_patchable_failures": tight_report["n_patchable_failures"],
            "repairability_score": tight_report.get("repairability_score", 0.0),
        },
    }
    (output_dir / "seam_report.json").write_text(json.dumps(combined, indent=2))
    return combined
