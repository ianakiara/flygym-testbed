from __future__ import annotations

import json
from pathlib import Path

from ..sleep import TraceStore, analyze_seam_failures
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
    report = analyze_seam_failures(episodes)
    (output_dir / "seam_report.json").write_text(json.dumps(report, indent=2))
    return report
