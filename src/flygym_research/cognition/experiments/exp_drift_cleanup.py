from __future__ import annotations

import json
from pathlib import Path

from ..sleep import cleanup_memory_bank, compress_trace_bank
from .exp_sleep_trace_compressor import collect_trace_bank



def run_experiment(
    output_dir: str | Path = "results/exp_drift_cleanup",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=24)
    artifact_a = compress_trace_bank(episodes, trace_bank_path=str(output_dir / "trace_a.json"))
    artifact_b = compress_trace_bank(episodes, trace_bank_path=str(output_dir / "trace_b.json"))
    artifact_a.metadata.update({"source": "daily_trace_bank", "age_days": 1.0, "reuse_count": 5.0})
    artifact_b.metadata.update({"source": "daily_trace_bank", "age_days": 30.0, "reuse_count": 0.0})
    report = cleanup_memory_bank([artifact_a, artifact_b])
    (output_dir / "drift_cleanup_report.json").write_text(json.dumps(report, indent=2))
    return report
