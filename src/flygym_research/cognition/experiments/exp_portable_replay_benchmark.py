from __future__ import annotations

import json
from pathlib import Path

from ..sleep import CompressionConfig, TraceStore, benchmark_portable_replay, compress_trace_bank
from .exp_sleep_trace_compressor import collect_trace_bank


def run_experiment(
    output_dir: str | Path = "results/exp_portable_replay_benchmark",
    *,
    config: CompressionConfig | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=32)
    store = TraceStore(output_dir)
    trace_path = store.save_trace_bank(
        episodes,
        filename="trace_bank.json",
        metadata={"experiment": "exp_portable_replay_benchmark"},
    )
    artifact = compress_trace_bank(episodes, trace_bank_path=str(trace_path), config=config)
    replay = benchmark_portable_replay(episodes, artifact)
    store.save_sleep_artifact(artifact, filename="sleep_artifact.json")
    summary = {"artifact": artifact.to_dict(), "portable_replay": replay}
    (output_dir / "portable_replay_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
