from __future__ import annotations

import json
from pathlib import Path

from ..sleep import TraceStore, build_memory_packets, compress_trace_bank
from .exp_sleep_trace_compressor import collect_trace_bank



def run_experiment(
    output_dir: str | Path = "results/exp_memory_closure",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=32)
    store = TraceStore(output_dir)
    trace_path = store.save_trace_bank(
        episodes,
        filename="trace_bank.json",
        metadata={"experiment": "exp_memory_closure"},
    )
    artifact = compress_trace_bank(episodes, trace_bank_path=str(trace_path))
    packets = build_memory_packets(episodes, artifact)
    (output_dir / "memory_packets.json").write_text(json.dumps(packets, indent=2))
    return packets
