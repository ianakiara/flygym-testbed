from __future__ import annotations

import json
from pathlib import Path

from ..sleep import TraceStore, build_alignment_registry
from .exp_sleep_trace_compressor import collect_trace_bank



def run_experiment(
    output_dir: str | Path = "results/exp_interop_alignment",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = collect_trace_bank(max_steps=32)
    store = TraceStore(output_dir)
    store.save_trace_bank(
        episodes,
        filename="trace_bank.json",
        metadata={"experiment": "exp_interop_alignment"},
    )
    registry = build_alignment_registry(episodes)
    (output_dir / "alignment_registry.json").write_text(json.dumps(registry, indent=2))
    return registry
