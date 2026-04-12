from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trace_schema import SleepArtifact, TraceEpisode


class TraceStore:
    """Simple JSON-backed storage for sleep traces and artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_trace_bank(
        self,
        episodes: list[TraceEpisode],
        filename: str = "trace_bank.json",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        path = self.root / filename
        payload = {
            "metadata": metadata or {},
            "episodes": [episode.to_dict() for episode in episodes],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load_trace_bank(self, filename: str = "trace_bank.json") -> list[TraceEpisode]:
        payload = json.loads((self.root / filename).read_text())
        return [TraceEpisode.from_dict(item) for item in payload["episodes"]]

    def save_sleep_artifact(
        self,
        artifact: SleepArtifact,
        filename: str = "sleep_artifact.json",
    ) -> Path:
        path = self.root / filename
        path.write_text(json.dumps(artifact.to_dict(), indent=2))
        return path
