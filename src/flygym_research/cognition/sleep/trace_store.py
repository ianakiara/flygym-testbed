from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .trace_schema import ScoreComponent, SleepArtifact, SleepCandidate, TraceEpisode


def _coerce_score_component(value: Any) -> ScoreComponent:
    if type(value) is bool:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported score component type: {type(value).__name__}")


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

    def load_sleep_artifact(
        self,
        filename: str = "sleep_artifact.json",
    ) -> SleepArtifact:
        payload = json.loads((self.root / filename).read_text())
        candidates = [
            SleepCandidate(
                candidate_id=c["candidate_id"],
                representative_episode_id=c["representative_episode_id"],
                member_episode_ids=c["member_episode_ids"],
                evidence=c["evidence"],
                score_components={
                    str(k): _coerce_score_component(v)
                    for k, v in c["score_components"].items()
                },
                residual_episode_ids=c.get("residual_episode_ids", []),
                redundancy_tier=c.get("redundancy_tier", "local"),
                portability_evidence=c.get("portability_evidence", {}),
                functional_utility={str(k): float(v) for k, v in c.get("functional_utility", {}).items()},
                decision=c.get("decision", "review"),
                retained_exception_rationale=c.get("retained_exception_rationale", {}),
            )
            for c in payload["candidates"]
        ]
        return SleepArtifact(
            artifact_id=payload["artifact_id"],
            trace_bank_path=payload.get("trace_bank_path"),
            candidates=candidates,
            compressed_episode_ids=payload["compressed_episode_ids"],
            residual_episode_ids=payload["residual_episode_ids"],
            validation=payload.get("validation", {}),
            reports=payload.get("reports", {}),
            metadata=payload.get("metadata", {}),
        )
