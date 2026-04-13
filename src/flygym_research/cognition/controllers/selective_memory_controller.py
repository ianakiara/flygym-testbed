from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import BrainInterface, BrainObservation, DescendingCommand


@dataclass(slots=True)
class SelectiveMemoryController(BrainInterface):
    memory_slots: int = 4
    slot_dim: int = 8
    gate_decay: float = 0.9
    distractor_suppression: float = 0.35
    _slots: np.ndarray = field(init=False)
    _strengths: np.ndarray = field(init=False)
    _last_query: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._slots = np.zeros((self.memory_slots, self.slot_dim), dtype=np.float64)
        self._strengths = np.zeros(self.memory_slots, dtype=np.float64)
        self._last_query = np.zeros(self.slot_dim, dtype=np.float64)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._slots.fill(0.0)
        self._strengths.fill(0.0)
        self._last_query.fill(0.0)
        self._step_count = 0

    def act(self, observation: BrainObservation) -> DescendingCommand:
        self._step_count += 1
        query = self._build_query(observation)
        read_vec, attention = self._read(query)
        self._write(query, observation, attention)

        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        cue_vector = np.asarray(
            observation.world.observables.get("cue_vector", np.zeros(2)),
            dtype=np.float64,
        )
        context_key = float(observation.world.observables.get("context_key", 0.0))
        recall_bias = np.tanh(read_vec[:2]) if read_vec.size >= 2 else np.zeros(2)
        distractor_flag = float(observation.world.info.get("distractor_active", False))
        focus = 1.0 - self.distractor_suppression * distractor_flag
        blended = focus * target_vector + (1.0 - focus) * (cue_vector + recall_bias)
        if cue_vector.any():
            blended = 0.6 * blended + 0.4 * cue_vector
        move_intent = float(np.clip(blended[0] + 0.15 * recall_bias[0], -1.0, 1.0))
        turn_intent = float(np.clip(blended[1] + 0.15 * recall_bias[1] + 0.1 * context_key, -1.0, 1.0))
        stability = float(observation.summary.features.get("stability", 0.0))
        speed_mod = float(np.clip(0.25 + 0.3 * np.max(self._strengths), -1.0, 1.0))
        return DescendingCommand(
            move_intent=move_intent,
            turn_intent=turn_intent,
            speed_modulation=speed_mod,
            stabilization_priority=float(np.clip(stability + 0.25, 0.0, 1.0)),
            target_bias=(float(blended[0]), float(blended[1])),
        )

    def _build_query(self, observation: BrainObservation) -> np.ndarray:
        features = observation.summary.features
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        cue_vector = np.asarray(
            observation.world.observables.get("cue_vector", np.zeros(2)),
            dtype=np.float64,
        )
        visited = float(sum(observation.world.observables.get("visited", [])))
        context_key = float(observation.world.observables.get("context_key", 0.0))
        base = np.array(
            [
                target_vector[0] if target_vector.size > 0 else 0.0,
                target_vector[1] if target_vector.size > 1 else 0.0,
                cue_vector[0] if cue_vector.size > 0 else 0.0,
                cue_vector[1] if cue_vector.size > 1 else 0.0,
                float(features.get("stability", 0.0)),
                float(features.get("phase", 0.0)),
                context_key,
                visited,
            ],
            dtype=np.float64,
        )
        self._last_query = base
        return base

    def _read(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._strengths.max() <= 1e-8:
            return np.zeros(self.slot_dim, dtype=np.float64), np.zeros(self.memory_slots, dtype=np.float64)
        logits = self._slots @ query
        logits = logits - logits.max()
        weights = np.exp(logits) * np.maximum(self._strengths, 1e-6)
        total = float(weights.sum())
        if total <= 1e-8:
            return np.zeros(self.slot_dim, dtype=np.float64), np.zeros(self.memory_slots, dtype=np.float64)
        attention = weights / total
        return attention @ self._slots, attention

    def _write(self, query: np.ndarray, observation: BrainObservation, attention: np.ndarray) -> None:
        cue_vector = np.asarray(
            observation.world.observables.get("cue_vector", np.zeros(2)), dtype=np.float64
        )
        context_key = float(observation.world.observables.get("context_key", 0.0))
        cue_strength = float(np.linalg.norm(cue_vector)) + abs(context_key)
        write_gate = max(cue_strength, 0.2)
        slot_index = self._select_slot(attention)
        self._slots[slot_index] = self.gate_decay * self._slots[slot_index] + (1.0 - self.gate_decay) * query
        self._strengths *= self.gate_decay
        self._strengths[slot_index] = float(np.clip(self._strengths[slot_index] + write_gate, 0.0, 1.0))


    def _select_slot(self, attention: np.ndarray) -> int:
        if attention.size == 0 or attention.max() < 0.35:
            return int(np.argmin(self._strengths))
        return int(np.argmax(attention))

    def get_internal_state(self) -> dict[str, float]:
        return {
            "memory_slots": float(self.memory_slots),
            "active_slots": float(np.sum(self._strengths > 0.1)),
            "max_slot_strength": float(np.max(self._strengths)) if self._strengths.size else 0.0,
            "mean_slot_strength": float(np.mean(self._strengths)) if self._strengths.size else 0.0,
            "query_norm": float(np.linalg.norm(self._last_query)),
            "step_count": float(self._step_count),
        }
