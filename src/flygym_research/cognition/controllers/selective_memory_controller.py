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
    temperature_floor: float = 0.35
    temperature_scale: float = 1.5
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, observation: BrainObservation) -> DescendingCommand:
        self._step_count += 1
        query = self._build_query(observation)
        read_vec, attention = self._read(query)
        self._write(query, observation, attention)

        context_key = float(observation.world.observables.get("context_key", 0.0))
        stability = float(observation.summary.features.get("stability", 0.0))
        speed_mod = float(np.clip(0.25 + 0.3 * float(np.max(self._strengths)), -1.0, 1.0))

        is_query_mode = context_key < -0.1

        if is_query_mode and float(np.max(self._strengths)) > 0.05:
            # Selective recall: winner-take-all retrieval.
            # Weighted-average read contaminates the cue value from the best slot
            # with cue values from lower-ranked slots; a hard argmax on the
            # attention-weighted strengths eliminates cross-slot bleed.
            _, attention = self._read(query)
            top_idx = int(np.argmax(attention * self._strengths))
            cue_dims = self._slots[top_idx, :2]
            recalled = np.tanh(cue_dims)
            move_intent = float(np.clip(recalled[0], -1.0, 1.0))
            turn_intent = float(np.clip(recalled[1], -1.0, 1.0))
        else:
            target_vector = np.asarray(
                observation.world.observables.get("target_vector", np.zeros(2)),
                dtype=np.float64,
            )
            cue_vector = np.asarray(
                observation.world.observables.get("cue_vector", np.zeros(2)),
                dtype=np.float64,
            )
            recall_bias = np.tanh(read_vec[:2]) if read_vec.size >= 2 else np.zeros(2)
            distractor_flag = float(observation.world.info.get("distractor_active", False))
            focus = float(np.clip(1.0 - self.distractor_suppression * distractor_flag, 0.0, 1.0))
            blended = focus * target_vector + (1.0 - focus) * (cue_vector + recall_bias)
            move_intent = float(np.clip(blended[0], -1.0, 1.0))
            turn_intent = float(np.clip(blended[1], -1.0, 1.0))

        return DescendingCommand(
            move_intent=move_intent,
            turn_intent=turn_intent,
            speed_modulation=speed_mod,
            stabilization_priority=float(np.clip(stability + 0.25, 0.0, 1.0)),
            target_bias=(move_intent, turn_intent),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_query(self, observation: BrainObservation) -> np.ndarray:
        """Build an 8-dim slot query.

        Slot layout (dim 0-7):
          0-1  : cue value (written during write phase; zeros during query phase)
          2-4  : one-hot slot identity for cues A/B/C  (abs(context_key) ∈ {1,2,3})
          5    : stability
          6    : phase
          7    : visited count

        Using one-hot identity (dims 2-4) means that the dot-product similarity
        between a query for cue A and a stored slot for cue A is 1.0, while the
        similarity with B or C is 0.0, giving unambiguous retrieval even with
        cosine-normalised scoring.
        """
        features = observation.summary.features
        cue_vector = np.asarray(
            observation.world.observables.get("cue_vector", np.zeros(2)),
            dtype=np.float64,
        )
        context_key = float(observation.world.observables.get("context_key", 0.0))
        visited = float(sum(observation.world.observables.get("visited", [])))

        is_query_mode = context_key < -0.1
        slot_id = abs(context_key)  # 0 = none, 1 = A, 2 = B, 3 = C

        # One-hot slot identity
        slot_oh = np.zeros(3, dtype=np.float64)
        slot_index = self._cue_slot_index(slot_id)
        if slot_index is not None:
            slot_oh[slot_index] = 1.0

        # Cue value dims: store during write phase, zeros during query phase
        if is_query_mode:
            cue_dims = np.zeros(2, dtype=np.float64)
        else:
            cue_dims = np.array(
                [cue_vector[0] if cue_vector.size > 0 else 0.0,
                 cue_vector[1] if cue_vector.size > 1 else 0.0],
                dtype=np.float64,
            )

        base = np.array(
            [
                cue_dims[0],
                cue_dims[1],
                slot_oh[0],  # A
                slot_oh[1],  # B
                slot_oh[2],  # C
                float(features.get("stability", 0.0)),
                float(features.get("phase", 0.0)),
                visited,
            ],
            dtype=np.float64,
        )
        self._last_query = base
        return base

    def _read(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if float(self._strengths.max()) <= 1e-8:
            return (
                np.zeros(self.slot_dim, dtype=np.float64),
                np.zeros(self.memory_slots, dtype=np.float64),
            )
        query_norm = float(np.linalg.norm(query))
        slot_norms = np.linalg.norm(self._slots, axis=1)
        denom = np.maximum(slot_norms * max(query_norm, 1e-8), 1e-8)
        logits = (self._slots @ query) / denom
        # Sharpen temperature when memory is concentrated (low strength std)
        temperature = max(
            self.temperature_floor,
            1.0 - self.temperature_scale * float(np.std(self._strengths)),
        )
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        weights = np.exp(logits) * np.maximum(self._strengths, 1e-6)
        total = float(weights.sum())
        if total <= 1e-8:
            return (
                np.zeros(self.slot_dim, dtype=np.float64),
                np.zeros(self.memory_slots, dtype=np.float64),
            )
        attention = weights / total
        return attention @ self._slots, attention

    def _cue_slot_index(self, slot_id: float) -> int | None:
        cue_id = int(np.rint(slot_id))
        if not np.isclose(slot_id, cue_id):
            return None
        slot_index = cue_id - 1
        if 0 <= slot_index < min(3, self.memory_slots):
            return slot_index
        return None

    def _write(
        self, query: np.ndarray, observation: BrainObservation, attention: np.ndarray
    ) -> None:
        """Write query into the appropriate slot.

        Rules:
        • Query mode (context_key < 0): no write — we are reading only.
        • Known cue phase (context_key ∈ {1,2,3}): hard-write to the corresponding
          slot with full strength.  Using a direct overwrite (not a leaky gate)
          preserves the full cue value so that retrieval produces a clean ±1 signal.
        • Distractor phase / unknown phase: suppress write.
        """
        context_key = float(observation.world.observables.get("context_key", 0.0))
        is_query_mode = context_key < -0.1

        if is_query_mode:
            return  # read-only in query phase

        distractor_flag = float(observation.world.info.get("distractor_active", False))
        slot_id = abs(context_key)

        slot_index = self._cue_slot_index(slot_id)
        if slot_index is not None:
            # Known cue slot — hard write, no global strength decay.
            # Using a direct overwrite preserves the full cue value; gate-based
            # averaging would scale cue_dims to (1-decay^N)×cue, losing amplitude.
            self._slots[slot_index] = query.copy()
            self._strengths[slot_index] = 1.0
        elif not distractor_flag:
            # Untagged non-distractor step: lightweight leaky write to spare slot
            spare_start = min(3, self.memory_slots)
            if spare_start >= self.memory_slots:
                return
            slot_index = spare_start + int(np.argmin(self._strengths[spare_start:]))
            self._slots[slot_index] = (
                self.gate_decay * self._slots[slot_index]
                + (1.0 - self.gate_decay) * query
            )
            self._strengths[slot_index] = float(
                np.clip(self._strengths[slot_index] + 0.3, 0.0, 1.0)
            )
        # else: distractor phase — suppress write entirely

    def get_internal_state(self) -> dict[str, float]:
        return {
            "memory_slots": float(self.memory_slots),
            "active_slots": float(np.sum(self._strengths > 0.1)),
            "max_slot_strength": float(np.max(self._strengths)) if self._strengths.size else 0.0,
            "mean_slot_strength": float(np.mean(self._strengths)) if self._strengths.size else 0.0,
            "query_norm": float(np.linalg.norm(self._last_query)),
            "step_count": float(self._step_count),
        }
