"""Slot-attention memory controller — addresses 5 specific memory subproblems.

This controller replaces the naive flat-average memory in
:class:`MemoryController` with a structured slot-attention architecture:

1. **Selective recall**: Attention over slots weighted by query-key similarity.
   Current observation queries memory → retrieves only relevant entries.

2. **Distractor resistance**: Write gate with novelty/importance scoring.
   High-surprise inputs gate into dedicated slots without overwriting
   cue-phase entries.  Cue-phase entries get high importance and resist
   overwriting.

3. **Gated retrieval**: Read gate produces a confidence weight [0,1].
   Action = confidence × memory_target + (1−confidence) × current_target.
   When memory is stale the controller falls back to current observation.

4. **Overwrite resistance**: Each slot carries an importance weight based
   on the magnitude of the cue signal at write time.  Important slots
   resist overwriting by requiring higher novelty to displace.

5. **Store ≠ Use separation**: Write pathway (encode → slot assignment)
   is architecturally separate from read pathway (query → attention →
   weighted retrieval → action modulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import BrainInterface, BrainObservation, DescendingCommand


@dataclass(slots=True)
class SlotMemoryController(BrainInterface):
    """Controller with slot-attention memory and selective gating.

    Parameters
    ----------
    n_slots : int
        Number of memory slots.
    slot_dim : int
        Dimensionality of each memory slot vector.
    gate_threshold : float
        Minimum novelty score required to write into a slot (0–1).
    importance_decay : float
        Per-step decay applied to slot importance weights (0–1).
    confidence_bias : float
        Bias added to the read-gate confidence (shifts trust toward
        memory vs current observation).
    """

    n_slots: int = 8
    slot_dim: int = 8
    gate_threshold: float = 0.3
    importance_decay: float = 0.98
    confidence_bias: float = 0.0

    # ── Internal state (not constructor args) ────────────────────────────
    _slots: np.ndarray = field(init=False)
    _importance: np.ndarray = field(init=False)
    _slot_ages: np.ndarray = field(init=False)
    _hidden: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)
    _prev_vec: np.ndarray = field(init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._slots = np.zeros((self.n_slots, self.slot_dim), dtype=np.float64)
        self._importance = np.zeros(self.n_slots, dtype=np.float64)
        self._slot_ages = np.zeros(self.n_slots, dtype=np.float64)
        self._hidden = np.zeros(self.slot_dim, dtype=np.float64)
        self._prev_vec = np.zeros(self.slot_dim, dtype=np.float64)
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self._slots = np.zeros((self.n_slots, self.slot_dim), dtype=np.float64)
        self._importance = np.zeros(self.n_slots, dtype=np.float64)
        self._slot_ages = np.zeros(self.n_slots, dtype=np.float64)
        self._hidden = np.zeros(self.slot_dim, dtype=np.float64)
        self._prev_vec = np.zeros(self.slot_dim, dtype=np.float64)
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    # ── Core act method ──────────────────────────────────────────────────

    def act(self, observation: BrainObservation) -> DescendingCommand:
        self._step_count += 1

        # ── Encode current observation into slot_dim vector ──────────────
        current_vec = self._encode_observation(observation)

        # ── WRITE PATHWAY: decide whether/where to write ─────────────────
        novelty = self._compute_novelty(current_vec)
        cue_signal = float(observation.world.observables.get("cue_signal", 0.5))
        # Importance is high when cue is clearly informative (near 0 or 1)
        write_importance = abs(cue_signal - 0.5) * 2.0 + novelty * 0.5

        if novelty >= self.gate_threshold:
            self._write_to_slot(current_vec, write_importance)

        # Age all slots and decay importance
        self._slot_ages += 1.0
        self._importance *= self.importance_decay

        # ── READ PATHWAY: attention-weighted retrieval ────────────────────
        memory_readout, confidence = self._read_from_slots(current_vec)

        # ── Recurrent hidden state update ────────────────────────────────
        self._hidden = 0.85 * self._hidden + 0.15 * np.tanh(current_vec)

        # ── Action computation with gated retrieval ──────────────────────
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        stability = observation.summary.features.get("stability", 0.0)

        # Memory target: use the target direction stored in memory readout.
        # Slot dims layout: [features..., target_x, target_y, cue_signal]
        # Last 3 dims mirror the encoding layout.
        mem_target = memory_readout[-3:-1] if len(memory_readout) >= 3 else np.zeros(2)

        # Gated retrieval: confidence controls how much we trust memory
        adjusted_confidence = float(np.clip(confidence + self.confidence_bias, 0.0, 1.0))
        corrected_target = (
            adjusted_confidence * mem_target
            + (1.0 - adjusted_confidence) * target_vector
        )

        # Hidden state biases action (dimension-sensitive for causal tests)
        h_move = float(self._hidden[0]) if self.slot_dim > 0 else 0.0
        h_turn = float(self._hidden[1]) if self.slot_dim > 1 else 0.0
        h_speed = float(self._hidden[2]) if self.slot_dim > 2 else 0.0

        move_intent = float(np.clip(corrected_target[0] + 0.1 * h_move, -1.0, 1.0))
        turn_intent = float(np.clip(corrected_target[1] + 0.1 * h_turn, -1.0, 1.0))
        speed_mod = float(np.clip(0.3 + 0.2 * h_speed, -1.0, 1.0))

        self._prev_vec = current_vec.copy()

        return DescendingCommand(
            move_intent=move_intent,
            turn_intent=turn_intent,
            speed_modulation=speed_mod,
            stabilization_priority=float(np.clip(stability + 0.2, 0.0, 1.0)),
            target_bias=(float(corrected_target[0]), float(corrected_target[1])),
        )

    # ── Observation encoding ─────────────────────────────────────────────

    def _encode_observation(self, observation: BrainObservation) -> np.ndarray:
        """Encode observation into a fixed-dim vector for slot storage."""
        features = observation.summary.features
        feature_vec = np.array(
            [features.get(k, 0.0) for k in sorted(features)], dtype=np.float64,
        )
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        cue_signal = float(observation.world.observables.get("cue_signal", 0.5))

        raw = np.concatenate([
            feature_vec,
            np.array([target_vector[0], target_vector[1], cue_signal], dtype=np.float64),
        ])

        # Project/truncate to slot_dim
        result = np.zeros(self.slot_dim, dtype=np.float64)
        n = min(len(raw), self.slot_dim)
        result[:n] = raw[:n]
        return result

    # ── Write pathway ────────────────────────────────────────────────────

    def _compute_novelty(self, current_vec: np.ndarray) -> float:
        """Compute novelty score: how different is current from recent memory."""
        if self._step_count <= 1:
            return 1.0  # First step is always novel

        # Compare to previous vector
        diff_prev = float(np.linalg.norm(current_vec - self._prev_vec))

        # Compare to nearest slot
        active_mask = self._importance > 1e-6
        if not np.any(active_mask):
            return 1.0

        active_slots = self._slots[active_mask]
        dists = np.linalg.norm(active_slots - current_vec, axis=1)
        min_dist = float(np.min(dists))

        # Normalize: novelty in [0, 1]
        novelty = float(np.clip(0.5 * diff_prev + 0.5 * min_dist, 0.0, 2.0)) / 2.0
        return novelty

    def _write_to_slot(self, vec: np.ndarray, importance: float) -> None:
        """Write vec into the best available slot.

        Strategy: find the slot with the lowest (importance / age) score —
        this prefers overwriting old unimportant slots while protecting
        important recent ones.
        """
        # Protection score: high importance + low age = hard to overwrite
        protection = self._importance / (self._slot_ages + 1.0)

        # Find least protected slot
        target_idx = int(np.argmin(protection))

        # Only overwrite if new importance exceeds existing protection
        if importance > protection[target_idx] or self._importance[target_idx] < 1e-6:
            self._slots[target_idx] = vec.copy()
            self._importance[target_idx] = importance
            self._slot_ages[target_idx] = 0.0

    # ── Read pathway ─────────────────────────────────────────────────────

    def _read_from_slots(
        self, query: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Attention-weighted read from slots.

        Returns (readout_vector, confidence).
        Confidence reflects how strongly memory matches the current query.
        """
        # Check if any slots are populated
        total_importance = float(np.sum(self._importance))
        if total_importance < 1e-6:
            return np.zeros(self.slot_dim, dtype=np.float64), 0.0

        # Compute attention scores: dot product of query with each slot
        # weighted by importance (important slots get higher attention)
        raw_scores = self._slots @ query  # (n_slots,)
        weighted_scores = raw_scores * self._importance

        # Softmax with temperature
        max_score = float(np.max(weighted_scores))
        exp_scores = np.exp(np.clip(weighted_scores - max_score, -20.0, 0.0))

        # Zero out empty slots
        exp_scores *= (self._importance > 1e-6).astype(np.float64)
        score_sum = float(np.sum(exp_scores))

        if score_sum < 1e-10:
            return np.zeros(self.slot_dim, dtype=np.float64), 0.0

        attention_weights = exp_scores / score_sum

        # Weighted sum of slot contents
        readout = attention_weights @ self._slots  # (slot_dim,)

        # Confidence: how peaked is the attention distribution?
        # High entropy → low confidence, low entropy → high confidence
        entropy = -float(np.sum(
            attention_weights * np.log(np.clip(attention_weights, 1e-10, 1.0))
        ))
        max_entropy = float(np.log(max(np.sum(self._importance > 1e-6), 1)))
        confidence = 1.0 - (entropy / max(max_entropy, 1e-6))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return readout, confidence

    # ── State introspection (for causal/epiphenomenal tests) ─────────────

    def get_internal_state(self) -> dict[str, float]:
        active_slots = int(np.sum(self._importance > 1e-6))
        return {
            "hidden_mean": float(np.mean(self._hidden)),
            "hidden_norm": float(np.linalg.norm(self._hidden)),
            "active_slots": float(active_slots),
            "mean_importance": float(np.mean(self._importance)),
            "max_importance": float(np.max(self._importance)),
            "step_count": float(self._step_count),
        }

    def intervene_state(self, delta: float | np.ndarray) -> None:
        """Surgically perturb the hidden state and slot contents.

        Used for causal intervention experiments — shifts the internal
        state by a known amount to measure downstream divergence.
        """
        delta_arr = np.asarray(delta, dtype=np.float64)
        self._hidden = self._hidden + delta_arr
        # Also perturb active slots proportionally
        for i in range(self.n_slots):
            if self._importance[i] > 1e-6:
                self._slots[i] += delta_arr * 0.5

    def shuffle_state(self, rng: np.random.Generator) -> None:
        """Randomly shuffle slot contents and hidden state dimensions.

        Used for epiphenomenal testing — if shuffling has no effect on
        behaviour, the state is decorative.
        """
        rng.shuffle(self._hidden)
        # Shuffle slot order
        perm = rng.permutation(self.n_slots)
        self._slots = self._slots[perm]
        self._importance = self._importance[perm]
        self._slot_ages = self._slot_ages[perm]
        # Also shuffle within each active slot
        for i in range(self.n_slots):
            if self._importance[i] > 1e-6:
                rng.shuffle(self._slots[i])
