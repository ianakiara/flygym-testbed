"""Slot-attention memory controller — addresses 5 specific memory subproblems.

This controller replaces the naive flat-average memory in
:class:`MemoryController` with a structured slot-attention architecture:

1. **Selective recall**: Attention over slots weighted by query-key similarity
   *and* phase reliability.  Current observation queries memory → retrieves
   only relevant, trustworthy entries.

2. **Distractor resistance**: Write gate with novelty/importance scoring
   *plus* phase tagging.  Distractor-phase writes are tagged so they can be
   down-weighted during reads.  Cue-phase entries get high importance and
   resist overwriting.

3. **Gated retrieval**: Read gate produces a confidence weight [0,1] that
   accounts for both attention peakedness and phase reliability of retrieved
   slots.  Action = confidence × memory_target + (1−confidence) ×
   current_target.

4. **Overwrite resistance**: Each slot carries an importance weight based
   on the cue signal magnitude at write time.  Cue-phase slots have a
   large protection multiplier that makes them very hard to displace.

5. **Store ≠ Use separation**: Write pathway (encode → phase tag → slot
   assignment) is architecturally separate from read pathway (query →
   phase-aware attention → weighted retrieval → action modulation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..interfaces import BrainInterface, BrainObservation, DescendingCommand


# Phase tags for memory slots
_PHASE_EMPTY: int = 0
_PHASE_CUE: int = 1
_PHASE_DISTRACTOR: int = 2
_PHASE_AMBIGUOUS: int = 3


def _classify_phase(cue_signal: float, distractor_active: bool) -> int:
    """Classify the current observation phase for slot tagging."""
    if distractor_active:
        return _PHASE_DISTRACTOR
    if abs(cue_signal - 0.5) > 0.3:
        # Strong cue signal (near 0 or 1) → informative cue phase
        return _PHASE_CUE
    return _PHASE_AMBIGUOUS


@dataclass(slots=True)
class SlotMemoryController(BrainInterface):
    """Controller with phase-aware slot-attention memory and selective gating.

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
    cue_protection_multiplier : float
        Extra protection factor for cue-phase slots.  A cue-phase slot's
        protection score is multiplied by this factor, making it very
        hard to overwrite.
    distractor_read_penalty : float
        Multiplicative penalty applied to attention scores of
        distractor-tagged slots during read (0–1).  Lower = more
        suppression of distractor memory.
    """

    n_slots: int = 8
    slot_dim: int = 8
    gate_threshold: float = 0.3
    importance_decay: float = 0.98
    confidence_bias: float = 0.0
    cue_protection_multiplier: float = 3.0
    distractor_read_penalty: float = 0.15

    # ── Internal state (not constructor args) ────────────────────────────
    _slots: np.ndarray = field(init=False)
    _importance: np.ndarray = field(init=False)
    _slot_ages: np.ndarray = field(init=False)
    _phase_tags: np.ndarray = field(init=False)  # per-slot phase tag
    _hidden: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)
    _prev_vec: np.ndarray = field(init=False)
    _cue_memory: np.ndarray = field(init=False)  # dedicated cue snapshot
    _cue_confidence: float = field(default=0.0, init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._slots = np.zeros((self.n_slots, self.slot_dim), dtype=np.float64)
        self._importance = np.zeros(self.n_slots, dtype=np.float64)
        self._slot_ages = np.zeros(self.n_slots, dtype=np.float64)
        self._phase_tags = np.full(self.n_slots, _PHASE_EMPTY, dtype=np.int32)
        self._hidden = np.zeros(self.slot_dim, dtype=np.float64)
        self._prev_vec = np.zeros(self.slot_dim, dtype=np.float64)
        self._cue_memory = np.zeros(self.slot_dim, dtype=np.float64)
        self._cue_confidence = 0.0
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self._slots = np.zeros((self.n_slots, self.slot_dim), dtype=np.float64)
        self._importance = np.zeros(self.n_slots, dtype=np.float64)
        self._slot_ages = np.zeros(self.n_slots, dtype=np.float64)
        self._phase_tags = np.full(self.n_slots, _PHASE_EMPTY, dtype=np.int32)
        self._hidden = np.zeros(self.slot_dim, dtype=np.float64)
        self._prev_vec = np.zeros(self.slot_dim, dtype=np.float64)
        self._cue_memory = np.zeros(self.slot_dim, dtype=np.float64)
        self._cue_confidence = 0.0
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    # ── Core act method ──────────────────────────────────────────────────

    def act(self, observation: BrainObservation) -> DescendingCommand:
        self._step_count += 1

        # ── Encode current observation into slot_dim vector ──────────────
        current_vec = self._encode_observation(observation)

        # ── Determine current phase ──────────────────────────────────────
        cue_signal = float(observation.world.observables.get("cue_signal", 0.5))
        distractor_active = bool(
            observation.world.observables.get("distractor_active", False)
        )
        current_phase = _classify_phase(cue_signal, distractor_active)

        # ── WRITE PATHWAY: phase-aware slot writing ──────────────────────
        novelty = self._compute_novelty(current_vec)
        # Importance is high when cue is clearly informative (near 0 or 1)
        cue_informativeness = abs(cue_signal - 0.5) * 2.0
        write_importance = cue_informativeness + novelty * 0.5

        # Cue-phase entries get a large importance boost
        if current_phase == _PHASE_CUE:
            write_importance *= 2.0
            # Snapshot the cue encoding for direct recall later
            self._cue_memory = current_vec.copy()
            self._cue_confidence = min(1.0, cue_informativeness + 0.3)
        elif current_phase == _PHASE_DISTRACTOR:
            # Distractor-phase: reduce write importance to discourage
            # overwriting cue slots, but still allow writing to empty slots
            write_importance *= 0.3

        if novelty >= self.gate_threshold:
            self._write_to_slot(current_vec, write_importance, current_phase)

        # Age all slots and decay importance
        self._slot_ages += 1.0
        self._importance *= self.importance_decay

        # ── READ PATHWAY: phase-aware attention-weighted retrieval ────────
        memory_readout, read_confidence = self._read_from_slots(current_vec)

        # ── Cue snapshot fallback ────────────────────────────────────────
        # During ambiguous phase, if we have a strong cue snapshot, prefer
        # it over the general slot readout (which may be corrupted by
        # distractor entries).
        if current_phase == _PHASE_AMBIGUOUS and self._cue_confidence > 0.3:
            # Blend: cue snapshot weighted by cue_confidence
            cue_weight = self._cue_confidence
            memory_readout = (
                cue_weight * self._cue_memory
                + (1.0 - cue_weight) * memory_readout
            )
            read_confidence = max(read_confidence, self._cue_confidence * 0.8)

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
        mem_target = memory_readout[-3:-1] if len(memory_readout) >= 3 else np.zeros(2)

        # ── Phase-aware confidence gating ────────────────────────────────
        # During cue phase: trust current observation (cue is visible)
        # During distractor phase: trust memory MORE (current input is misleading)
        # During ambiguous phase: trust memory if confident
        if current_phase == _PHASE_CUE:
            # Cue is visible — use current observation primarily
            effective_confidence = float(np.clip(
                read_confidence * 0.3 + self.confidence_bias, 0.0, 0.4
            ))
        elif current_phase == _PHASE_DISTRACTOR:
            # Distractor is active — current target_vector is WRONG.
            # Trust memory strongly (the cue-phase entries).
            effective_confidence = float(np.clip(
                read_confidence * 0.9 + 0.3 + self.confidence_bias, 0.5, 1.0
            ))
        else:
            # Ambiguous phase — use memory confidence as-is
            effective_confidence = float(np.clip(
                read_confidence + self.confidence_bias, 0.0, 1.0
            ))

        corrected_target = (
            effective_confidence * mem_target
            + (1.0 - effective_confidence) * target_vector
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

    def _write_to_slot(
        self, vec: np.ndarray, importance: float, phase: int
    ) -> None:
        """Write vec into the best available slot with phase tagging.

        Strategy: find the slot with the lowest protection score.
        Cue-phase slots get extra protection via cue_protection_multiplier.
        Distractor-phase slots get no extra protection, making them easy
        to displace.
        """
        # Protection score: high importance + low age = hard to overwrite
        protection = self._importance / (self._slot_ages + 1.0)

        # Cue-phase slots get extra protection
        for i in range(self.n_slots):
            if self._phase_tags[i] == _PHASE_CUE:
                protection[i] *= self.cue_protection_multiplier

        # Find least protected slot
        target_idx = int(np.argmin(protection))

        # Only overwrite if new importance exceeds existing protection
        if importance > protection[target_idx] or self._importance[target_idx] < 1e-6:
            self._slots[target_idx] = vec.copy()
            self._importance[target_idx] = importance
            self._slot_ages[target_idx] = 0.0
            self._phase_tags[target_idx] = phase

    # ── Read pathway ─────────────────────────────────────────────────────

    def _read_from_slots(
        self, query: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Phase-aware attention-weighted read from slots.

        Distractor-tagged slots get penalized attention weight.
        Cue-tagged slots get boosted attention weight.

        Returns (readout_vector, confidence).
        """
        # Check if any slots are populated
        total_importance = float(np.sum(self._importance))
        if total_importance < 1e-6:
            return np.zeros(self.slot_dim, dtype=np.float64), 0.0

        # Compute attention scores: dot product of query with each slot
        # weighted by importance
        raw_scores = self._slots @ query  # (n_slots,)
        weighted_scores = raw_scores * self._importance

        # ── Phase-aware attention modulation ──────────────────────────────
        # Penalize distractor-tagged slots, boost cue-tagged slots
        phase_modulation = np.ones(self.n_slots, dtype=np.float64)
        for i in range(self.n_slots):
            if self._phase_tags[i] == _PHASE_DISTRACTOR:
                phase_modulation[i] = self.distractor_read_penalty
            elif self._phase_tags[i] == _PHASE_CUE:
                phase_modulation[i] = 2.0  # boost cue slots

        weighted_scores *= phase_modulation

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

        # ── Confidence: combines attention peakedness + phase reliability ─
        # How peaked is the attention distribution?
        entropy = -float(np.sum(
            attention_weights * np.log(np.clip(attention_weights, 1e-10, 1.0))
        ))
        max_entropy = float(np.log(max(np.sum(self._importance > 1e-6), 1)))
        attention_confidence = 1.0 - (entropy / max(max_entropy, 1e-6))

        # Phase reliability: what fraction of attention goes to cue slots?
        cue_attention_mass = float(np.sum(
            attention_weights * (self._phase_tags == _PHASE_CUE).astype(np.float64)
        ))
        distractor_attention_mass = float(np.sum(
            attention_weights * (self._phase_tags == _PHASE_DISTRACTOR).astype(np.float64)
        ))
        # Reliability bonus: more cue attention → higher confidence
        phase_reliability = cue_attention_mass - distractor_attention_mass

        confidence = float(np.clip(
            0.5 * attention_confidence + 0.5 * phase_reliability, 0.0, 1.0
        ))

        return readout, confidence

    # ── State introspection (for causal/epiphenomenal tests) ─────────────

    def get_internal_state(self) -> dict[str, float]:
        active_slots = int(np.sum(self._importance > 1e-6))
        n_cue_slots = int(np.sum(
            (self._phase_tags == _PHASE_CUE)
            & (self._importance > 1e-6)
        ))
        n_distractor_slots = int(np.sum(
            (self._phase_tags == _PHASE_DISTRACTOR)
            & (self._importance > 1e-6)
        ))
        return {
            "hidden_mean": float(np.mean(self._hidden)),
            "hidden_norm": float(np.linalg.norm(self._hidden)),
            "active_slots": float(active_slots),
            "cue_slots": float(n_cue_slots),
            "distractor_slots": float(n_distractor_slots),
            "cue_confidence": self._cue_confidence,
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
        self._phase_tags = self._phase_tags[perm]
        # Also shuffle within each active slot
        for i in range(self.n_slots):
            if self._importance[i] > 1e-6:
                rng.shuffle(self._slots[i])
