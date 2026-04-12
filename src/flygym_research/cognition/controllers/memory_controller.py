"""Memory-augmented controller — uses an explicit memory buffer with
recurrent-style gating to maintain persistent internal state.

This controller goes beyond the simple scalar ``memory_trace`` in
:class:`ReducedDescendingController` by maintaining a fixed-size memory
buffer of past observations and using exponential gating to produce
context-dependent actions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..interfaces import BrainInterface, BrainObservation, DescendingCommand


@dataclass(slots=True)
class MemoryController(BrainInterface):
    """Controller with explicit memory buffer and recurrent hidden state.

    Parameters
    ----------
    memory_size : int
        Number of past observation summaries to retain.
    hidden_dim : int
        Dimensionality of the recurrent hidden state vector.
    gate_decay : float
        Exponential decay rate for the gating mechanism (0–1).
    """

    memory_size: int = 16
    hidden_dim: int = 8
    gate_decay: float = 0.85
    _memory: deque[np.ndarray] = field(init=False)
    _hidden: np.ndarray = field(init=False)
    _step_count: int = field(default=0, init=False)
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._memory = deque(maxlen=self.memory_size)
        self._hidden = np.zeros(self.hidden_dim, dtype=np.float64)
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self._memory.clear()
        self._hidden = np.zeros(self.hidden_dim, dtype=np.float64)
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

    def act(self, observation: BrainObservation) -> DescendingCommand:
        self._step_count += 1

        # Extract current observation features as a vector.
        features = observation.summary.features
        current_vec = np.array(
            [features.get(k, 0.0) for k in sorted(features)], dtype=np.float64
        )
        self._memory.append(current_vec)

        # Recurrent hidden state update: gated combination of previous
        # hidden state and current input projection.
        input_proj = np.zeros(self.hidden_dim, dtype=np.float64)
        n = min(len(current_vec), self.hidden_dim)
        input_proj[:n] = current_vec[:n]
        gate = self.gate_decay
        self._hidden = gate * self._hidden + (1.0 - gate) * np.tanh(input_proj)

        # Compute memory context: weighted average of memory buffer entries
        # with more recent entries weighted higher.
        if len(self._memory) > 1:
            weights = np.array(
                [i + 1 for i in range(len(self._memory))], dtype=np.float64
            )
            weights /= weights.sum()
            memory_vecs = np.array(list(self._memory), dtype=np.float64)
            # Truncate/pad to uniform length.
            min_len = min(v.shape[0] for v in self._memory)
            memory_context = np.average(
                memory_vecs[:, :min_len], axis=0, weights=weights
            )
        else:
            memory_context = current_vec.copy()

        # Target-directed action using both current observation and memory.
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        stability = features.get("stability", 0.0)

        # Memory-modulated move intent: hidden state biases action strength.
        hidden_bias = float(np.mean(self._hidden))
        memory_signal = float(np.mean(memory_context[:2])) if len(memory_context) >= 2 else 0.0

        move_intent = float(np.clip(target_vector[0] + 0.1 * hidden_bias, -1.0, 1.0))
        turn_intent = float(np.clip(target_vector[1] + 0.1 * memory_signal, -1.0, 1.0))
        speed_mod = float(np.clip(0.3 + 0.2 * hidden_bias, -1.0, 1.0))

        return DescendingCommand(
            move_intent=move_intent,
            turn_intent=turn_intent,
            speed_modulation=speed_mod,
            stabilization_priority=float(np.clip(stability + 0.2, 0.0, 1.0)),
            target_bias=(float(target_vector[0]), float(target_vector[1])),
        )

    def get_internal_state(self) -> dict[str, float]:
        return {
            "hidden_mean": float(np.mean(self._hidden)),
            "hidden_norm": float(np.linalg.norm(self._hidden)),
            "memory_length": float(len(self._memory)),
            "step_count": float(self._step_count),
            "gate_decay": self.gate_decay,
        }
