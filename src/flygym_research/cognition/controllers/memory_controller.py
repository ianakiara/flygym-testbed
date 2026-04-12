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
            # Pad/truncate all vectors to a common length so stacking is safe
            # even if the feature set changes between steps.
            target_len = len(current_vec)
            padded: list[np.ndarray] = []
            for v in self._memory:
                if len(v) == target_len:
                    padded.append(v)
                elif len(v) < target_len:
                    padded.append(np.pad(v, (0, target_len - len(v))))
                else:
                    padded.append(v[:target_len])
            memory_vecs = np.array(padded, dtype=np.float64)
            memory_context = np.average(memory_vecs, axis=0, weights=weights)
        else:
            memory_context = current_vec.copy()

        # Target-directed action using both current observation and memory.
        target_vector = np.asarray(
            observation.world.observables.get("target_vector", np.zeros(2)),
            dtype=np.float64,
        )
        stability = features.get("stability", 0.0)

        # Memory-modulated move intent: hidden state biases action strength.
        # Use individual hidden dimensions so that the controller is sensitive
        # to dimension ordering (required for shuffle-based epiphenomenal tests).
        h_move = float(self._hidden[0]) if self.hidden_dim > 0 else 0.0
        h_turn = float(self._hidden[1]) if self.hidden_dim > 1 else 0.0
        h_speed = float(self._hidden[2]) if self.hidden_dim > 2 else 0.0
        memory_signal = float(np.mean(memory_context[:2])) if len(memory_context) >= 2 else 0.0

        move_intent = float(np.clip(target_vector[0] + 0.1 * h_move, -1.0, 1.0))
        turn_intent = float(np.clip(target_vector[1] + 0.1 * (h_turn + memory_signal), -1.0, 1.0))
        speed_mod = float(np.clip(0.3 + 0.2 * h_speed, -1.0, 1.0))

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

    def intervene_state(self, delta: float | np.ndarray) -> None:
        """Surgically perturb the hidden state by *delta*.

        Used for causal intervention experiments — shifts the hidden state
        by a known amount to measure downstream behavioural divergence.
        """
        self._hidden = self._hidden + np.asarray(delta, dtype=np.float64)

    def shuffle_state(self, rng: np.random.Generator) -> None:
        """Randomly shuffle the hidden state dimensions.

        Used for epiphenomenal testing — if shuffling the hidden state
        has no effect on behaviour, the state is decorative.
        """
        rng.shuffle(self._hidden)
