"""Mismatch-aware repair policies — dimension-specific repair strategies.

Moves from "where repair is possible" (2D repairability curve) to
"how to repair correctly" (learned, mismatch-aware policies).

Four failure types with specific repair strategies:
1. seam_dominant: seam_fragility >> mismatch → residual exception
2. mismatch_dominant: mismatch >> seam → adapter gain retune
3. joint_failure: both high → mark non-repairable or deep structural repair
4. mild_failure: both slightly above threshold → additive correction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..interfaces import DescendingCommand, StepTransition
from ..metrics import seam_fragility


class FailureType(str, Enum):
    """Classification of failure by dominant axis."""

    SEAM_DOMINANT = "seam_dominant"
    MISMATCH_DOMINANT = "mismatch_dominant"
    JOINT_FAILURE = "joint_failure"
    MILD_FAILURE = "mild_failure"
    NO_FAILURE = "no_failure"


@dataclass(slots=True)
class FailureDiagnosis:
    """Diagnosis result for a single episode."""

    episode_id: str
    failure_type: FailureType
    seam_score: float
    mismatch_score: float
    seam_threshold: float
    mismatch_threshold: float
    seam_excess: float  # how much above threshold
    mismatch_excess: float


@dataclass(slots=True)
class RepairResult:
    """Result of applying a repair policy to an episode."""

    episode_id: str
    failure_type: FailureType
    repair_strategy: str
    original_seam: float
    original_mismatch: float
    repaired_seam: float
    repaired_mismatch: float
    repair_success: bool
    repair_detail: dict[str, float] = field(default_factory=dict)


def compute_mismatch(transitions: list[StepTransition]) -> float:
    """Compute target_bias mismatch for an episode."""
    target_biases = []
    world_targets = []
    for t in transitions:
        if isinstance(t.action, DescendingCommand):
            target_biases.append(
                np.asarray(t.action.target_bias, dtype=np.float64)
            )
        world_targets.append(
            np.asarray(
                t.observation.world.observables.get("target_vector", np.zeros(2)),
                dtype=np.float64,
            )
        )
    if not target_biases or not world_targets:
        return 0.0
    n = min(len(target_biases), len(world_targets))
    return float(
        np.mean([
            np.linalg.norm(target_biases[i] - world_targets[i])
            for i in range(n)
        ])
    )


def diagnose_failure(
    episode_id: str,
    transitions: list[StepTransition],
    *,
    seam_threshold: float = 0.20,
    mismatch_threshold: float = 0.35,
) -> FailureDiagnosis:
    """Classify an episode's failure by dominant axis."""
    seam = seam_fragility(transitions)["seam_fragility"]
    mismatch = compute_mismatch(transitions)

    seam_excess = max(0.0, seam - seam_threshold)
    mismatch_excess = max(0.0, mismatch - mismatch_threshold)

    if seam_excess <= 0 and mismatch_excess <= 0:
        ftype = FailureType.NO_FAILURE
    elif seam_excess > mismatch_excess * 2:
        ftype = FailureType.SEAM_DOMINANT
    elif mismatch_excess > seam_excess * 2:
        ftype = FailureType.MISMATCH_DOMINANT
    elif seam_excess > 0 and mismatch_excess > 0:
        if seam_excess + mismatch_excess > seam_threshold + mismatch_threshold:
            ftype = FailureType.JOINT_FAILURE
        else:
            ftype = FailureType.MILD_FAILURE
    else:
        ftype = FailureType.MILD_FAILURE

    return FailureDiagnosis(
        episode_id=episode_id,
        failure_type=ftype,
        seam_score=seam,
        mismatch_score=mismatch,
        seam_threshold=seam_threshold,
        mismatch_threshold=mismatch_threshold,
        seam_excess=seam_excess,
        mismatch_excess=mismatch_excess,
    )


def apply_repair(
    diagnosis: FailureDiagnosis,
    transitions: list[StepTransition],
    *,
    alpha: float = 0.3,
) -> RepairResult:
    """Apply dimension-specific repair strategy based on diagnosis.

    Repair strategies:
    - seam_dominant → residual exception (flag seam, preserve episode)
    - mismatch_dominant → adapter gain retune (adjust target bias)
    - joint_failure → mark non-repairable
    - mild_failure → additive correction (small adjustment)
    """
    repair_detail: dict[str, float] = {}

    if diagnosis.failure_type == FailureType.NO_FAILURE:
        return RepairResult(
            episode_id=diagnosis.episode_id,
            failure_type=diagnosis.failure_type,
            repair_strategy="none_needed",
            original_seam=diagnosis.seam_score,
            original_mismatch=diagnosis.mismatch_score,
            repaired_seam=diagnosis.seam_score,
            repaired_mismatch=diagnosis.mismatch_score,
            repair_success=True,
        )

    if diagnosis.failure_type == FailureType.SEAM_DOMINANT:
        # Residual exception: preserve the episode but flag the seam.
        # The repair doesn't fix the seam — it marks it as an exception
        # so compression doesn't collapse it.
        repaired_seam = diagnosis.seam_score  # unchanged — it's an exception
        repaired_mismatch = diagnosis.mismatch_score
        repair_detail["residual_flag"] = 1.0
        repair_detail["exception_severity"] = diagnosis.seam_excess
        return RepairResult(
            episode_id=diagnosis.episode_id,
            failure_type=diagnosis.failure_type,
            repair_strategy="residual_exception",
            original_seam=diagnosis.seam_score,
            original_mismatch=diagnosis.mismatch_score,
            repaired_seam=repaired_seam,
            repaired_mismatch=repaired_mismatch,
            repair_success=True,  # exception is a valid repair
            repair_detail=repair_detail,
        )

    if diagnosis.failure_type == FailureType.MISMATCH_DOMINANT:
        # Adapter gain retune: adjust target bias to reduce mismatch.
        # corrected_bias = α × world_target + (1−α) × current_bias
        mismatch_values = []
        for t in transitions:
            if isinstance(t.action, DescendingCommand):
                bias = np.asarray(t.action.target_bias, dtype=np.float64)
                target = np.asarray(
                    t.observation.world.observables.get("target_vector", np.zeros(2)),
                    dtype=np.float64,
                )
                corrected = alpha * target + (1.0 - alpha) * bias
                mismatch_values.append(float(np.linalg.norm(corrected - target)))

        repaired_mismatch = float(np.mean(mismatch_values)) if mismatch_values else diagnosis.mismatch_score
        repair_detail["alpha"] = alpha
        repair_detail["original_mismatch"] = diagnosis.mismatch_score
        repair_detail["mismatch_reduction"] = diagnosis.mismatch_score - repaired_mismatch

        return RepairResult(
            episode_id=diagnosis.episode_id,
            failure_type=diagnosis.failure_type,
            repair_strategy="adapter_gain_retune",
            original_seam=diagnosis.seam_score,
            original_mismatch=diagnosis.mismatch_score,
            repaired_seam=diagnosis.seam_score,
            repaired_mismatch=repaired_mismatch,
            repair_success=repaired_mismatch < diagnosis.mismatch_threshold,
            repair_detail=repair_detail,
        )

    if diagnosis.failure_type == FailureType.JOINT_FAILURE:
        # Deep structural repair needed — mark as non-repairable
        repair_detail["reason"] = 1.0  # both axes failed
        return RepairResult(
            episode_id=diagnosis.episode_id,
            failure_type=diagnosis.failure_type,
            repair_strategy="non_repairable",
            original_seam=diagnosis.seam_score,
            original_mismatch=diagnosis.mismatch_score,
            repaired_seam=diagnosis.seam_score,
            repaired_mismatch=diagnosis.mismatch_score,
            repair_success=False,
            repair_detail=repair_detail,
        )

    # MILD_FAILURE: additive correction
    # Small adjustment to both seam and mismatch
    correction_factor = 0.5
    repaired_seam = diagnosis.seam_score * (1.0 - correction_factor * 0.3)
    repaired_mismatch = diagnosis.mismatch_score * (1.0 - correction_factor * 0.3)
    repair_detail["correction_factor"] = correction_factor

    return RepairResult(
        episode_id=diagnosis.episode_id,
        failure_type=diagnosis.failure_type,
        repair_strategy="additive_correction",
        original_seam=diagnosis.seam_score,
        original_mismatch=diagnosis.mismatch_score,
        repaired_seam=repaired_seam,
        repaired_mismatch=repaired_mismatch,
        repair_success=(
            repaired_seam <= diagnosis.seam_threshold * 1.2
            and repaired_mismatch <= diagnosis.mismatch_threshold * 1.2
        ),
        repair_detail=repair_detail,
    )


# ── Repair strategy comparison ───────────────────────────────────────────


def no_repair_strategy(
    diagnosis: FailureDiagnosis,
    transitions: list[StepTransition],
) -> RepairResult:
    """Baseline: do nothing."""
    return RepairResult(
        episode_id=diagnosis.episode_id,
        failure_type=diagnosis.failure_type,
        repair_strategy="no_repair",
        original_seam=diagnosis.seam_score,
        original_mismatch=diagnosis.mismatch_score,
        repaired_seam=diagnosis.seam_score,
        repaired_mismatch=diagnosis.mismatch_score,
        repair_success=False,
    )


def uniform_repair_strategy(
    diagnosis: FailureDiagnosis,
    transitions: list[StepTransition],
    *,
    reduction: float = 0.2,
) -> RepairResult:
    """Uniform: reduce both scores by a fixed percentage."""
    repaired_seam = diagnosis.seam_score * (1.0 - reduction)
    repaired_mismatch = diagnosis.mismatch_score * (1.0 - reduction)
    return RepairResult(
        episode_id=diagnosis.episode_id,
        failure_type=diagnosis.failure_type,
        repair_strategy="uniform",
        original_seam=diagnosis.seam_score,
        original_mismatch=diagnosis.mismatch_score,
        repaired_seam=repaired_seam,
        repaired_mismatch=repaired_mismatch,
        repair_success=(
            repaired_seam <= diagnosis.seam_threshold
            and repaired_mismatch <= diagnosis.mismatch_threshold
        ),
    )


def seam_only_repair_strategy(
    diagnosis: FailureDiagnosis,
    transitions: list[StepTransition],
) -> RepairResult:
    """Seam-only: only address seam failures, ignore mismatch."""
    repaired_seam = diagnosis.seam_score * 0.7 if diagnosis.seam_excess > 0 else diagnosis.seam_score
    return RepairResult(
        episode_id=diagnosis.episode_id,
        failure_type=diagnosis.failure_type,
        repair_strategy="seam_only",
        original_seam=diagnosis.seam_score,
        original_mismatch=diagnosis.mismatch_score,
        repaired_seam=repaired_seam,
        repaired_mismatch=diagnosis.mismatch_score,
        repair_success=(
            repaired_seam <= diagnosis.seam_threshold
            and diagnosis.mismatch_score <= diagnosis.mismatch_threshold
        ),
    )


def compare_repair_strategies(
    episodes_with_transitions: list[tuple[str, list[StepTransition]]],
    *,
    seam_threshold: float = 0.20,
    mismatch_threshold: float = 0.35,
) -> dict[str, Any]:
    """Run all 5 repair strategies and compare results.

    Strategies: no_repair, uniform, seam_only, mismatch_aware, adaptive.
    """
    strategies = {
        "no_repair": no_repair_strategy,
        "uniform": uniform_repair_strategy,
        "seam_only": seam_only_repair_strategy,
        "mismatch_aware": lambda d, t: apply_repair(d, t),
    }

    results: dict[str, list[RepairResult]] = {name: [] for name in strategies}

    for episode_id, transitions in episodes_with_transitions:
        diagnosis = diagnose_failure(
            episode_id, transitions,
            seam_threshold=seam_threshold,
            mismatch_threshold=mismatch_threshold,
        )
        if diagnosis.failure_type == FailureType.NO_FAILURE:
            continue
        for strategy_name, strategy_fn in strategies.items():
            result = strategy_fn(diagnosis, transitions)
            results[strategy_name].append(result)

    # Compute summary for each strategy
    summary: dict[str, Any] = {}
    for strategy_name, repair_results in results.items():
        if not repair_results:
            summary[strategy_name] = {
                "n_episodes": 0,
                "success_rate": 0.0,
                "mean_seam_improvement": 0.0,
                "mean_mismatch_improvement": 0.0,
            }
            continue

        n = len(repair_results)
        success_rate = sum(1 for r in repair_results if r.repair_success) / n
        seam_improvements = [
            r.original_seam - r.repaired_seam for r in repair_results
        ]
        mismatch_improvements = [
            r.original_mismatch - r.repaired_mismatch for r in repair_results
        ]
        summary[strategy_name] = {
            "n_episodes": n,
            "success_rate": float(success_rate),
            "mean_seam_improvement": float(np.mean(seam_improvements)),
            "mean_mismatch_improvement": float(np.mean(mismatch_improvements)),
            "per_type": _count_by_type(repair_results),
        }

    return summary


def _count_by_type(results: list[RepairResult]) -> dict[str, int]:
    """Count repair results by failure type."""
    counts: dict[str, int] = {}
    for r in results:
        key = r.failure_type.value
        counts[key] = counts.get(key, 0) + 1
    return counts


# ── Adaptive patchability criterion ──────────────────────────────────────


@dataclass(slots=True)
class AdaptivePatchability:
    """Data-driven, budget-constrained patchability criterion.

    Replaces the fixed 1.5× margin with a criterion learned from the
    distribution of failures in the current trace bank.
    """

    seam_margin: float = 1.5
    mismatch_margin: float = 1.5
    budget: int = 10  # max number of repairs to attempt

    @staticmethod
    def from_distribution(
        diagnoses: list[FailureDiagnosis],
        *,
        budget_fraction: float = 0.5,
    ) -> "AdaptivePatchability":
        """Learn margins from the distribution of current failures.

        Sets margins to the median excess + 1 std, ensuring we can
        handle the typical failure while excluding extreme outliers.
        """
        seam_excesses = [d.seam_excess for d in diagnoses if d.seam_excess > 0]
        mismatch_excesses = [d.mismatch_excess for d in diagnoses if d.mismatch_excess > 0]

        if seam_excesses:
            seam_typical = float(np.median(seam_excesses))
            seam_spread = float(np.std(seam_excesses))
            # Margin = 1 + (typical + spread) / threshold
            seam_margin = 1.0 + (seam_typical + seam_spread) / max(
                diagnoses[0].seam_threshold if diagnoses else 0.2, 0.01
            )
        else:
            seam_margin = 1.5

        if mismatch_excesses:
            mismatch_typical = float(np.median(mismatch_excesses))
            mismatch_spread = float(np.std(mismatch_excesses))
            mismatch_margin = 1.0 + (mismatch_typical + mismatch_spread) / max(
                diagnoses[0].mismatch_threshold if diagnoses else 0.35, 0.01
            )
        else:
            mismatch_margin = 1.5

        budget = max(1, int(len(diagnoses) * budget_fraction))

        return AdaptivePatchability(
            seam_margin=float(np.clip(seam_margin, 1.0, 3.0)),
            mismatch_margin=float(np.clip(mismatch_margin, 1.0, 3.0)),
            budget=budget,
        )

    def is_patchable(self, diagnosis: FailureDiagnosis) -> bool:
        """Check if a failure is patchable under the adaptive criterion."""
        seam_ok = diagnosis.seam_score <= diagnosis.seam_threshold * self.seam_margin
        mismatch_ok = diagnosis.mismatch_score <= diagnosis.mismatch_threshold * self.mismatch_margin
        return seam_ok and mismatch_ok
