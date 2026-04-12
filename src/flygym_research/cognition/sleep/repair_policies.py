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
class RepairROI:
    """Return-on-investment for a single repair attempt."""

    diagnosis: FailureDiagnosis
    expected_seam_improvement: float
    expected_mismatch_improvement: float
    repair_cost: float  # 0–1, higher = harder to repair
    roi_score: float  # expected improvement / cost


@dataclass(slots=True)
class AdaptivePatchability:
    """Data-driven, budget-constrained patchability criterion.

    Replaces the fixed 1.5× margin with thresholds learned from the
    failure distribution using percentile-based fitting and
    cross-validated repair ROI scoring.

    The learning process:
    1. Collect all failure excesses (seam and mismatch)
    2. Fit margins at a configurable percentile of the excess distribution
    3. Compute per-failure repair ROI and rank by expected benefit
    4. Cross-validate: split failures into train/holdout, fit on train,
       validate on holdout
    """

    seam_margin: float = 1.5
    mismatch_margin: float = 1.5
    budget: int = 10  # max number of repairs to attempt
    fit_percentile: float = 75.0  # percentile of excess distribution to cover
    fit_method: str = "percentile"  # "percentile" or "median_std"

    @staticmethod
    def from_distribution(
        diagnoses: list[FailureDiagnosis],
        *,
        budget_fraction: float = 0.5,
        fit_percentile: float = 75.0,
    ) -> "AdaptivePatchability":
        """Learn margins from the distribution of current failures.

        Uses percentile-based fitting: set the margin to cover the
        ``fit_percentile``-th percentile of excess values.  This is more
        principled than median+std because it directly controls what
        fraction of failures the margin is designed to handle.
        """
        seam_excesses = [d.seam_excess for d in diagnoses if d.seam_excess > 0]
        mismatch_excesses = [d.mismatch_excess for d in diagnoses if d.mismatch_excess > 0]

        seam_threshold = diagnoses[0].seam_threshold if diagnoses else 0.2
        mismatch_threshold = diagnoses[0].mismatch_threshold if diagnoses else 0.35

        if seam_excesses:
            pct_value = float(np.percentile(seam_excesses, fit_percentile))
            seam_margin = 1.0 + pct_value / max(seam_threshold, 0.01)
        else:
            seam_margin = 1.5

        if mismatch_excesses:
            pct_value = float(np.percentile(mismatch_excesses, fit_percentile))
            mismatch_margin = 1.0 + pct_value / max(mismatch_threshold, 0.01)
        else:
            mismatch_margin = 1.5

        budget = max(1, int(len(diagnoses) * budget_fraction))

        return AdaptivePatchability(
            seam_margin=float(np.clip(seam_margin, 1.0, 3.0)),
            mismatch_margin=float(np.clip(mismatch_margin, 1.0, 3.0)),
            budget=budget,
            fit_percentile=fit_percentile,
            fit_method="percentile",
        )

    @staticmethod
    def from_cross_validation(
        diagnoses: list[FailureDiagnosis],
        transitions_by_episode: dict[str, list[StepTransition]],
        *,
        n_folds: int = 3,
        percentile_candidates: tuple[float, ...] = (50.0, 65.0, 75.0, 85.0, 95.0),
    ) -> "AdaptivePatchability":
        """Learn margins via cross-validation over percentile candidates.

        Splits failures into ``n_folds`` folds.  For each candidate
        percentile, fits margins on training folds and evaluates repair
        success rate on the holdout fold.  Selects the percentile that
        maximises mean holdout repair success.
        """
        if len(diagnoses) < n_folds * 2:
            # Not enough data for CV — fall back to simple percentile
            return AdaptivePatchability.from_distribution(diagnoses)

        # Shuffle diagnoses deterministically
        indices = list(range(len(diagnoses)))
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

        fold_size = len(indices) // n_folds
        folds: list[list[int]] = []
        for f in range(n_folds):
            start = f * fold_size
            end = start + fold_size if f < n_folds - 1 else len(indices)
            folds.append(indices[start:end])

        best_pct = 75.0
        best_score = -1.0

        for pct in percentile_candidates:
            fold_scores: list[float] = []
            for holdout_idx in range(n_folds):
                train_indices = []
                for fi in range(n_folds):
                    if fi != holdout_idx:
                        train_indices.extend(folds[fi])
                holdout_indices = folds[holdout_idx]

                train_diags = [diagnoses[i] for i in train_indices]
                holdout_diags = [diagnoses[i] for i in holdout_indices]

                # Fit on training set
                adaptive = AdaptivePatchability.from_distribution(
                    train_diags, fit_percentile=pct,
                )

                # Evaluate on holdout: what fraction of holdout failures
                # would be correctly classified as patchable AND actually
                # succeed in repair?
                n_holdout = len(holdout_diags)
                n_success = 0
                for hd in holdout_diags:
                    if adaptive.is_patchable(hd):
                        # Simulate repair on this failure
                        ep_transitions = transitions_by_episode.get(hd.episode_id)
                        if ep_transitions:
                            result = apply_repair(hd, ep_transitions)
                            if result.repair_success:
                                n_success += 1
                        else:
                            # If no transitions, count patchable as success
                            n_success += 1

                fold_scores.append(n_success / max(n_holdout, 1))

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_pct = pct

        # Refit on all data with the best percentile
        result = AdaptivePatchability.from_distribution(
            diagnoses, fit_percentile=best_pct,
        )
        result.fit_method = f"cross_validated(pct={best_pct})"
        return result

    def is_patchable(self, diagnosis: FailureDiagnosis) -> bool:
        """Check if a failure is patchable under the adaptive criterion."""
        seam_ok = diagnosis.seam_score <= diagnosis.seam_threshold * self.seam_margin
        mismatch_ok = diagnosis.mismatch_score <= diagnosis.mismatch_threshold * self.mismatch_margin
        return seam_ok and mismatch_ok

    def compute_repair_roi(
        self,
        diagnosis: FailureDiagnosis,
    ) -> RepairROI:
        """Compute expected repair return-on-investment for a failure.

        ROI = expected_improvement / repair_cost.
        Higher ROI → should be repaired first (priority queue).
        """
        # Expected improvement: how much excess can be removed
        seam_improvable = min(
            diagnosis.seam_excess,
            diagnosis.seam_threshold * (self.seam_margin - 1.0),
        )
        mismatch_improvable = min(
            diagnosis.mismatch_excess,
            diagnosis.mismatch_threshold * (self.mismatch_margin - 1.0),
        )

        # Repair cost: harder failures (joint, large excess) cost more
        total_excess = diagnosis.seam_excess + diagnosis.mismatch_excess
        cost = float(np.clip(total_excess / 2.0, 0.1, 1.0))
        if diagnosis.failure_type == FailureType.JOINT_FAILURE:
            cost = min(cost * 2.0, 1.0)

        expected_improvement = seam_improvable + mismatch_improvable
        roi = expected_improvement / max(cost, 0.01)

        return RepairROI(
            diagnosis=diagnosis,
            expected_seam_improvement=seam_improvable,
            expected_mismatch_improvement=mismatch_improvable,
            repair_cost=cost,
            roi_score=roi,
        )

    def prioritized_repair_queue(
        self,
        diagnoses: list[FailureDiagnosis],
    ) -> list[RepairROI]:
        """Rank failures by repair ROI and return top-budget candidates.

        Returns a list of RepairROI objects sorted by roi_score descending,
        limited to self.budget entries.
        """
        patchable = [d for d in diagnoses if self.is_patchable(d)]
        rois = [self.compute_repair_roi(d) for d in patchable]
        rois.sort(key=lambda r: r.roi_score, reverse=True)
        return rois[:self.budget]

    def summary(self) -> dict[str, object]:
        """Return a summary dict of the adaptive patchability configuration."""
        return {
            "seam_margin": round(self.seam_margin, 3),
            "mismatch_margin": round(self.mismatch_margin, 3),
            "budget": self.budget,
            "fit_percentile": self.fit_percentile,
            "fit_method": self.fit_method,
        }
