"""Validation standards — negative controls, baseline comparisons, and
ablation survival documentation.

This module provides utilities for running the validation checks required
by Phase 9: every major claim must survive baseline comparison, ablation,
and negative control testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..experiments.benchmark_harness import run_episode
from ..interfaces import BrainInterface, StepTransition
from ..metrics import summarize_metrics


@dataclass(slots=True)
class ValidationResult:
    """Result of a single validation check."""

    check_name: str
    passed: bool
    baseline_value: float
    experimental_value: float
    threshold: float
    details: str = ""


@dataclass
class ValidationSuite:
    """Run validation checks for experimental claims.

    Implements the three-tier validation from the master prompt:
    - useful: beats simpler baseline + survives one ablation + reproducible
    - strong: survives multiple tasks + controller/world swaps + documented failures
    - promoted: cross-condition robust + negative controls fail correctly
    """

    results: list[ValidationResult] = field(default_factory=list)

    def check_beats_baseline(
        self,
        experimental_transitions: list[StepTransition],
        baseline_transitions: list[StepTransition],
        *,
        metric_key: str = "return",
        margin: float = 0.0,
    ) -> ValidationResult:
        """Check that the experimental condition beats the baseline."""
        exp_metrics = summarize_metrics(experimental_transitions)
        base_metrics = summarize_metrics(baseline_transitions)
        exp_val = exp_metrics.get(metric_key, 0.0)
        base_val = base_metrics.get(metric_key, 0.0)
        passed = exp_val > base_val + margin

        result = ValidationResult(
            check_name=f"beats_baseline_{metric_key}",
            passed=passed,
            baseline_value=base_val,
            experimental_value=exp_val,
            threshold=margin,
            details=(
                f"Experimental {metric_key}={exp_val:.4f} vs "
                f"baseline {metric_key}={base_val:.4f} (margin={margin})"
            ),
        )
        self.results.append(result)
        return result

    def check_ablation_survival(
        self,
        intact_transitions: list[StepTransition],
        ablated_transitions: list[StepTransition],
        *,
        metric_key: str = "return",
        max_degradation: float = 0.5,
    ) -> ValidationResult:
        """Check that ablation does not completely destroy the metric."""
        intact_metrics = summarize_metrics(intact_transitions)
        ablated_metrics = summarize_metrics(ablated_transitions)
        intact_val = intact_metrics.get(metric_key, 0.0)
        ablated_val = ablated_metrics.get(metric_key, 0.0)

        if abs(intact_val) < 1e-8:
            degradation = 0.0
        else:
            degradation = (intact_val - ablated_val) / abs(intact_val)
        passed = degradation < max_degradation

        result = ValidationResult(
            check_name=f"ablation_survival_{metric_key}",
            passed=passed,
            baseline_value=intact_val,
            experimental_value=ablated_val,
            threshold=max_degradation,
            details=(
                f"Degradation={degradation:.4f} (max allowed={max_degradation}). "
                f"Intact={intact_val:.4f}, ablated={ablated_val:.4f}"
            ),
        )
        self.results.append(result)
        return result

    def check_reproducibility(
        self,
        env_factory,
        controller: BrainInterface,
        *,
        seeds: list[int] | None = None,
        metric_key: str = "return",
        max_cv: float = 1.0,
    ) -> ValidationResult:
        """Check that results are reproducible across seeds (low coefficient of variation)."""
        seeds = seeds or [0, 1, 2, 3, 4]
        values: list[float] = []
        for seed in seeds:
            env = env_factory()
            transitions = run_episode(env, controller, seed=seed)
            metrics = summarize_metrics(transitions)
            values.append(metrics.get(metric_key, 0.0))

        arr = np.array(values, dtype=np.float64)
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        cv = std_val / (abs(mean_val) + 1e-8)
        passed = cv < max_cv

        result = ValidationResult(
            check_name=f"reproducibility_{metric_key}",
            passed=passed,
            baseline_value=mean_val,
            experimental_value=cv,
            threshold=max_cv,
            details=(
                f"CV={cv:.4f} across {len(seeds)} seeds "
                f"(mean={mean_val:.4f}, std={std_val:.4f}, max_cv={max_cv})"
            ),
        )
        self.results.append(result)
        return result

    def check_negative_control(
        self,
        negative_transitions: list[StepTransition],
        experimental_transitions: list[StepTransition],
        *,
        metric_key: str = "return",
    ) -> ValidationResult:
        """Check that negative control performs worse than experimental condition.

        Negative controls (random, shuffled) MUST fail — if they succeed,
        the metric or task is too easy.
        """
        neg_metrics = summarize_metrics(negative_transitions)
        exp_metrics = summarize_metrics(experimental_transitions)
        neg_val = neg_metrics.get(metric_key, 0.0)
        exp_val = exp_metrics.get(metric_key, 0.0)
        # Negative control should be strictly worse.
        passed = neg_val < exp_val

        result = ValidationResult(
            check_name=f"negative_control_{metric_key}",
            passed=passed,
            baseline_value=neg_val,
            experimental_value=exp_val,
            threshold=0.0,
            details=(
                f"Negative control {metric_key}={neg_val:.4f} vs "
                f"experimental {metric_key}={exp_val:.4f}. "
                f"{'PASS: negative correctly worse' if passed else 'FAIL: negative not worse — task may be trivial'}"
            ),
        )
        self.results.append(result)
        return result

    def summary(self) -> dict[str, Any]:
        """Return a summary of all validation results."""
        n_passed = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)
        return {
            "total_checks": n_total,
            "passed": n_passed,
            "failed": n_total - n_passed,
            "pass_rate": n_passed / max(n_total, 1),
            "checks": [
                {
                    "name": r.check_name,
                    "passed": r.passed,
                    "details": r.details,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        """Export validation results as markdown."""
        s = self.summary()
        lines = [
            "# Validation Results",
            "",
            f"**Pass rate**: {s['pass_rate']:.0%} ({s['passed']}/{s['total_checks']})",
            "",
            "| Check | Result | Details |",
            "| --- | --- | --- |",
        ]
        for check in s["checks"]:
            status = "PASS" if check["passed"] else "**FAIL**"
            lines.append(f"| {check['name']} | {status} | {check['details']} |")
        lines.append("")
        return "\n".join(lines)
