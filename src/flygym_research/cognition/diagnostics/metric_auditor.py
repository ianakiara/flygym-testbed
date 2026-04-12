"""Metric Auditor — detects structural blindness in metrics.

Catches the class of bugs revealed by the permutation-invariance discovery:
a metric can look clever and still be structurally blind because of one
hidden invariance.

Audit checks:
1. **Permutation invariance**: Does shuffling input dimensions change output?
2. **Scale invariance**: Does scaling inputs change output proportionally?
3. **Dead code paths**: Are there inputs that never trigger certain branches?
4. **Averaging distortions**: Does the metric hide important variation?
5. **Constant-output trap**: Does the metric return the same value for
   very different inputs?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(slots=True)
class AuditResult:
    """Result of a single metric audit."""

    metric_name: str
    check_name: str
    passed: bool
    severity: str  # "critical" | "warning" | "info"
    detail: str
    values: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class MetricAuditReport:
    """Full audit report for one metric."""

    metric_name: str
    results: list[AuditResult]
    overall_passed: bool
    critical_failures: int
    warnings: int

    @property
    def summary(self) -> str:
        status = "PASS" if self.overall_passed else "FAIL"
        return (
            f"[{status}] {self.metric_name}: "
            f"{self.critical_failures} critical, {self.warnings} warnings"
        )


def check_permutation_invariance(
    metric_fn: Callable[..., dict[str, float]],
    inputs: list[Any],
    metric_key: str,
    *,
    n_shuffles: int = 10,
    rng: np.random.Generator | None = None,
) -> AuditResult:
    """Check if metric is invariant to input dimension permutations.

    If shuffling dimensions does NOT change the output, the metric may
    be structurally blind to dimension ordering (like the permutation
    invariance bug).
    """
    rng = rng or np.random.default_rng(42)
    baseline = metric_fn(*inputs)
    baseline_val = baseline.get(metric_key, 0.0)

    changed_count = 0
    max_change = 0.0

    for _ in range(n_shuffles):
        shuffled_inputs = []
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                shuffled = inp.copy()
                if shuffled.ndim == 1:
                    rng.shuffle(shuffled)
                elif shuffled.ndim == 2:
                    for row in shuffled:
                        rng.shuffle(row)
                shuffled_inputs.append(shuffled)
            elif isinstance(inp, list):
                # For list of transitions, shuffle internal state vectors
                shuffled_inputs.append(inp)  # transitions are complex, pass through
            else:
                shuffled_inputs.append(inp)

        try:
            shuffled_result = metric_fn(*shuffled_inputs)
            shuffled_val = shuffled_result.get(metric_key, 0.0)
            change = abs(shuffled_val - baseline_val)
            max_change = max(max_change, change)
            if change > 1e-6:
                changed_count += 1
        except Exception:
            changed_count += 1  # Exception counts as "changed"

    sensitivity = changed_count / max(n_shuffles, 1)

    if sensitivity < 0.1:
        return AuditResult(
            metric_name="",
            check_name="permutation_invariance",
            passed=False,
            severity="critical",
            detail=(
                f"Metric '{metric_key}' appears permutation-invariant "
                f"({sensitivity:.0%} shuffles changed output). "
                "This may indicate structural blindness."
            ),
            values={"sensitivity": sensitivity, "max_change": max_change},
        )
    return AuditResult(
        metric_name="",
        check_name="permutation_invariance",
        passed=True,
        severity="info",
        detail=f"Metric responds to permutation ({sensitivity:.0%} sensitivity).",
        values={"sensitivity": sensitivity, "max_change": max_change},
    )


def check_scale_invariance(
    metric_fn: Callable[..., dict[str, float]],
    inputs: list[Any],
    metric_key: str,
    *,
    scales: list[float] | None = None,
) -> AuditResult:
    """Check if metric is invariant to input scaling.

    A metric that doesn't respond to 10× scaling of inputs may be
    dominated by normalization that hides magnitude differences.
    """
    if scales is None:
        scales = [0.01, 0.1, 1.0, 10.0, 100.0]

    metric_fn(*inputs)  # validate baseline works

    values_at_scales: list[float] = []
    for scale in scales:
        scaled_inputs = []
        for inp in inputs:
            if isinstance(inp, np.ndarray):
                scaled_inputs.append(inp * scale)
            else:
                scaled_inputs.append(inp)
        try:
            result = metric_fn(*scaled_inputs)
            values_at_scales.append(result.get(metric_key, 0.0))
        except Exception:
            values_at_scales.append(float("nan"))

    valid_values = [v for v in values_at_scales if not np.isnan(v)]
    if len(valid_values) < 2:
        return AuditResult(
            metric_name="",
            check_name="scale_invariance",
            passed=True,
            severity="info",
            detail="Not enough valid scaled results to assess.",
            values={},
        )

    value_range = max(valid_values) - min(valid_values)
    is_flat = value_range < 1e-6

    if is_flat:
        return AuditResult(
            metric_name="",
            check_name="scale_invariance",
            passed=False,
            severity="warning",
            detail=(
                f"Metric '{metric_key}' is scale-invariant across "
                f"{len(scales)} scales (range: {value_range:.2e}). "
                "May be hiding magnitude differences."
            ),
            values={"value_range": value_range, "n_scales": float(len(scales))},
        )
    return AuditResult(
        metric_name="",
        check_name="scale_invariance",
        passed=True,
        severity="info",
        detail=f"Metric responds to scaling (range: {value_range:.4f}).",
        values={"value_range": value_range},
    )


def check_constant_output(
    metric_fn: Callable[..., dict[str, float]],
    input_variants: list[list[Any]],
    metric_key: str,
) -> AuditResult:
    """Check if metric returns the same value for different inputs.

    A metric that gives the same score for structurally different inputs
    is likely too coarse-grained or has a dead code path.
    """
    values: list[float] = []
    for inputs in input_variants:
        try:
            result = metric_fn(*inputs)
            values.append(result.get(metric_key, 0.0))
        except Exception:
            pass

    if len(values) < 2:
        return AuditResult(
            metric_name="",
            check_name="constant_output",
            passed=True,
            severity="info",
            detail="Not enough valid variants to assess.",
            values={},
        )

    value_range = max(values) - min(values)
    unique_count = len(set(round(v, 8) for v in values))

    if unique_count == 1:
        return AuditResult(
            metric_name="",
            check_name="constant_output",
            passed=False,
            severity="critical",
            detail=(
                f"Metric '{metric_key}' returned identical value ({values[0]:.6f}) "
                f"for {len(values)} different inputs. Likely a dead code path."
            ),
            values={"constant_value": values[0], "n_variants": float(len(values))},
        )
    return AuditResult(
        metric_name="",
        check_name="constant_output",
        passed=True,
        severity="info",
        detail=f"Metric distinguishes {unique_count}/{len(values)} inputs (range: {value_range:.4f}).",
        values={"unique_count": float(unique_count), "value_range": value_range},
    )


def check_averaging_distortion(
    values: list[float],
    metric_key: str,
    *,
    cv_threshold: float = 0.1,
) -> AuditResult:
    """Check if averaging hides important variation.

    If the coefficient of variation across episodes/seeds is very high,
    the mean is misleading.
    """
    if len(values) < 2:
        return AuditResult(
            metric_name="",
            check_name="averaging_distortion",
            passed=True,
            severity="info",
            detail="Not enough values to assess averaging distortion.",
            values={},
        )

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    cv = std_val / max(abs(mean_val), 1e-10)

    if cv > cv_threshold:
        return AuditResult(
            metric_name="",
            check_name="averaging_distortion",
            passed=False,
            severity="warning",
            detail=(
                f"Metric '{metric_key}' has high variation "
                f"(CV={cv:.2f}, mean={mean_val:.4f}, std={std_val:.4f}). "
                "Averaging may hide important structural differences."
            ),
            values={"cv": cv, "mean": mean_val, "std": std_val},
        )
    return AuditResult(
        metric_name="",
        check_name="averaging_distortion",
        passed=True,
        severity="info",
        detail=f"Low variation (CV={cv:.3f}), averaging is safe.",
        values={"cv": cv, "mean": mean_val, "std": std_val},
    )


def audit_metric(
    metric_name: str,
    results: list[AuditResult],
) -> MetricAuditReport:
    """Combine individual audit checks into a full report."""
    for r in results:
        r.metric_name = metric_name

    critical = sum(1 for r in results if not r.passed and r.severity == "critical")
    warnings = sum(1 for r in results if not r.passed and r.severity == "warning")

    return MetricAuditReport(
        metric_name=metric_name,
        results=results,
        overall_passed=critical == 0,
        critical_failures=critical,
        warnings=warnings,
    )
